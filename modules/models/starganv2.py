"""
StarGAN v2 pour harmonisation PET multi-sites.
===============================================

Architecture :
  StyleEncoder          : CNN + FFT(x) → style_code (B, style_dim)
                          Global Average Pooling → pas d'info spatiale/anatomique
  Generator             : UNet existant conditionné par style via cond_embedder
  StarGANv2Discriminator: PatchGAN avec une tête par domaine

Résolution du problème patch-par-patch :
  La signature scanner est un biais global de calibration matériel,
  indépendant de l'anatomie locale. Un patch de foie du site A peut
  apprendre le style d'un patch de cerveau du site B.
  En inférence : moyenne des style codes sur N patches du volume référence
  → un seul vecteur style appliqué uniformément à tous les patches source.

Losses :
  L_adv  : adversariale BCE + R1 gradient penalty (stabilité D)
  L_sty  : reconstruction style — E(G(x, s_tgt)) ≈ s_tgt
  L_cyc  : cohérence cyclique — G(G(x, s_tgt), s_src) ≈ x
  L_rec  : auto-reconstruction warmup — G(x, E(x)) ≈ x
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import wandb
from pytorch_lightning import LightningModule
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid

from modules.models.fft import LearnableFFTHighPassFilter
from modules.models.unet import UNet

from modules.models.base import (
    BasicBlock,
    BasicDown,
    BasicUp,
    UnetBasicBlock,
    UnetResBlock,
    SequentialEmb,
    save_add,
)
from modules.models.attention import Attention, zero_module
from monai.networks.blocks import UnetOutBlock

# =============================================================================
# AdaINResBlock
# =============================================================================

class AdaINResBlock(nn.Module):
    """
    ResBlock with Adaptive Instance Normalization.
    
    For each normalization:
      1. InstanceNorm2d normalizes the feature map → mu=0, std=1 per channel
      2. Two linear projections from style_emb predict gamma and beta
      3. output = (1 + gamma) * normalized + beta
         (gamma centered on 0 → neutral behavior at initialization)
    
    Parameters
    ----------
    in_channels  : int
    out_channels : int
    style_dim    : int  — dimension of style_emb (= style_embedding_dim)
    dropout      : float
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        style_dim:    int,
        dropout:      float = 0.0,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels,  out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=False)
        self.act   = nn.SiLU()
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Style → (gamma, beta) projections for each normalization
        # gamma centered on 0 (we add 1 in _adain) → neutral at init
        self.adain1 = nn.Linear(style_dim, out_channels * 2)
        self.adain2 = nn.Linear(style_dim, out_channels * 2)

        # Init: gamma=0, beta=0 → identity mapping at start
        nn.init.zeros_(self.adain1.weight)
        nn.init.zeros_(self.adain1.bias)
        nn.init.zeros_(self.adain2.weight)
        nn.init.zeros_(self.adain2.bias)

        # Skip connection if channel change
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def _adain(
        self,
        x:         torch.Tensor,   # (B, C, H, W) already normalized by InstanceNorm
        style_emb: torch.Tensor,   # (B, style_dim)
        proj:      nn.Linear,
    ) -> torch.Tensor:
        params        = proj(style_emb)                             # (B, 2*C)
        gamma, beta   = params.chunk(2, dim=1)                      # (B, C) each
        gamma         = gamma.unsqueeze(-1).unsqueeze(-1)           # (B, C, 1, 1)
        beta          = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * x + beta                             # affine modulation

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self._adain(self.norm1(h), emb, self.adain1)
        h = self.act(h)
        h = self.drop(h)

        h = self.conv2(h)
        h = self._adain(self.norm2(h), emb, self.adain2)
        h = self.act(h)

        return h + self.skip(x)

# =============================================================================
# StarGANv2Generator
# =============================================================================

class StarGANv2Generator(nn.Module):
    """
    StarGAN v2 Generator for PET Harmonization with proper skip connections.
    
    Architecture:
      - Input conv: x → initial features
      - Encoder: downsampling with AdaIN blocks + skip connections
      - Middle: AdaIN → Attention → AdaIN (fixed, no loop)
      - Decoder: upsampling with AdaIN blocks + skip concatenation
      - Output conv: features → x_fake
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 5,
        hidden_channels: List[int] = [64, 128, 256, 512],
        kernel_sizes: List[int] = [3, 3, 3, 3],
        strides: List[int] = [1, 2, 2, 2],
        style_embedding_dim: int = 256,
        num_residual_blocks: int = 2,
        spatial_dims: int = 2,
        normalization: Tuple = ('group', {'num_groups': 32, 'affine': True}),
        activation: Tuple = ('swish', {}),
        dropout: float = 0.0,
        learnable_interpolation: bool = True,
        attention_type: Union[str, List[str]] = 'none',
    ):
        super().__init__()

        self.depth = len(hidden_channels)
        self.num_residual_blocks = num_residual_blocks
        self.style_embedding_dim = style_embedding_dim

        attention_type = (
            attention_type if isinstance(attention_type, list)
            else [attention_type] * self.depth
        )

        # ── Input Convolution ─────────────────────────────────────────────────
        self.input_conv = BasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=hidden_channels[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
        )

        # ── Encoder ────────────────────────────────────────────────────────────
        encoder_blocks = []
        for i in range(1, self.depth):
            for k in range(num_residual_blocks):
                seq_list = []
                in_ch = hidden_channels[i - 1] if k == 0 else hidden_channels[i]
                out_ch = hidden_channels[i]
                
                seq_list.append(
                    AdaINResBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        style_dim=style_embedding_dim,
                        dropout=dropout,
                    )
                )
                
                seq_list.append(
                    Attention(
                        spatial_dims=spatial_dims,
                        in_channels=out_ch,
                        out_channels=out_ch,
                        num_heads=8,
                        ch_per_head=max(1, out_ch // 8),
                        depth=1,
                        norm_name=normalization,
                        dropout=dropout,
                        emb_dim=None,
                        attention_type=attention_type[i],
                    )
                )
                encoder_blocks.append(SequentialEmb(*seq_list))

            if i < self.depth - 1:
                encoder_blocks.append(
                    BasicDown(
                        spatial_dims=spatial_dims,
                        in_channels=hidden_channels[i],
                        out_channels=hidden_channels[i],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        learnable_interpolation=learnable_interpolation,
                    )
                )

        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        # ── Middle blocks: AdaIN → Attention → AdaIN ───────────────────────────
        self.middle_block = SequentialEmb(
            AdaINResBlock(
                in_channels=hidden_channels[-1],
                out_channels=hidden_channels[-1],
                style_dim=style_embedding_dim,
                dropout=dropout,
            ),
            Attention(
                spatial_dims=spatial_dims,
                in_channels=hidden_channels[-1],
                out_channels=hidden_channels[-1],
                num_heads=8,
                ch_per_head=max(1, hidden_channels[-1] // 8),
                depth=1,
                norm_name=normalization,
                dropout=dropout,
                emb_dim=None,
                attention_type=attention_type[-1],
            ),
            AdaINResBlock(
                in_channels=hidden_channels[-1],
                out_channels=hidden_channels[-1],
                style_dim=style_embedding_dim,
                dropout=dropout,
            ),
        )

        # ── Decoder ────────────────────────────────────────────────────────────
        decoder_blocks = []
        for i in range(1, self.depth):
            for k in range(num_residual_blocks + 1):
                seq_list = []
                out_ch = hidden_channels[i - 1] if k == 0 else hidden_channels[i]
                in_ch = hidden_channels[i] + out_ch
                
                seq_list.append(
                    AdaINResBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        style_dim=style_embedding_dim,
                        dropout=dropout,
                    )
                )

                seq_list.append(
                    Attention(
                        spatial_dims=spatial_dims,
                        in_channels=out_ch,
                        out_channels=out_ch,
                        num_heads=8,
                        ch_per_head=max(1, out_ch // 8),
                        depth=1,
                        norm_name=normalization,
                        dropout=dropout,
                        emb_dim=None,
                        attention_type=attention_type[i],
                    )
                )

                if (i > 1) and k == 0:
                    seq_list.append(
                        BasicUp(
                            spatial_dims=spatial_dims,
                            in_channels=out_ch,
                            out_channels=out_ch,
                            kernel_size=strides[i],
                            stride=strides[i],
                            learnable_interpolation=learnable_interpolation,
                        )
                    )

                decoder_blocks.append(SequentialEmb(*seq_list))

        self.decoder_blocks = nn.ModuleList(decoder_blocks)

        # ── Output Convolution ─────────────────────────────────────────────────
        self.output_conv = zero_module(
            UnetOutBlock(spatial_dims, hidden_channels[0], out_channels, dropout=None)
        )

    def forward(
        self,
        x: torch.Tensor,
        style_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with proper skip connection handling.
        
        Parameters
        ----------
        x : torch.Tensor
            Input image (B, C, H, W)
        style_emb : torch.Tensor
            Style embedding (B, style_embedding_dim)
        
        Returns
        -------
        torch.Tensor
            Generated/translated image (B, C, H, W)
        """
        # ── Input ──────────────────────────────────────────────────────────────
        h = self.input_conv(x)
        x_skips = [h]

        # ── Encoder ────────────────────────────────────────────────────────────
        for i in range(len(self.encoder_blocks)):
            h = self.encoder_blocks[i](h, style_emb)
            x_skips.append(h)

        # ── Middle ─────────────────────────────────────────────────────────────
        h = self.middle_block(h, style_emb)

        # ── Decoder (reverse loop) ─────────────────────────────────────────────
        for i in range(len(self.decoder_blocks) - 1, -1, -1):
            h = torch.cat([h, x_skips.pop()], dim=1)
            h = self.decoder_blocks[i](h, style_emb)

        # ── Output ─────────────────────────────────────────────────────────────
        y = self.output_conv(h)

        return y

# =============================================================================
# StyleEncoder
# =============================================================================

class StyleEncoder(nn.Module):
    """
    Encode un patch en un vecteur de style global.

    Entrée : [x, FFT(x)] concaténés → capte la signature spectrale du scanner.
    Sortie : style_code (B, style_dim) via Global Average Pooling.

    Le GAP supprime toute information spatiale/anatomique — seules les
    statistiques globales (luminosité, contraste, bruit scanner) sont retenues.
    """

    def __init__(
        self,
        input_shape:     Tuple[int, int],
        in_channels:     int = 5,
        style_dim:       int = 64,
        hidden_channels: List[int] = [64, 128, 256, 512],
        fft_sigma:       float = 7.5,
    ):
        super().__init__()

        self.fft_filter = LearnableFFTHighPassFilter(
            input_shape, 
            in_channels=in_channels,
            sigma=fft_sigma
        )

        # Backbone CNN : accepte 2 * in_channels (x + fft_x concaténés)
        layers = [
            nn.Conv2d(in_channels * 2, hidden_channels[0], kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_channels[0]),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for i in range(1, len(hidden_channels)):
            layers += [
                nn.Conv2d(hidden_channels[i-1], hidden_channels[i], kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(hidden_channels[i]),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        self.backbone = nn.Sequential(*layers)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.flatten  = nn.Flatten()
        self.fc       = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1]),
            nn.SiLU(),
            nn.Linear(hidden_channels[-1], style_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C, H, W) → style_code : (B, style_dim)"""
        fft_x = self.fft_filter(x)
        h = self.backbone(torch.cat([x, fft_x], dim=1))
        return self.fc(self.flatten(self.pool(h)))
    
    
class StyleEmbedder(nn.Module):
    """
    Projects 1D style code (B, style_channels) to embedding (B, style_embedding_dim)
    used as AdaIN condition throughout the generator.
    
    Same role and structure as the time_embedder in diffusion models:
        Linear(style_channels → style_embedding_dim) → SiLU → Linear(...)
    
    Parameters
    ----------
    style_channels : int
        Dimension of the input style code (output of StyleEncoder).
    style_embedding_dim : int
        Output embedding dimension (= style_dim passed to AdaINResBlock).
    """

    def __init__(self, style_channels: int = 256, style_embedding_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(style_channels, style_embedding_dim, bias=False),
            nn.SiLU(),
            nn.Linear(style_embedding_dim, style_embedding_dim, bias=False),
        )

    def forward(self, z_style: torch.Tensor) -> torch.Tensor:
        """z_style : (B, style_channels) → style_emb : (B, style_embedding_dim)"""
        return self.net(z_style)


# =============================================================================
# Discriminator
# =============================================================================

class StarGANv2Discriminator(nn.Module):
    """
    PatchGAN multi-domaines.

    Backbone CNN partagé → features → num_domains têtes linéaires indépendantes.
    D_d(x) prédit si x est une vraie image du domaine d.

    Usage :
        scores = discriminator(x, domain_ids)  # (B, 1)
    """

    def __init__(
        self,
        in_channels:     int = 5,
        num_domains:     int = 3,
        hidden_channels: List[int] = [64, 128, 256, 512],
    ):
        super().__init__()

        self.num_domains = num_domains

        layers = [
            nn.Conv2d(in_channels, hidden_channels[0], kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for i in range(1, len(hidden_channels)):
            layers += [
                nn.Conv2d(hidden_channels[i-1], hidden_channels[i], kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(hidden_channels[i]),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        self.backbone = nn.Sequential(*layers)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.flatten  = nn.Flatten()

        # Une tête de classification par domaine
        self.heads = nn.ModuleList([
            nn.Linear(hidden_channels[-1], 1) for _ in range(num_domains)
        ])

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.pool(self.backbone(x)))

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        """
        x          : (B, C, H, W)
        domain_ids : (B,) LongTensor ∈ {0, …, num_domains-1}
        Returns    : (B, 1) scores (logits, pas de sigmoid)
        """
        features   = self.get_features(x)                                            # (B, feat_dim)
        all_scores = torch.stack([h(features) for h in self.heads], dim=1)           # (B, num_domains, 1)
        d_onehot   = F.one_hot(domain_ids, self.num_domains).float().unsqueeze(-1)   # (B, num_domains, 1)
        return (all_scores * d_onehot).sum(dim=1)                                     # (B, 1)


# =============================================================================
# StarGANv2 — Lightning module
# =============================================================================

class StarGANv2(LightningModule):
    """
    Pipeline StarGAN v2 pour harmonisation PET multi-sites.

    Le générateur est le UNet existant. Le style est injecté via le mécanisme
    cond_embedder du UNet (Linear style_dim → style_embedding_dim), qui alimente
    le scale-shift norm de tous les blocs.

    Stage 1 (warmup) :
        G(x, E(x)) ≈ x — auto-reconstruction pure, pas d'adversarial.
        Stabilise G et E avant d'introduire D.

    Stage 2 :
        D update : D(real, d_src) → 1, D(G(x, s_tgt), d_tgt) → 0 + R1
        G update : adversarial + style_recon + cycle + recon

    Inférence :
        Extraire style moyen sur N patches du volume référence (méthode harmonize).
        Appliquer G(x_patch, s_ref_mean) pour chaque patch source.
    """

    def __init__(
        self,
        generator:          StarGANv2Generator,
        style_encoder:      StyleEncoder,
        style_embedder:     StyleEmbedder,
        discriminator:      StarGANv2Discriminator,
        num_domains:        int   = 3,
        style_dim:          int   = 64,
        style_embedding_dim: int  = 256,
        warmup_epochs:      int   = 5,
        lambda_sty:         float = 1.0,
        lambda_cyc:         float = 1.0,
        lambda_rec:         float = 1.0,
        lambda_r1:          float = 1.0,
        r1_every:           int   = 16,
        lr_G:               float = 1e-4,
        lr_D:               float = 1e-4,
        weight_decay:       float = 1e-5,
        suv_global_log_max: float = 6.0,
    ):
        super().__init__()

        self.generator      = generator
        self.style_encoder  = style_encoder
        self.style_embedder = style_embedder
        self.discriminator  = discriminator
        self.num_domains    = num_domains
        self.warmup_epochs  = warmup_epochs
        self.lambda_sty     = lambda_sty
        self.lambda_cyc     = lambda_cyc
        self.lambda_rec     = lambda_rec
        self.lambda_r1      = lambda_r1
        self.r1_every       = r1_every
        self.lrs            = {"G": lr_G, "D": lr_D}
        self.weight_decay   = weight_decay
        self.suv_global_log_max = suv_global_log_max

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.automatic_optimization = False

        self.save_hyperparameters(
            ignore=["generator", "style_encoder", "style_embedder", "discriminator"]
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Utilitaires
    # ──────────────────────────────────────────────────────────────────────────

    def _normalize(self, suv: torch.Tensor) -> torch.Tensor:
        log = torch.log1p(suv)
        return 2.0 * (log.clamp(0, self.suv_global_log_max) / self.suv_global_log_max) - 1.0

    def _denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        log = 0.5 * (x_norm.clamp(-1, 1) + 1.0) * self.suv_global_log_max
        return torch.expm1(log)

    def _log_dict(self, d: dict, batch_size: int):
        for k, v in d.items():
            self.log(k, v, prog_bar=True, sync_dist=True,
                     on_step=True, on_epoch=True, batch_size=batch_size)

    def _generate(self, x: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        """G(x, style_code) — style injecté via cond_embedder du UNet."""
        style_emb = self.style_embedder(style_code)
        return self.generator(x, style_emb)

    def _sample_cross_domain_ref(
        self, x: torch.Tensor, domain_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pour chaque sample i, tire aléatoirement un sample j avec d_j ≠ d_i.
        Clé : les patches ne sont PAS alignés spatialement — c'est voulu.
        Retourne (x_ref, d_tgt).
        """
        B = x.shape[0]
        x_ref = torch.zeros_like(x)
        d_tgt = domain_ids.clone()

        for i in range(B):
            d_src = domain_ids[i].item()
            cands = [j for j in range(B) if domain_ids[j].item() != d_src]
            if not cands:
                cands = [j for j in range(B) if j != i]
            if cands:
                j         = random.choice(cands)
                x_ref[i]  = x[j].detach()
                d_tgt[i]  = domain_ids[j]

        return x_ref, d_tgt

    def _r1_penalty(
        self, real: torch.Tensor, domain_ids: torch.Tensor
    ) -> torch.Tensor:
        """R1 gradient penalty sur les vraies images pour stabiliser D."""
        real_req = real.detach().requires_grad_(True)
        scores   = self.discriminator(real_req, domain_ids)
        grad     = torch.autograd.grad(
            outputs=scores.sum(),
            inputs=real_req,
            create_graph=True,
        )[0]
        return grad.pow(2).reshape(real.shape[0], -1).sum(1).mean()

    # ──────────────────────────────────────────────────────────────────────────
    # configure_optimizers
    # ──────────────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        """
        Two optimizers:
        - opt_G: Generator + StyleEncoder + StyleEmbedder
        - opt_D: Discriminator
        """
        opt_G = torch.optim.AdamW(
            list(self.generator.parameters())
            + list(self.style_encoder.parameters())
            + list(self.style_embedder.parameters()),
            lr=self.lrs["G"],
            betas=(0.0, 0.99),  # GAN-recommended betas
            weight_decay=self.weight_decay,
        )
        opt_D = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.lrs["D"],
            betas=(0.0, 0.99),
            weight_decay=self.weight_decay,
        )
        return [opt_G, opt_D]

    # ──────────────────────────────────────────────────────────────────────────
    # training_step
    # ──────────────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        opt_G, opt_D = self.optimizers()

        suv_source = batch["source"][tio.DATA].float()
        domain_ids = batch["domain_id"]

        if suv_source.ndim == 5:
            suv_source = suv_source.squeeze(1)

        x         = self._normalize(suv_source)
        bs        = x.shape[0]
        is_warmup = self.current_epoch < self.warmup_epochs

        # ══════════════════════════════════════════════════════════════════════
        # STAGE 1 — Warmup : G(x, E(x)) ≈ x, pas d'adversarial
        # ══════════════════════════════════════════════════════════════════════
        if is_warmup:
            s_src    = self.style_encoder(x)
            x_rec    = self._generate(x, s_src)
            loss_rec = F.l1_loss(x_rec, x)

            opt_G.zero_grad()
            self.manual_backward(loss_rec)
            opt_G.step()

            self._log_dict({"train/rec_loss_warmup": loss_rec}, bs)
            return

        # ══════════════════════════════════════════════════════════════════════
        # STAGE 2 — Entraînement adversarial complet
        # ══════════════════════════════════════════════════════════════════════

        x_ref, d_tgt = self._sample_cross_domain_ref(x, domain_ids)

        # ── Étape D ───────────────────────────────────────────────────────────
        with torch.no_grad():
            s_tgt_detached = self.style_encoder(x_ref)
            x_fake_detached = self._generate(x, s_tgt_detached)

        loss_D_real = F.binary_cross_entropy_with_logits(
            self.discriminator(x, domain_ids),
            torch.ones(bs, 1, device=x.device),
        )
        loss_D_fake = F.binary_cross_entropy_with_logits(
            self.discriminator(x_fake_detached, d_tgt),
            torch.zeros(bs, 1, device=x.device),
        )
        loss_D = (loss_D_real + loss_D_fake) / 2

        # R1 penalty (calcul périodique pour économiser la mémoire)
        loss_r1 = torch.tensor(0.0, device=x.device)
        if batch_idx % self.r1_every == 0:
            loss_r1 = self._r1_penalty(x, domain_ids)
            loss_D  = loss_D + self.lambda_r1 * loss_r1

        opt_D.zero_grad()
        self.manual_backward(loss_D)
        opt_D.step()

        # ── Étape G ───────────────────────────────────────────────────────────
        # Recalcul avec gradient pour G et E
        s_tgt    = self.style_encoder(x_ref)
        s_src    = self.style_encoder(x)
        x_fake   = self._generate(x, s_tgt)

        # Adversariale : D(G(x, s_tgt), d_tgt) → 1
        loss_G_adv = F.binary_cross_entropy_with_logits(
            self.discriminator(x_fake, d_tgt),
            torch.ones(bs, 1, device=x.device),
        )

        # Style reconstruction : E(G(x, s_tgt)) ≈ s_tgt
        s_tgt_hat  = self.style_encoder(x_fake)
        loss_sty   = F.l1_loss(s_tgt_hat, s_tgt.detach())

        # Cohérence cyclique : G(G(x, s_tgt), s_src) ≈ x
        x_cyc    = self._generate(x_fake, s_src)
        loss_cyc = F.l1_loss(x_cyc, x)

        # Auto-reconstruction : G(x, E(x)) ≈ x
        x_rec    = self._generate(x, s_src)
        loss_rec = F.l1_loss(x_rec, x)

        loss_G = (
            loss_G_adv
            + self.lambda_sty * loss_sty
            + self.lambda_cyc * loss_cyc
            + self.lambda_rec * loss_rec
        )

        opt_G.zero_grad()
        self.manual_backward(loss_G)
        torch.nn.utils.clip_grad_norm_(
            list(self.generator.parameters())
            + list(self.style_encoder.parameters())
            + list(self.style_embedder.parameters()),
            max_norm=1.0,
        )
        opt_G.step()

        self._log_dict({
            "train/D_real":    loss_D_real,
            "train/D_fake":    loss_D_fake,
            "train/D_r1":      loss_r1,
            "train/G_adv":     loss_G_adv,
            "train/sty_recon": loss_sty,
            "train/cyc":       loss_cyc,
            "train/rec":       loss_rec,
            "train/G_total":   loss_G,
        }, bs)

    # ──────────────────────────────────────────────────────────────────────────
    # validation_step
    # ──────────────────────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        suv_source = batch["source"][tio.DATA].float()
        domain_ids = batch["domain_id"]

        if suv_source.ndim == 5:
            suv_source = suv_source.squeeze(1)

        x  = self._normalize(suv_source)
        bs = x.shape[0]

        s_src = self.style_encoder(x)

        # Auto-reconstruction
        x_rec    = self._generate(x, s_src)
        loss_rec = F.l1_loss(x_rec, x)

        # SSIM
        x_01   = (x.clamp(-1, 1) + 1.0) / 2.0
        rec_01 = (x_rec.clamp(-1, 1) + 1.0) / 2.0
        ssim   = self.ssim(rec_01, x_01)

        # Translation cross-domain + style reconstruction
        x_ref, d_tgt = self._sample_cross_domain_ref(x, domain_ids)
        s_tgt         = self.style_encoder(x_ref)
        x_fake        = self._generate(x, s_tgt)
        s_tgt_hat     = self.style_encoder(x_fake)
        loss_sty      = F.l1_loss(s_tgt_hat, s_tgt)

        # Cycle
        x_cyc    = self._generate(x_fake, s_src)
        loss_cyc = F.l1_loss(x_cyc, x)

        composite = loss_rec + 0.1 * loss_sty + 0.1 * loss_cyc

        self._log_dict({
            "val/rec_loss":        loss_rec,
            "val/ssim":            ssim,
            "val/sty_recon":       loss_sty,
            "val/cyc":             loss_cyc,
            "val/composite_score": composite,
        }, bs)

        if batch_idx == 0:
            self._log_images(x, x_rec, x_fake, suv_source)

        return loss_rec

    # ──────────────────────────────────────────────────────────────────────────
    # Inférence
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def harmonize(
        self,
        x_source:       torch.Tensor,
        z_style_fixed:  Optional[torch.Tensor] = None,
        x_style_ref:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Harmonise x_source vers un style cible.

        Priorité : z_style_fixed > x_style_ref > style propre (reconstruction)

        Parameters
        ----------
        x_source      : (B, C, H, W) patch source normalisé [-1, 1]
        z_style_fixed : (1, style_dim) style moyen précalculé sur le volume référence
        x_style_ref   : (B, C, H, W) patch de référence (style extrait à la volée)
        """
        if z_style_fixed is not None:
            style_code = z_style_fixed.expand(x_source.shape[0], -1)
        elif x_style_ref is not None:
            style_code = self.style_encoder(x_style_ref)
        else:
            style_code = self.style_encoder(x_source)   # reconstruction

        return self._generate(x_source, style_code)

    # ──────────────────────────────────────────────────────────────────────────
    # Logging images
    # ──────────────────────────────────────────────────────────────────────────

    def _log_images(
        self,
        x:          torch.Tensor,
        x_rec:      torch.Tensor,
        x_fake:     torch.Tensor,
        suv_source: torch.Tensor,
    ):
        if self.trainer.global_rank != 0:
            return

        suv_rec  = self._denormalize(x_rec)
        suv_fake = self._denormalize(x_fake)
        mid      = suv_source.shape[1] // 2

        src_s  = suv_source[:, mid:mid+1, :, :]
        rec_s  = suv_rec[:,   mid:mid+1, :, :]
        fake_s = suv_fake[:,  mid:mid+1, :, :]

        display_max = max(5.0, src_s.max().item())
        imgs = torch.cat([src_s, rec_s, fake_s], dim=3)
        imgs = (imgs / display_max).clamp(0, 1)

        grid = make_grid(imgs, nrow=1, padding=2)
        wandb.log({
            "Validation/Images": wandb.Image(
                grid.permute(1, 2, 0).cpu().numpy(),
                caption=f"Source | Reconstruction | Traduit cross-domain (epoch {self.current_epoch})"
            )
        })