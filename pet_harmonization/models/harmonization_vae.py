import os
from typing import Dict, List, Optional, Tuple, Union
import math
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import numpy as np
import wandb
from pytorch_lightning import LightningModule
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid

from monai.networks.blocks import UnetOutBlock
from pet_harmonization.models.base import (
    BasicBlock,
    BasicDown,
    BasicUp,
    UnetBasicBlock,
    UnetResBlock,
    SequentialEmb,
    save_add,
)

from pet_harmonization.models.attention import Attention, zero_module
from pet_harmonization.models.fft import FFTHighPassFilter, LearnableFFTHighPassFilter


# =============================================================================
# Utilitaires
# =============================================================================

class SobelFilter(nn.Module):
    """
    Filtre de Sobel 2D channel-wise à poids fixes (aucun gradient).
    Retourne la magnitude du gradient.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        # (in_channels, 1, 3, 3) — convolution dépthwise
        kx = kx.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        ky = ky.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_3d = x.dim() == 5
        if is_3d:
            b, c, z, h, w = x.shape
            # (B, C, Z, H, W) -> (B, Z, C, H, W) -> (B*Z, C, H, W)
            x = x.transpose(1, 2).reshape(b * z, c, h, w)
            
        gx = F.conv2d(x, self.kx, padding=1, groups=self.in_channels)
        gy = F.conv2d(x, self.ky, padding=1, groups=self.in_channels)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        
        if is_3d:
            # (B*Z, C, H, W) -> (B, Z, C, H, W) -> (B, C, Z, H, W)
            mag = mag.reshape(b, z, c, h, w).transpose(1, 2)
            
        return mag


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Echantillonnage reparamétrisé : z = mu + eps * std."""
    logvar = torch.clamp(logvar, min=-30.0, max=20.0)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mu)
    return mu + eps * std


def kl_loss_spatial(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q||N(0,I)) pour un posterior spatial (B, C, H, W). Retourne [B]."""
    logvar = torch.clamp(logvar, min=-30.0, max=20.0)
    return 0.5 * torch.sum(
        mu.pow(2) + logvar.exp() - 1.0 - logvar,
        dim=list(range(1, mu.dim())),
    )


def kl_loss_1d(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q||N(0,I)) pour un posterior 1D (B, D). Retourne [B]."""
    logvar = torch.clamp(logvar, min=-30.0, max=20.0)
    return 0.5 * torch.sum(
        mu.pow(2) + logvar.exp() - 1.0 - logvar,
        dim=1,
    )


# =============================================================================
# ContentStyleEncoder
# =============================================================================

class BifurcatedContentStyleEncoder(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        fft_sigma: float = 7.5,
        in_channels: int = 5,
        hidden_channels: List[int] = [64, 128, 256, 512],
        kernel_sizes: List[int] = [3, 3, 3, 3],
        strides: List[int] = [1, 2, 2, 2],
        latent_channels: int = 8,
        style_channels: int = 256,
        num_residual_blocks: int = 1,
        spatial_dims: int = 2,
        normalization: Tuple = ('group', {'num_groups': 32, 'affine': True}),
        activation: Tuple = ('swish', {}),
        dropout: float = 0.0,
        use_residual_block: bool = True,
        learnable_interpolation: bool = True,
        attention_type: Union[str, List[str]] = 'none',
    ):
        super().__init__()

        self.depth = len(hidden_channels)
        self.num_residual_blocks = num_residual_blocks

        attention_type = (
            attention_type if isinstance(attention_type, list)
            else [attention_type] * self.depth
        )
        ConvBlock = UnetResBlock if use_residual_block else UnetBasicBlock

        # ── FFT Filter ────────────────────────────────────────────────────────
        self.fft_filter = LearnableFFTHighPassFilter(
            input_shape, in_channels=in_channels, sigma=fft_sigma, spatial_dims=spatial_dims
        )

        # ── In-Convolution (Tronc commun) ─────────────────────────────────────
        self.input_conv = BasicBlock(
            spatial_dims, in_channels * 2, hidden_channels[0],
            kernel_size=kernel_sizes[0], stride=strides[0],
        )

        # ── Fonction pour dédoubler l'architecture sans dupliquer le code ─────
        def _build_branch():
            encoder_block_list = []
            for i in range(1, self.depth):
                for k in range(num_residual_blocks):
                    seq = [
                        ConvBlock(
                            spatial_dims=spatial_dims,
                            in_channels=hidden_channels[i - 1] if k == 0 else hidden_channels[i],
                            out_channels=hidden_channels[i],
                            kernel_size=kernel_sizes[i],
                            stride=1,
                            norm_name=normalization,
                            act_name=activation,
                            dropout=dropout,
                            emb_channels=None,        # pas de conditioning
                        ),
                        Attention(
                            spatial_dims=spatial_dims,
                            in_channels=hidden_channels[i],
                            out_channels=hidden_channels[i],
                            num_heads=8,
                            ch_per_head=hidden_channels[i] // 8,
                            depth=1,
                            norm_name=normalization,
                            dropout=dropout,
                            emb_dim=None,
                            attention_type=attention_type[i],
                        ),
                    ]
                    encoder_block_list.append(SequentialEmb(*seq))

                # --- OLD CODE (Pour compatibilité avec les anciens checkpoints) ---
                # if i < self.depth - 1:
                #     encoder_block_list.append(
                #         BasicDown(
                #             spatial_dims=spatial_dims,
                #             in_channels=hidden_channels[i],
                #             out_channels=hidden_channels[i],
                #             kernel_size=kernel_sizes[i], # OLD
                #             stride=strides[i],
                #             learnable_interpolation=learnable_interpolation,
                #         )
                #     )
                
                # --- NEW CODE (Correction 3 downsamples) ---
                encoder_block_list.append(
                    BasicDown(
                        spatial_dims=spatial_dims,
                        in_channels=hidden_channels[i],
                        out_channels=hidden_channels[i],
                        kernel_size=kernel_sizes[i], # REMIS A 3 + PADDING ASYMETRIQUE
                        stride=strides[i],
                        learnable_interpolation=learnable_interpolation,
                    )
                )

            middle_block = SequentialEmb(
                ConvBlock(
                    spatial_dims=spatial_dims,
                    in_channels=hidden_channels[-1], out_channels=hidden_channels[-1],
                    kernel_size=kernel_sizes[-1], stride=1,
                    norm_name=normalization, act_name=activation,
                    dropout=dropout, emb_channels=None,
                ),
                Attention(
                    spatial_dims=spatial_dims,
                    in_channels=hidden_channels[-1], out_channels=hidden_channels[-1],
                    num_heads=8, ch_per_head=hidden_channels[-1] // 8, depth=1,
                    norm_name=normalization, dropout=dropout,
                    emb_dim=None, attention_type=attention_type[-1],
                ),
                ConvBlock(
                    spatial_dims=spatial_dims,
                    in_channels=hidden_channels[-1], out_channels=hidden_channels[-1],
                    kernel_size=kernel_sizes[-1], stride=1,
                    norm_name=normalization, act_name=activation,
                    dropout=dropout, emb_channels=None,
                ),
            )
            return nn.ModuleList(encoder_block_list), middle_block

        # ── Instanciation des deux branches indépendantes ─────────────────────
        self.content_encoder_blocks, self.content_middle_block = _build_branch()
        self.style_encoder_blocks,   self.style_middle_block   = _build_branch()

        # ── Content and Style heads : spatial posterior ─────────────────────────────────
        self.content_head = nn.Sequential(
            BasicBlock(spatial_dims, hidden_channels[-1], 2 * latent_channels, kernel_size=3),
            BasicBlock(spatial_dims, 2 * latent_channels, 2 * latent_channels, kernel_size=1)
        )
        self.style_head   = BasicBlock(spatial_dims, hidden_channels[-1], 2 * style_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        mu_content      : (B, latent_channels, H', W')
        logvar_content  : (B, latent_channels, H', W')
        mu_style        : (B, style_channels)
        logvar_style    : (B, style_channels)
        """
        # ── Tronc commun ─────────────────────────────────────────────────────
        fft_x = self.fft_filter(x)
        h_shared = self.input_conv(torch.cat([x, fft_x], dim=1))  
        
        # ── Branche Content ──────────────────────────────────────────────────
        h_c = h_shared
        for block in self.content_encoder_blocks:
            h_c = block(h_c, None)
        h_c = self.content_middle_block(h_c, None)

        # ── Branche Style ────────────────────────────────────────────────────
        h_s = h_shared
        for block in self.style_encoder_blocks:
            h_s = block(h_s, None)
        h_s = self.style_middle_block(h_s, None)

        # ── Content head ─────────────────────────────────────────────────────
        moments_c = self.content_head(h_c)
        mu_c, logvar_c = moments_c.chunk(2, dim=1)
        
        # ── Style head ───────────────────────────────────────────────────────
        moments_s = self.style_head(h_s)
        mu_s, logvar_s = moments_s.chunk(2, dim=1)

        return mu_c, logvar_c, mu_s, logvar_s



# =============================================================================
# StyleEmbedder
# =============================================================================

class StyleEmbedder(nn.Module):
    """
    Projette z_style 1D (B, style_channels) vers un vecteur d'embedding
    (B, style_embedding_dim) utilisé comme condition scale-shift dans le décodeur.

    Même rôle et même structure que le time_embedder du UNet :
        Linear(style_channels → style_embedding_dim) → SiLU → Linear(style_embedding_dim → style_embedding_dim)

    Parameters
    ----------
    style_embedding_dim : dimension de l'embedding de sortie = tembedding_channels du décodeur
    """

    def __init__(self, style_channels: int = 256, style_embedding_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(style_channels, style_embedding_dim, bias=False),
            nn.SiLU(),
            nn.Linear(style_embedding_dim, style_embedding_dim, bias=False),
        )

    def forward(self, z_style: torch.Tensor) -> torch.Tensor:
        """z_style : (B, style_channels) → style_emb : (B, style_embedding_dim)"""
        return self.net(z_style)

    
class AdaINResBlock(nn.Module):
    """
    ResBlock avec Adaptive Instance Normalization.
 
    Pour chaque normalisation :
      1. InstanceNorm2d normalise z_content → mu=0, sigma=1 par canal et par instance
      2. Deux projections linéaires depuis style_emb prédisent gamma et beta
      3. output = (1 + gamma) * normalized + beta
         (gamma centré sur 0 → comportement neutre au départ)
 
    Parameters
    ----------
    in_channels  : int
    out_channels : int
    style_dim    : int  — dimension de style_emb (= style_embedding_dim)
    dropout      : float
    spatial_dims : int  — 2 ou 3 (par défaut 2)
    """
 
    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        style_dim:    int,
        dropout:      float = 0.0,
        spatial_dims: int = 2
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        
        # 💡 Instanciation dynamique selon la dimension spatiale
        Conv = getattr(nn, f"Conv{spatial_dims}d")
        InstanceNorm = getattr(nn, f"InstanceNorm{spatial_dims}d")
        Dropout = getattr(nn, f"Dropout{spatial_dims}d") if dropout > 0 else nn.Identity

        self.conv1 = Conv(in_channels,  out_channels, kernel_size=3, padding=1, padding_mode='replicate')
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate')
        
        self.norm1 = InstanceNorm(out_channels, affine=False)
        self.norm2 = InstanceNorm(out_channels, affine=False)
        
        self.act   = nn.SiLU()
        self.drop  = Dropout(dropout) if dropout > 0 else nn.Identity()

        # Projections style → (gamma, beta)
        self.adain1 = nn.Linear(style_dim, out_channels * 2)
        self.adain2 = nn.Linear(style_dim, out_channels * 2)

        nn.init.normal_(self.adain1.weight, std=0.02)
        nn.init.zeros_(self.adain1.bias)
        nn.init.normal_(self.adain2.weight, std=0.02)
        nn.init.zeros_(self.adain2.bias)

        self.skip = (
            Conv(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def _adain(
        self,
        x:         torch.Tensor,   # Déjà normalisé par InstanceNorm
        style_emb: torch.Tensor,   
        proj:      nn.Linear,
    ) -> torch.Tensor:
        params        = proj(style_emb)
        gamma, beta   = params.chunk(2, dim=1)
        
        # Broadcast adaptatif (ajoute 2 dimensions en 2D, 3 en 3D)
        for _ in range(self.spatial_dims):
            gamma = gamma.unsqueeze(-1)
            beta  = beta.unsqueeze(-1)
            
        return (1.0 + gamma) * x + beta

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self._adain(self.norm1(h), emb, self.adain1)
        h = self.act(h)
        h = self.drop(h)

        h = self.conv2(h)
        h = self._adain(self.norm2(h), emb, self.adain2)
        h = self.act(h)

        return h + self.skip(x)


class StyleConditionedDecoder(nn.Module):
    """
    Parameters
    ----------
    latent_channels : int       canaux de z_content en entrée
    out_channels    : int       canaux de l'image reconstruite
    hidden_channels : List[int] doit être identique à ContentStyleEncoder.hidden_channels
    kernel_sizes    : List[int]
    strides         : List[int]
    style_embedding_dim : int = tembedding_channels du décodeur (sortie de StyleEmbedder)
    num_residual_blocks : int doit être identique à ContentStyleEncoder.num_residual_blocks
    spatial_dims, normalization, activation, dropout, use_residual_block,
    learnable_interpolation, attention_type
    """
    
    def __init__(
        self,
        latent_channels:    int = 8,
        out_channels:       int = 5,
        hidden_channels:    List[int] = [64, 128, 256, 512],
        kernel_sizes:       List[int] = [3, 3, 3, 3],
        strides:            List[int] = [1, 2, 2, 2],
        style_embedding_dim: int = 256,
        num_residual_blocks: int = 1,
        spatial_dims:       int = 2,
        normalization:      Tuple = ('group', {'num_groups': 32, 'affine': True}),
        activation:         Tuple = ('swish', {}),
        dropout:            float = 0.0,
        use_residual_block: bool = True,        # conservé pour compatibilité YAML
        learnable_interpolation: bool = True,
        attention_type:     Union[str, List[str]] = 'none',
    ):
        super().__init__()

        self.depth = len(hidden_channels)
        self.num_residual_blocks = num_residual_blocks

        attention_type = (
            attention_type if isinstance(attention_type, list)
            else [attention_type] * self.depth
        )
        
        # ── latent_to_features : z_content → hidden_channels[-1] via AdaIN ──
        self.latent_to_features = AdaINResBlock(
            in_channels=latent_channels,
            out_channels=hidden_channels[-1],
            style_dim=style_embedding_dim,
            dropout=dropout,
            spatial_dims=spatial_dims
        )
        
        # ConvBlock = UnetResBlock if use_residual_block else UnetBasicBlock
        # ConvBlock = AdaINResBlock

        # ── Blocs décodeur — trois listes parallèles ─────────────────────────
        # adain_blocks[j]  : AdaINResBlock
        # attn_blocks[j]   : Attention (ou None si attention_type == 'none')
        # up_blocks[j]     : BasicUp  (ou None si pas d'upsample à ce bloc)
        #
        # Index j parcourt les mêmes cases que l'ancienne decoder_block_list.
        adain_blocks = []
        attn_blocks  = []
        up_blocks    = []
 
        for i in range(1, self.depth):
            for k in range(num_residual_blocks + 1):
                out_ch_k = hidden_channels[i - 1 if k == 0 else i]
                in_ch_k  = hidden_channels[i]
 
                # AdaIN remplace le ConvBlock
                adain_blocks.append(AdaINResBlock(
                    in_channels=in_ch_k,
                    out_channels=out_ch_k,
                    style_dim=style_embedding_dim,
                    dropout=dropout,
                    spatial_dims=spatial_dims
                ))
 
                # Attention (inchangée)
                attn_blocks.append(Attention(
                    spatial_dims=spatial_dims,
                    in_channels=out_ch_k,
                    out_channels=out_ch_k,
                    num_heads=8,
                    ch_per_head=max(1, out_ch_k // 8),
                    depth=1,
                    norm_name=normalization,
                    dropout=dropout,
                    emb_dim=None,                   # Attention sans conditioning
                    attention_type=attention_type[i],
                ))
 
                # --- OLD CODE (Pour compatibilité avec les anciens checkpoints) ---
                # if (i > 1) and (k == 0):
                #     up_blocks.append(BasicUp(
                #         spatial_dims=spatial_dims,
                #         in_channels=out_ch_k,
                #         out_channels=out_ch_k,
                #         kernel_size=strides[i], # Dans ton code original, tu avais strides[i]
                #         stride=strides[i],
                #         learnable_interpolation=learnable_interpolation,
                #     ))
                # else:
                #     up_blocks.append(None)

                # --- NEW CODE (Correction upsample à toutes les couches i >= 1) ---
                if k == 0:
                    up_blocks.append(BasicUp(
                        spatial_dims=spatial_dims,
                        in_channels=out_ch_k,
                        out_channels=out_ch_k,
                        kernel_size=strides[i],
                        stride=strides[i],
                        learnable_interpolation=learnable_interpolation,
                    ))
                else:
                    up_blocks.append(None)
 
        self.adain_blocks = nn.ModuleList(adain_blocks)
        self.attn_blocks  = nn.ModuleList(attn_blocks)
        # BasicUp peut être None — on stocke séparément les vrais modules
        self._up_blocks_raw = up_blocks
        self.up_blocks = nn.ModuleList([b for b in up_blocks if b is not None])
 
        # ── Out-Convolution ───────────────────────────────────────────────────
        self.output_conv = zero_module(
            UnetOutBlock(spatial_dims, hidden_channels[0], out_channels, dropout=None)
        )

    def forward(
        self,
        z_content: torch.Tensor,
        style_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_content     : (B, latent_channels, H', W')
        style_emb     : (B, style_embedding_dim) — produit par StyleEmbedder

        Returns
        -------
        x_hat : (B, out_channels, H, W)
        """
        # latent_to_features : z_content → h (B, hidden_channels[-1], H', W')
        h = self.latent_to_features(z_content, emb=style_emb)
        
        up_idx = 0  # pointeur dans self.up_blocks

        for j in range(len(self.adain_blocks) - 1, -1, -1):
            # AdaIN block
            h = self.adain_blocks[j](h, style_emb)
 
            # Attention (passée sans emb — purement spatiale)
            h = self.attn_blocks[j](h, None)
 
            # Upsample si présent à cet index
            if self._up_blocks_raw[j] is not None:
                real_up_idx = sum(
                    1 for b in self._up_blocks_raw[:j] if b is not None
                )
                h = self.up_blocks[real_up_idx](h)
 
        return self.output_conv(h)

# =============================================================================
# DisentangledHarmonizationVAE — assemblage complet
# =============================================================================

class DisentangledHarmonizationVAE(nn.Module):
    """
    Parameters
    ----------
    in_channels, out_channels : int
    hidden_channels, kernel_sizes, strides : List[int] (identiques pour tous les sous-modules)
    latent_channels : int dim de z_content
    style_channels : int dim de z_style (1D)
    style_embedding_dim : int dim de l'embedding de style (= tembedding_channels du décodeur)
    num_residual_blocks : int
    spatial_dims : int
    normalization, activation, dropout, use_residual_block, learnable_interpolation, attention_type
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        fft_sigma: float = 7.5,
        in_channels: int = 5,
        out_channels: int = 5,
        hidden_channels: List[int] = [64, 128, 256, 512],
        kernel_sizes: List[int] = [3, 3, 3, 3],
        strides: List[int] = [1, 2, 2, 2],
        latent_channels: int = 8,
        style_channels: int = 256,
        style_embedding_dim: int = 512,
        num_residual_blocks: int = 1,
        spatial_dims: int = 2,
        normalization: Tuple = ('group', {'num_groups': 32, 'affine': True}),
        activation: Tuple = ('swish', {}),
        dropout: float = 0.0,
        use_residual_block: bool = True,
        learnable_interpolation: bool = True,
        attention_type: Union[str, List[str]] = 'none'
    ):
        super().__init__()

        self.shared_kwargs = dict(
            hidden_channels=hidden_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            num_residual_blocks=num_residual_blocks,
            spatial_dims=spatial_dims,
            normalization=normalization,
            activation=activation,
            dropout=dropout,
            use_residual_block=use_residual_block,
            learnable_interpolation=learnable_interpolation,
            attention_type=attention_type,
        )

        # self.content_style_encoder = ContentStyleEncoder(
        self.content_style_encoder = BifurcatedContentStyleEncoder(
            input_shape=input_shape,
            fft_sigma=fft_sigma,
            in_channels=in_channels,
            latent_channels=latent_channels,
            style_channels=style_channels,
            **self.shared_kwargs,
        )
        
        InstanceNorm = getattr(nn, f"InstanceNorm{spatial_dims}d")
        self.content_norm = InstanceNorm(latent_channels, affine=False)
        
        AdaptiveAvgPool = getattr(nn, f"AdaptiveAvgPool{spatial_dims}d")
        self.pool    = AdaptiveAvgPool(1)

        self.style_embedder = StyleEmbedder(
            style_channels=style_channels,
            style_embedding_dim=style_embedding_dim,
        )

        self.decoder = StyleConditionedDecoder(
            latent_channels=latent_channels,
            out_channels=out_channels,
            style_embedding_dim=style_embedding_dim,
            **self.shared_kwargs,
        )
        
        AdaptiveAvgPool = getattr(nn, f"AdaptiveAvgPool{spatial_dims}d")
        self.pool    = AdaptiveAvgPool(1)
        self.flatten = nn.Flatten()

    def to_style(self, x: torch.Tensor) -> torch.Tensor:
        """Réduit un tenseur spatial en vecteur 1D via Pool + Flatten."""
        return self.flatten(self.pool(x))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retourne (mu_c, logvar_c, mu_s, logvar_s) issus de l'encodeur bifurqué."""
        return self.content_style_encoder(x)

    def decode(self, z_content: torch.Tensor, z_style: torch.Tensor) -> torch.Tensor:
        """Décode z_content conditionné par z_style."""
        style_emb     = self.style_embedder(z_style)
        return self.decoder(z_content, style_emb)

    def forward(self, x: torch.Tensor, sample_posterior: bool = True, style_dropout_p: float = 0.0
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        mu_c, logvar_c, mu_s, logvar_s = self.encode(x)

        # Réduction de mu_s et logvar_s en vecteurs 1D pour le style
        mu_s, logvar_s = self.to_style(mu_s), self.to_style(logvar_s)
        
        # 1. Traitement du Content
        z_content = reparameterize(mu_c, logvar_c) if sample_posterior else mu_c
        z_style = reparameterize(mu_s, logvar_s) if sample_posterior else mu_s
        
        # 2. Normalisation du Content pour supprimer les stats de style
        z_content_norm = self.content_norm(z_content)
        
        if self.training and style_dropout_p > 0.0:
            mask    = (torch.rand(z_style.shape[0], 1, device=z_style.device) > style_dropout_p).float()
            z_style = z_style * mask

        # 3. Décodage
        x_hat = self.decode(z_content_norm, z_style)

        return x_hat, (mu_c, logvar_c), (mu_s, logvar_s), z_content, z_style

    @torch.no_grad()
    def harmonize(
        self,
        x_source: torch.Tensor,
        x_style_ref: Optional[torch.Tensor] = None,
        z_style_fixed: Optional[torch.Tensor] = None,
        style_dropout_p: Optional[float] = None,
        alpha_style: float = 1.0
    ) -> torch.Tensor:
        """
        alpha_style permet d'introduire le style petit à petit à l'inférence.
        """
        mu_c, logvar_c, mu_s_spatial, _ = self.encode(x_source)
        
        z_content = reparameterize(mu_c, logvar_c)
        z_content_norm = self.content_norm(z_content)
        
        mu_s = self.to_style(mu_s_spatial)

        if z_style_fixed is not None:
            z_style = z_style_fixed.expand(x_source.shape[0], -1)
        elif x_style_ref is not None:
            _, _, mu_s_ref_spatial, _ = self.encode(x_style_ref)
            mu_s_ref = self.to_style(mu_s_ref_spatial)
            z_style = mu_s_ref
        elif style_dropout_p is not None and style_dropout_p > 0.0:
            mask    = (torch.rand(mu_s.shape[0], 1, device=mu_s.device) > style_dropout_p).float()
            z_style = mu_s * mask
        else:
            z_style = torch.zeros(
                x_source.shape[0], self.style_embedder.net[0].in_features,
                device=x_source.device, dtype=x_source.dtype,
            )

        # Mixage de style
        z_style = alpha_style * z_style

        return self.decode(z_content_norm, z_style)

    
"""
UnlearningVAE — Lightning module
=================================
Pipeline d'entraînement adversarial pour le DisentangledHarmonizationVAE.

Deux classifieurs de domaine internes :
  - style_classifier   : classifie z_style (1D) → doit discriminer les sites
                         (on VEUT que z_style encode le biais de scanner)
  - content_classifier : classifie z_content (spatial, poolé) → doit être confus
                         (on VEUT que z_content soit invariant au site)

Mécanisme de désapprentissage (Dinsdale) :
  Stage 1 — Warmup (epochs < warmup_epochs) :
      Forward VAE complet → L_rec + L_kl
      Classify z_style   → cross_entropy  (entraîne style_classifier)
      Classify z_content → cross_entropy  (entraîne content_classifier)
      Tout ensemble, un seul backward.

  Stage 2 — Unlearning :
      Étape A  : L_rec + L_kl  → opt_vae  (tâche de reconstruction)
      Étape B  : cross_entropy sur z_style/z_content détachés → opt_classifiers
      Étape C  : confusion loss KL→uniforme sur z_content → opt_unlearn
                 (seul le content_style_encoder est mis à jour)

3 optimiseurs (optimisation manuelle) :
  opt_vae         : VAE complet
  opt_classifiers : style_classifier + content_classifier
  opt_unlearn     : content_style_encoder uniquement
"""

# =============================================================================
# Classifieurs de domaine internes
# =============================================================================


class SpatialDomainClassifier(nn.Module):
    def __init__(self, channels: int, num_domains: int, hidden_dim: int = 128, spatial_dims: int = 3):
        super().__init__()
        self.pool = getattr(nn, f"AdaptiveAvgPool{spatial_dims}d")(1)
        self.flatten = nn.Flatten()
        
        self.net = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_domains)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.flatten(self.pool(x))
        return self.net(features)

class DomainClassifier(nn.Module):
    def __init__(self, channels: int, num_domains: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, z_style: torch.Tensor) -> torch.Tensor:
        return self.net(z_style)


# =============================================================================
# UnlearningVAE — Lightning module
# =============================================================================

class UnlearningVAE(LightningModule):
    """
    Parameters
    ----------
    vae : DisentangledHarmonizationVAE
        Architecture VAE disentanglée (définie dans harmonization_vae.py).
    num_domains : int
        Nombre de sites/domaines (ex: 3 pour CHB, Rennes, A100).
    classifier_hidden_dim : int
        Dimension cachée des classifieurs de domaine.
    warmup_epochs : int
        Nombre d'époques de warmup (stage 1) avant le désapprentissage.
    beta_confusion : float
        Poids de la confusion loss (désapprentissage sur z_content).
    kl_content_weight : float
        Poids de la KL loss sur z_content (régularisation VAE spatiale).
    kl_style_weight : float
        Poids de la KL loss sur z_style (régularisation VAE 1D).
    lr_vae : float
        Learning rate pour le VAE complet (tâche de reconstruction).
    lr_classifiers : float
        Learning rate pour les classifieurs de domaine.
    lr_unlearn : float
        Learning rate pour la confusion loss (plus faible = désapprentissage doux).
    weight_decay : float
    suv_global_log_max : float
        Constante de normalisation clinique (log1p SUV → [-1, 1]).
    """

    def __init__(
        self,
        vae: DisentangledHarmonizationVAE,
        num_domains: int = 3,
        classifier_hidden_dim: int = 256,
        warmup_epochs: int = 10,
        beta_confusion: float = 1.0,
        max_epochs: int = 300,
        kl_content_weight: float = 1e-4,
        kl_style_weight: float = 1e-4,
        ssim_weight: float = 0.5,
        lr_vae: float = 1e-4,
        lr_classifiers: float = 1e-4,
        lr_unlearn: float = 1e-5,
        lr_classifiers_final: float = 1e-5,
        lr_unlearn_final: float = 1e-7,
        k_style_steps: int = 2,
        weight_decay: float = 1e-5,
        suv_global_log_max: float = 6.0,
    ):
        super().__init__()

        self.vae                = vae
        self.num_domains        = num_domains
        self.warmup_epochs      = warmup_epochs
        self.beta_confusion     = beta_confusion
        self.max_epochs         = max_epochs
        self.kl_content_weight  = kl_content_weight
        self.kl_style_weight    = kl_style_weight
        self.ssim_weight        = ssim_weight
        self.lrs                = {"vae": lr_vae, "classifiers": lr_classifiers, "unlearn": lr_unlearn}
        self.lr_classifiers_final = lr_classifiers_final
        self.lr_unlearn_final   = lr_unlearn_final
        self.k_style_steps      = k_style_steps
        self.weight_decay       = weight_decay
        self.suv_global_log_max = suv_global_log_max
        self.spatial_dims        = vae.shared_kwargs.get("spatial_dims", 2)

        # ── Classifieurs de domaine ───────────────────────────────────────────
        style_channels  = vae.content_style_encoder.style_head.conv.out_channels // 2

        self.style_classifier = DomainClassifier(
            channels=style_channels,
            num_domains=num_domains,
            hidden_dim=classifier_hidden_dim
        )

        latent_channels = vae.content_norm.num_features
        self.content_classifier = SpatialDomainClassifier(
            channels=latent_channels,
            num_domains=num_domains,
            hidden_dim=min(256, classifier_hidden_dim),
            spatial_dims=self.spatial_dims
        )

        # ── Métriques ────────────────────────────────────────────────────────
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        # Optimisation manuelle obligatoire (multi-optimiseurs adversariaux)
        self.automatic_optimization = False
    
        self.save_hyperparameters(ignore=["vae"])

    # ──────────────────────────────────────────────────────────────────────────
    # Paramètres par groupe
    # ──────────────────────────────────────────────────────────────────────────

    def _vae_parameters(self):
        """Tous les paramètres du VAE (reconstruction + KL)."""
        return list(self.vae.parameters())

    def _classifier_parameters(self):
        """Paramètres des deux classifieurs de domaine."""
        return (
            list(self.style_classifier.parameters())
            + list(self.content_classifier.parameters())
        )

    def _encoder_parameters(self):
        return list(self.vae.content_style_encoder.parameters())

    # ──────────────────────────────────────────────────────────────────────────
    # Utilitaires
    # ──────────────────────────────────────────────────────────────────────────

    def _normalize(self, suv: torch.Tensor) -> torch.Tensor:
        """SUV → espace log normalisé [-1, 1]."""
        log = torch.log1p(suv)
        return 2.0 * (log.clamp(0, self.suv_global_log_max) / self.suv_global_log_max) - 1.0

    def _denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Espace log normalisé [-1, 1] → SUV."""
        log = 0.5 * (x_norm.clamp(-1, 1) + 1.0) * self.suv_global_log_max
        return torch.expm1(log)

    def _kl_confusion_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """KL(pred || uniforme) — pousse le classifieur vers l'incertitude maximale."""
        uniform = torch.full_like(logits, 1.0 / self.num_domains)
        return F.kl_div(F.log_softmax(logits, dim=1), uniform, reduction="batchmean")
    
    def _confusion_loss(self, logits):
        p    = F.softmax(logits, dim=1)
        logp = torch.log(p + 1e-8)
        return -logp.mean()

    def _confusion_loss_spatial(self, logits):
        p    = F.softmax(logits, dim=1)
        logp = torch.log(p + 1e-8)
        return -logp.mean()

    def _log_dict(self, d: Dict[str, torch.Tensor], batch_size: int):
        for key, value in d.items():
            self.log(key, value, prog_bar=True, sync_dist=True,
                     on_step=True, on_epoch=True, batch_size=batch_size)
            

    # ──────────────────────────────────────────────────────────────────────────
    # scheduler
    # ──────────────────────────────────────────────────────────────────────────
    def _stage2_progress(self) -> float:
        """t ∈ [0, 1] : progression dans le stage 2."""
        t = max(0, self.current_epoch - self.warmup_epochs)
        T = max(1, self.max_epochs - self.warmup_epochs)
        return min(t / T, 1.0)
    
    def _cosine_lr(self, lr_init: float, lr_final: float) -> float:
        p = self._stage2_progress()
        return lr_final + (lr_init - lr_final) * (1 + math.cos(math.pi * p)) / 2

    def _scheduled_lr_classifiers(self) -> float:
        """lr_classifiers : descend vers lr * lr_classifiers_min_factor (cosine)."""
        p       = self._stage2_progress()
        lr_init = self.lrs["classifiers"]
        lr_min  = lr_init * self.lr_classifiers_min_factor
        return lr_min + (lr_init - lr_min) * (1 + math.cos(math.pi * p)) / 2
    
    def _is_warmup(self) -> bool:
        if isinstance(self.warmup_epochs, float):
            iters_per_epoch = self.trainer.num_training_batches
            return self.global_step < int(self.warmup_epochs * iters_per_epoch)
        return self.current_epoch < self.warmup_epochs

    def on_train_epoch_start(self):
        """Met à jour beta et lr_classifiers au début de chaque époque du stage 2."""
        if self._is_warmup():
            return

        _, opt_style_clf, opt_content_clf, opt_unlearn = self.optimizers()
        
        new_lr_clf = self._cosine_lr(self.lrs["classifiers"], self.hparams.lr_classifiers_final)
        for opt in [opt_style_clf, opt_content_clf]:
            for pg in opt.param_groups:
                pg["lr"] = new_lr_clf

        new_lr_unlearn = self._cosine_lr(self.lrs["unlearn"], self.hparams.lr_unlearn_final)
        for pg in opt_unlearn.param_groups:
            pg["lr"] = new_lr_unlearn

        self.log("debug/lr_classifiers", new_lr_clf,    on_step=False, on_epoch=True)
        self.log("debug/lr_unlearn",     new_lr_unlearn, on_step=False, on_epoch=True)      
        

    # ──────────────────────────────────────────────────────────────────────────
    # configure_optimizers
    # ──────────────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        # Reconstruction + KL : VAE complet
        opt_vae = torch.optim.AdamW(
            self._vae_parameters(),
            lr=self.lrs["vae"],
            weight_decay=self.weight_decay,
        )
        # Discrimination de site : classifieurs seuls
        opt_style_clf = torch.optim.AdamW(
            self.style_classifier.parameters(),
            lr=self.lrs["classifiers"] * 2, # TODO: rectifier le facteur 2 (test d'une lr plus élevée pour les classifieurs)
            weight_decay=self.weight_decay,
        )
        opt_content_clf = torch.optim.AdamW(
            self.content_classifier.parameters(),
            lr=self.lrs["classifiers"],
            weight_decay=self.weight_decay,
        )
        # Désapprentissage : encodeur seul, LR plus faible
        opt_unlearn = torch.optim.AdamW(
            self._encoder_parameters(),
            lr=self.lrs["unlearn"],
            weight_decay=self.weight_decay,
        )
        return [opt_vae, opt_style_clf, opt_content_clf, opt_unlearn]

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hat, *_ = self.vae(x, sample_posterior=False)
        return x_hat

    # ──────────────────────────────────────────────────────────────────────────
    # training_step
    # ──────────────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        opt_vae, opt_style_clf, opt_content_clf, opt_unlearn = self.optimizers()

        # ── Données ──────────────────────────────────────────────────────────
        suv_source    = batch["source"][tio.DATA].float()
        domain_labels = batch["domain_id"]          # LongTensor [B] ∈ {0, …, num_domains-1}

        if self.spatial_dims == 2:
            suv_source = suv_source.squeeze(1)

        x  = self._normalize(suv_source)
        bs = x.shape[0]
        is_stage_1 = self.current_epoch < self.warmup_epochs

        # ══════════════════════════════════════════════════════════════════════
        # STAGE 1 — Warmup
        # ══════════════════════════════════════════════════════════════════════
        if is_stage_1:
            x_hat, kl_vars_c, kl_vars_s, z_content, z_style = self.vae.forward(x, sample_posterior=True)
            mu_c, logvar_c = kl_vars_c
            mu_s, logvar_s = kl_vars_s

            # Losses de reconstruction et KL
            # SSIM attend des valeurs dans [0, 1]
            x_01     = (x.clamp(-1, 1) + 1.0) / 2.0
            x_hat_01 = (x_hat.clamp(-1, 1) + 1.0) / 2.0
            
            loss_l1         = F.l1_loss(x_hat, x)
            loss_ssim       = 1.0 - self.ssim(x_hat_01, x_01)
            loss_rec        = loss_l1 + self.ssim_weight * loss_ssim
            
            loss_kl_content = kl_loss_spatial(mu_c, logvar_c).mean()
            loss_kl_style = kl_loss_1d(mu_s, logvar_s).mean()

            loss_vae = (
                loss_rec +
                self.kl_content_weight * loss_kl_content +
                self.kl_style_weight * loss_kl_style
            )

            # Classifieurs — graphe complet (pas de detach), les deux apprennent
            # style et s_content sont déterministes (taille style_channels)
            logits_style   = self.style_classifier(z_style)
            logits_content = self.content_classifier(z_content)

            loss_dm_style   = F.cross_entropy(logits_style, domain_labels)
            loss_dm_content = F.cross_entropy(logits_content, domain_labels)
            loss_classifiers = loss_dm_style + loss_dm_content

            total_loss = loss_classifiers + loss_vae

            opt_vae.zero_grad()
            opt_style_clf.zero_grad()
            opt_content_clf.zero_grad()
            self.manual_backward(total_loss)
            opt_vae.step()
            opt_style_clf.step()
            opt_content_clf.step()
            
            self._log_dict({
                "train/rec_loss":         loss_rec,
                "train/kl_content":       loss_kl_content,
                "train/dm_style":         loss_dm_style,
                "train/dm_content":       loss_dm_content,
                "train/ssim":             loss_ssim,
                "train/total":            total_loss,
            }, bs)

        # ══════════════════════════════════════════════════════════════════════
        # STAGE 2 — Unlearning (3 étapes dissociées)
        # ══════════════════════════════════════════════════════════════════════
        else:
            x_hat, kl_vars_c, kl_vars_s, z_content, z_style = self.vae.forward(x, sample_posterior=True)
            mu_c, logvar_c = kl_vars_c
            mu_s, logvar_s = kl_vars_s

            # SSIM attend des valeurs dans [0, 1]
            x_01     = (x.clamp(-1, 1) + 1.0) / 2.0
            x_hat_01 = (x_hat.clamp(-1, 1) + 1.0) / 2.0
            
            loss_l1         = F.l1_loss(x_hat, x)
            loss_ssim       = 1.0 - self.ssim(x_hat_01, x_01)
            loss_rec        = loss_l1 + self.ssim_weight * loss_ssim
            
            loss_kl_content = kl_loss_spatial(mu_c, logvar_c).mean()
            loss_kl_style = kl_loss_1d(mu_s, logvar_s).mean()

            loss_vae = (
                loss_rec +
                self.kl_content_weight * loss_kl_content +
                self.kl_style_weight * loss_kl_style
            )

            opt_vae.zero_grad()
            self.manual_backward(loss_vae)
            opt_vae.step()

            # ── Étape B : mise à jour classifieurs (encodeur GELÉ) ────────────
            # Les z sont recalculés sans gradient pour ne pas polluer l'encodeur.
            self.vae.content_style_encoder.eval()
            for p in self.vae.content_style_encoder.parameters():
                p.requires_grad = False
                
            z_style_det = z_style.detach()
            z_content_det = z_content.detach()

            for _ in range(self.k_style_steps):
                logits_style_det   = self.style_classifier(z_style_det)
                loss_dm_style      = F.cross_entropy(logits_style_det, domain_labels)
                opt_style_clf.zero_grad()
                self.manual_backward(loss_dm_style)
                torch.nn.utils.clip_grad_norm_(self.style_classifier.parameters(), max_norm=1.0)
                opt_style_clf.step()
            
            
            logits_content_det = self.content_classifier(z_content_det)
            loss_dm_content = F.cross_entropy(logits_content_det, domain_labels)
            opt_content_clf.zero_grad()
            self.manual_backward(loss_dm_content)
            torch.nn.utils.clip_grad_norm_(self.content_classifier.parameters(), max_norm=1.0)
            opt_content_clf.step()

            # ── Étape C : confusion loss (encodeur seul) ─────────────────────
            self.vae.content_style_encoder.train()
            for p in self.vae.content_style_encoder.parameters():
                p.requires_grad = True

            mu_c_conf, logvar_c_conf, _, _ = self.vae.encode(x)
            z_content_conf = reparameterize(mu_c_conf, logvar_c_conf)
            
            logits_content_conf = self.content_classifier(z_content_conf)
            loss_confusion = self.beta_confusion * self._confusion_loss_spatial(logits_content_conf)

            opt_unlearn.zero_grad()
            self.manual_backward(loss_confusion)
            torch.nn.utils.clip_grad_norm_(self._encoder_parameters(), max_norm=1.0)
            opt_unlearn.step()

            # ── Logs stage 2 ─────────────────────────────────────────────────
            self._log_dict({
                "train/rec_loss":     loss_rec,
                "train/kl_content":   loss_kl_content,
                "train/dm_style":     loss_dm_style,
                "train/ssim":         loss_ssim,
                "train/dm_content":   loss_dm_content,
                "train/confusion":    loss_confusion,
            }, bs)

    # ──────────────────────────────────────────────────────────────────────────
    # validation_step
    # ──────────────────────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        suv_source    = batch["source"][tio.DATA].float()
        domain_labels = batch["domain_id"]

        if self.spatial_dims == 2:
            suv_source = suv_source.squeeze(1)

        x  = self._normalize(suv_source)
        bs = x.shape[0]

        # ── Forward VAE ───────────────────────────────────────────────────────
        x_hat, kl_vars_c, kl_vars_s, z_content, z_style = self.vae.forward(x, sample_posterior=False, style_dropout_p=0.0)
        mu_c, logvar_c = kl_vars_c
        mu_s, logvar_s = kl_vars_s

        # ── SSIM & Reconstruction ─────────────────────────────────────────────
        x_01     = (x.clamp(-1, 1) + 1.0) / 2.0
        x_hat_01 = (x_hat.clamp(-1, 1) + 1.0) / 2.0
        
        ssim_score      = self.ssim(x_hat_01, x_01)
        loss_l1         = F.l1_loss(x_hat, x)
        loss_ssim       = 1.0 - ssim_score
        loss_rec        = loss_l1 + self.ssim_weight * loss_ssim
        
        loss_kl_content = kl_loss_spatial(mu_c, logvar_c).mean()
        loss_kl_style   = kl_loss_1d(mu_s, logvar_s).mean()

        # ── Classifieurs (sur les modes, pas d'échantillonnage) ───────────────
        logits_style   = self.style_classifier(z_style)
        logits_content = self.content_classifier(z_content)

        loss_dm_style   = F.cross_entropy(logits_style,   domain_labels)
        loss_dm_content = F.cross_entropy(logits_content, domain_labels)

        # Confusion loss de monitoring
        loss_confusion = self._confusion_loss_spatial(logits_content)

        # ── Accuracy (indicateurs clés) ───────────────────────────────────────
        # style_acc  : doit rester élevée (z_style discrimine le site)
        # content_acc: doit tendre vers 1/num_domains (z_content devient invariant)
        style_acc   = (logits_style.argmax(dim=1)   == domain_labels).float().mean()
        content_acc = (logits_content.argmax(dim=1) == domain_labels).float().mean()

        # ── Score composite ───────────────────────────────────────────────────
        
        # Score composite : bonne reconstruction + classifieur confus sur z_content
        # domain_acc_excess = combien content_acc dépasse le niveau du hasard
        chance_level      = 1.0 / self.num_domains
        content_acc_excess = (content_acc - chance_level).clamp(min=0.0)
        style_acc_deficit  = (1.0 - style_acc).clamp(min=0.0)   # on pénalise si style_acc chute
        composite_score    = loss_rec + 0.1 * content_acc_excess + 0.1 * style_acc_deficit

        self._log_dict({
            "val/rec_loss":        loss_rec,
            "val/kl_content":      loss_kl_content,
            "val/kl_style":        loss_kl_style,
            "val/dm_style":        loss_dm_style,
            "val/dm_content":      loss_dm_content,
            "val/confusion":       loss_confusion,
            "val/ssim":            ssim_score,
            "val/style_acc":       style_acc,    # doit rester haute
            "val/content_acc":     content_acc,  # doit tendre vers 1/num_domains
            "val/composite_score": composite_score,
        }, bs)

        if batch_idx == 0:
            self._log_images(x, x_hat, suv_source, z_content)

        return loss_rec

    # ──────────────────────────────────────────────────────────────────────────
    # Logging images
    # ──────────────────────────────────────────────────────────────────────────

    def on_validation_epoch_end(self):
        """Sauvegarde un checkpoint unique à la fin exacte du Stage 1 avec les métriques."""
        if self.current_epoch == self.warmup_epochs - 1:
            # Récupération des métriques agrégées à la fin de l'époque
            metrics = self.trainer.callback_metrics
            
            # Extraction sécurisée (float() gère à la fois les nombres normaux et les tenseurs PyTorch 0D)
            rec = float(metrics.get("val/rec_loss", 0.0))
            style = float(metrics.get("val/style_acc", 0.0))
            content = float(metrics.get("val/content_acc", 0.0))
            
            # Formatage identique au ModelCheckpoint classique
            filename = f"stage1_final_epoch={self.current_epoch:03d}-rec={rec:.4f}-style={style:.3f}-content={content:.3f}.ckpt"
            
            save_dir = self.trainer.default_root_dir
            ckpt_path = os.path.join(save_dir, "checkpoints", filename)
            
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            
            # Sauvegarde
            self.trainer.save_checkpoint(ckpt_path)
            self.print(f"\n💾 [UnlearningVAE] Checkpoint pré-confusion sauvegardé : {filename}")
            

    def _log_images(
        self,
        x_norm: torch.Tensor,
        x_hat_norm: torch.Tensor,
        suv_source: torch.Tensor,
        z_content: torch.Tensor,
    ):
        """Log WandB : Source vs Recon vs Harmonized (z_style=0)."""
        if self.trainer.global_rank != 0:
            return

        suv_pred = self._denormalize(x_hat_norm)

        # Inférence avec style neutre
        z_style_zero = torch.zeros(x_norm.shape[0], self.vae.style_embedder.net[0].in_features, device=x_norm.device)
        x_hat_neutral_norm = self.vae.decode(self.vae.content_norm(z_content), z_style_zero)
        suv_neutral = self._denormalize(x_hat_neutral_norm)

        if suv_source.ndim == 5:
            mid = suv_source.shape[2] // 2
            src_slice  = suv_source[:, :, mid, :, :]
            pred_slice = suv_pred[:, :, mid, :, :]
            neutral_slice = suv_neutral[:, :, mid, :, :]
        else:
            mid = suv_source.shape[1] // 2
            src_slice  = suv_source[:, mid:mid+1, :, :]
            pred_slice = suv_pred[:, mid:mid+1, :, :]
            neutral_slice = suv_neutral[:, mid:mid+1, :, :]

        display_max = max(5.0, src_slice.max().item(), pred_slice.max().item(), neutral_slice.max().item())
        
        imgs = torch.cat([src_slice, pred_slice, neutral_slice], dim=3)
        imgs = (imgs / display_max).clamp(0, 1)

        grid = make_grid(imgs, nrow=1, padding=2)
        wandb.log({
            "Validation/Reconstruction": wandb.Image(
                grid.permute(1, 2, 0).cpu().numpy(),
                caption=f"Source | Recon | Harmonisée (style nul) (epoch {self.current_epoch})"
            )
        })
        
        

# =============================================================================
# StandardHarmonizationVAE — Ablation (Sans Unlearning)
# =============================================================================

class StandardHarmonizationVAE(LightningModule):
    """
    Module Lightning standard pour l'ablation.
    Entraîne le DisentangledHarmonizationVAE uniquement sur la tâche de 
    reconstruction (MAE) et de régularisation de l'espace latent (KL), 
    sans aucun mécanisme de désapprentissage ni classifieur de domaine.
    """

    def __init__(
        self,
        vae: DisentangledHarmonizationVAE,
        kl_weight: float = 1e-4,
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        suv_global_log_max: float = 6.0,
    ):
        super().__init__()

        self.vae = vae
        self.kl_weight = kl_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.suv_global_log_max = suv_global_log_max

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        
        # Optimisation standard activée
        self.save_hyperparameters(ignore=["vae"])


    def _normalize(self, suv: torch.Tensor) -> torch.Tensor:
        """SUV → espace log normalisé [-1, 1]."""
        log = torch.log1p(suv)
        return 2.0 * (log.clamp(0, self.suv_global_log_max) / self.suv_global_log_max) - 1.0

    def _denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Espace log normalisé [-1, 1] → SUV."""
        log = 0.5 * (x_norm.clamp(-1, 1) + 1.0) * self.suv_global_log_max
        return torch.expm1(log)

    def configure_optimizers(self):
        """Un seul optimiseur pour tout le réseau."""
        optimizer = torch.optim.AdamW(
            self.vae.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hat, *_ = self.vae(x, sample_posterior=False)
        return x_hat


    def training_step(self, batch, batch_idx):
        suv_source = batch["source"][tio.DATA].float()

        if suv_source.ndim == 5:
            suv_source = suv_source.squeeze(1)

        x = self._normalize(suv_source)
        bs = x.shape[0]

        # Forward avec les mêmes probabilités de dropout que l'original
        x_hat, mu_content, logvar_content, mu_style, logvar_style = self.vae(
            x, sample_posterior=True, style_dropout_p=0.
        )

        # Calcul des losses standards
        loss_rec = F.l1_loss(x_hat, x)
        loss_kl_content = kl_loss_spatial(mu_content, logvar_content).mean()
        loss_kl_style = kl_loss_1d(mu_style, logvar_style).mean()
        loss_kl = torch.mean(loss_kl_content + loss_kl_style)

        loss_total = loss_rec + self.kl_weight * loss_kl

        # Logging
        self.log("train/rec_loss", loss_rec, on_step=True, on_epoch=True, batch_size=bs, sync_dist=True)
        self.log("train/kl_content", loss_kl_content, on_step=True, on_epoch=True, batch_size=bs, sync_dist=True)
        self.log("train/kl_style", loss_kl_style, on_step=True, on_epoch=True, batch_size=bs, sync_dist=True)
        self.log("train/total", loss_total, on_step=True, on_epoch=True, batch_size=bs, prog_bar=True, sync_dist=True)

        return loss_total

    # ──────────────────────────────────────────────────────────────────────────
    # Boucle de validation
    # ──────────────────────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        suv_source = batch["source"][tio.DATA].float()

        if suv_source.ndim == 5:
            suv_source = suv_source.squeeze(1)

        x = self._normalize(suv_source)
        bs = x.shape[0]

        # Forward d'évaluation (dropout activé comme dans l'original pour forcer l'usage du content)
        x_hat, mu_content, logvar_content, mu_style, logvar_style = self.vae(
            x, sample_posterior=False, style_dropout_p=0.95
        )

        loss_rec = F.l1_loss(x_hat, x)
        loss_kl_content = kl_loss_spatial(mu_content, logvar_content).mean()
        loss_kl_style = kl_loss_1d(mu_style, logvar_style).mean()

        # SSIM
        x_01 = (x.clamp(-1, 1) + 1.0) / 2.0
        x_hat_01 = (x_hat.clamp(-1, 1) + 1.0) / 2.0
        ssim_score = self.ssim(x_hat_01, x_01)

        # Score composite de base (uniquement sur la reco car pas de classifieurs)
        composite_score = loss_rec 

        self.log("val/rec_loss", loss_rec, batch_size=bs, sync_dist=True)
        self.log("val/kl_content", loss_kl_content, batch_size=bs, sync_dist=True)
        self.log("val/kl_style", loss_kl_style, batch_size=bs, sync_dist=True)
        self.log("val/ssim", ssim_score, batch_size=bs, sync_dist=True)
        self.log("val/composite_score", composite_score, batch_size=bs, sync_dist=True, prog_bar=True)

        if batch_idx == 0:
            self._log_images(x, x_hat, suv_source, z_content)

        return loss_rec

    # ──────────────────────────────────────────────────────────────────────────
    # Logging images
    # ──────────────────────────────────────────────────────────────────────────

    def _log_images(self, x_norm: torch.Tensor, x_hat_norm: torch.Tensor, suv_source: torch.Tensor):
        if self.trainer.global_rank != 0:
            return

        suv_pred = self._denormalize(x_hat_norm)

        mid = suv_source.shape[1] // 2
        src_slice = suv_source[:, mid:mid+1, :, :]
        pred_slice = suv_pred[:, mid:mid+1, :, :]

        display_max = max(5.0, src_slice.max().item(), pred_slice.max().item())
        imgs = torch.cat([src_slice, pred_slice], dim=3)
        imgs = (imgs / display_max).clamp(0, 1)

        grid = make_grid(imgs, nrow=1, padding=2)
        wandb.log({
            "Validation/Reconstruction_Ablation": wandb.Image(
                grid.permute(1, 2, 0).cpu().numpy(),
                caption=f"Gauche : PET | Droite : Harmonisée (Ablation) (epoch {self.current_epoch})"
            )
        })
