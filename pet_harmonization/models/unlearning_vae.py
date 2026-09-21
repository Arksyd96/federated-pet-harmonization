import resource
import os
from pet_harmonization.data import MultiDomainUnlearningDataModule
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import wandb
import math
from typing import Dict, List, Tuple, Union

from monai.networks.blocks import UnetOutBlock
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid


from pet_harmonization.models.attention import Attention, zero_module
from pet_harmonization.models.fft import LearnableFFTHighPassFilter
from pet_harmonization.utils import set_seed
from pet_harmonization.models.base import (
    BasicBlock,
    BasicUp,
    BasicDown,
    UnetBasicBlock,
    UnetResBlock,
    SequentialEmb,
    save_add,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy("file_system")


CONFIG = {
    "seed": 101,
    "num_domains": 5,
    "suv_global_log_max": 6.0,

    # Datamodule (même config que le VAE)
    "datamodule": {
        "root_dir": "./data/PET-EARL/",
        "split_config": [[60, 15], [60, 15], [0, 0], [60, 15], [60, 15], [60, 15]],
        "batch_size": 16,
        "patch_size": [16, 64, 64],
        "num_workers": 24,
        "queue_max_length": 4096,
        "samples_per_volume": 64,
    },

    # Modèle
    "input_shape": (16, 64, 64),
    "in_channels": 1,               # 1 = image seule, 2 = image + FFT
    "out_channels": 1,
    "hidden_channels": [32, 64, 128, 256],
    "kernel_sizes": [3, 3, 3, 3],
    "strides": [[1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2]],
    "num_residual_blocks": 1,
    "latent_channels": 64,
    "style_channels": 256,
    "use_fft": False,       # Concatène les features FFT en entrée
    "fft_sigma": 7.5,
    "fft_learnable": False, # False = filtre fixe (pas de gradient)

    # Entraînement
    "lr_vae": 1e-4,
    "lr_clf": 1e-5,
    "weight_decay": 1e-6,
    "max_epochs": 120,
    "precision": "bf16-mixed",
    "limit_train_batches": 250,
    "limit_val_batches": 50,
    
    # Paramètres d'ablation pour le classifieur
    "clf_weight": 1.0,         # Poids de la classification
    "kld_weight": 5e-7,        # Retour à 5e-7
    "rec_weight": 1.0,         # Poids final de la reconstruction
    "ssim_weight": 0.5,        # Poids du SSIM
    "warmup_epochs": 40,       # Epochs pour le warmup progressif de la reconstruction (0.1 -> 1.0)
    "k_style_steps": 2,
    
    # WandB / Sauvegarde
    "project_name": "federated-pet",
    "run_name": "Center Classifier (VAE Encoder) — 64 Channels + Decoupled + Vibrant z_c",
    "save_dir": "runs/site_classifier/",
}


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Echantillonnage reparamétrisé : z = mu + eps * std."""
    logvar = torch.clamp(logvar, min=-20.0, max=10.0)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mu)
    return mu + eps * std


def kl_loss_spatial(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q||N(0,I)) pour un posterior spatial (B, C, H, W). Retourne [B]."""
    logvar = torch.clamp(logvar, min=-20.0, max=10.0)
    return 0.5 * torch.sum(
        mu.pow(2) + logvar.exp() - 1.0 - logvar,
        dim=list(range(1, mu.dim())),
    )


def kl_loss_1d(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q||N(0,I)) pour un posterior 1D (B, D). Retourne [B]."""
    logvar = torch.clamp(logvar, min=-20.0, max=10.0)
    return 0.5 * torch.sum(
        mu.pow(2) + logvar.exp() - 1.0 - logvar,
        dim=1,
    )


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
    

class ContentStyleEncoder(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        fft_sigma: float = 7.5,
        in_channels: int = 5,
        hidden_channels: List[int] = [64, 128, 256, 512],
        kernel_sizes: List[int] = [3, 3, 3, 3],
        strides: List[int] = [1, 2, 2, 2],
        latent_channels: int = 64,
        style_channels: int = 256,
        num_residual_blocks: int = 1,
        spatial_dims: int = 2,
        normalization: Tuple = ('group', {'num_groups': 32, 'affine': True}),
        activation: Tuple = ('swish', {}),
        dropout: float = 0.0,
        use_residual_block: bool = True,
        learnable_interpolation: bool = True,
        attention_type: Union[str, List[str]] = 'none',
        use_fft: bool = True
    ):
        super().__init__()

        self.depth = len(hidden_channels)
        self.num_residual_blocks = num_residual_blocks
        self.use_fft = use_fft

        AdaptiveMaxPool = getattr(nn, f"AdaptiveMaxPool{spatial_dims}d")
        self.pool = AdaptiveMaxPool(1)
        self.flatten = nn.Flatten()

        attention_type = (
            attention_type if isinstance(attention_type, list)
            else [attention_type] * self.depth
        )
        ConvBlock = UnetResBlock if use_residual_block else UnetBasicBlock

        # ── FFT Filter ────────────────────────────────────────────────────────
        if self.use_fft:
            self.fft_filter = LearnableFFTHighPassFilter(
                input_shape, in_channels=in_channels, sigma=fft_sigma, spatial_dims=spatial_dims
            )

        # ── In-Convolution (Branches Indépendantes dès le départ) ────────────────
        input_dim = in_channels * 2 if self.use_fft else in_channels
        self.content_input_conv = BasicBlock(
            spatial_dims, input_dim, hidden_channels[0],
            kernel_size=kernel_sizes[0], stride=strides[0],
        )
        self.style_input_conv = BasicBlock(
            spatial_dims, input_dim, hidden_channels[0],
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
        
        # ── Initialisation Gaussienne des poids ──────────────────────────────
        self.apply(self._init_weights)

    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1 and hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("Norm") != -1 and hasattr(m, 'weight') and m.weight is not None:
            # Batch/Instance/Group/Layer Norm
            nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("Linear") != -1 and hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

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
        # ── Inputs ───────────────────────────────────────────────────────────
        if self.use_fft:
            fft_x = self.fft_filter(x)
            inp = torch.cat([x, fft_x], dim=1)
        else:
            inp = x
        
        # ── Branche Content ──────────────────────────────────────────────────
        h_c = self.content_input_conv(inp)
        for block in self.content_encoder_blocks:
            h_c = block(h_c, None)
        h_c = self.content_middle_block(h_c, None)

        # ── Branche Style ────────────────────────────────────────────────────
        h_s = self.style_input_conv(inp)
        for block in self.style_encoder_blocks:
            h_s = block(h_s, None)
        h_s = self.style_middle_block(h_s, None)

        # ── Content head ─────────────────────────────────────────────────────
        moments_c = self.content_head(h_c)
        mu_c, logvar_c = moments_c.chunk(2, dim=1)
        
        # ── Style head ───────────────────────────────────────────────────────
        moments_s = self.style_head(h_s)
        moments_s = self.flatten(self.pool(moments_s))
        mu_s, logvar_s = moments_s.chunk(2, dim=1)

        return (mu_c, logvar_c), (mu_s, logvar_s), h_c


    
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
        normalization:      Tuple = ('layer', {}),
        activation:         Tuple = ('swish', {}),
        dropout:            float = 0.0,
        use_residual_block: bool = True,     
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


class DisentangledVAE(nn.Module):
    def __init__(
        self, 
        num_classes: int,
        input_shape: Tuple[int, int, int] = (16, 64, 64),
        in_channels: int = 1, 
        out_channels: int = 1,
        hidden_channels: List[int] = [32, 64, 128, 256],
        kernel_sizes: List[int] = [3, 3, 3, 3],
        strides: List[int] = [[1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2]],
        num_residual_blocks: int = 1,
        use_fft: bool = False, 
        fft_sigma: float = 7.5, 
        latent_channels: int = 64,
        style_channels: int = 256,
        spatial_dims: int = 3,
        normalization: Tuple = ('batch', {}),
        activation: Tuple = ('swish', {}),
        use_residual_block: bool = True,
        learnable_interpolation: bool = True,
        attention_type: Union[str, List[str]] = 'none',
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
            use_residual_block=use_residual_block,
            learnable_interpolation=learnable_interpolation,
            attention_type=attention_type,
        )

        # 1. Encoder
        self.encoder = ContentStyleEncoder(
            input_shape=input_shape,
            fft_sigma=fft_sigma,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            latent_channels=latent_channels,
            style_channels=style_channels,
            spatial_dims=spatial_dims,
            use_fft=use_fft,
            normalization=normalization
        )

        # 2. Bottleneck
        self.content_norm = nn.InstanceNorm3d(latent_channels, affine=False)
        self.style_embedder = StyleEmbedder(
            style_channels=style_channels,
            style_embedding_dim=style_channels,
        )

        # 3. Decoder
        self.decoder = StyleConditionedDecoder(
            latent_channels=latent_channels,
            out_channels=out_channels,
            style_embedding_dim=style_channels,
            hidden_channels=hidden_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            spatial_dims=spatial_dims,
            normalization=('instance', {})
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (mu_c, logvar_c), (mu_s, logvar_s), _ = self.encoder(x)
        return mu_c, logvar_c, mu_s, logvar_s
    
    def decode(self, z_content: torch.Tensor, z_style: torch.Tensor) -> torch.Tensor:
        style_emb = self.style_embedder(z_style)
        return self.decoder(z_content, style_emb)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_c, logvar_c, mu_s, logvar_s = self.encode(x)

        z_c = reparameterize(mu_c, logvar_c)
        z_s = reparameterize(mu_s, logvar_s)

        norm_z_c = self.content_norm(z_c)

        x_hat = self.decode(norm_z_c, z_s)

        return x_hat, mu_c, logvar_c, mu_s, logvar_s, z_c, z_s
    

class UnlearningVAE(LightningModule):
    def __init__(
        self,
        num_classes: int = 5,

        input_shape: tuple = (16, 64, 64),
        in_channels: int = 1,
        hidden_channels: list = [32, 64, 128, 256],
        kernel_sizes: list = [3, 3, 3, 3],
        strides: list = [[1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2]],
        num_residual_blocks: int = 1,
        latent_channels: int = 64,
        style_channels: int = 256,
        
        use_fft: bool = True,
        fft_sigma: float = 7.5,
        
        lr_vae: float = 1e-4,
        lr_clf: float = 1e-5,
        lr_unlearn: float = 1e-5,
        lr_classifiers_final: float = 1e-5,
        lr_unlearn_final: float = 1e-7,
        weight_decay: float = 1e-5,
        clf_weight: float = 1.0,     # Base : 1.0. A tester : 10.0, 50.0...
        kld_weight: float = 5e-7,
        rec_weight: float = 1.0, 
        ssim_weight: float = 0.5,
        warmup_epochs: int = 0,      # Base : 0. A tester : 5, 10...
        max_epochs: int = 100,
        k_style_steps: int = 2,

        suv_global_log_max: float = 6.0
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.suv_global_log_max = suv_global_log_max
        self.lr_vae = lr_vae
        self.lr_clf = lr_clf
        self.lrs = {"vae": lr_vae, "classifiers": lr_clf, "unlearn": lr_unlearn}
        self.weight_decay = weight_decay
        self.use_fft = use_fft
        self.clf_weight = clf_weight
        self.kld_weight = kld_weight
        self.rec_weight = rec_weight
        self.ssim_weight = ssim_weight
        self.warmup_epochs = warmup_epochs
        self.k_style_steps = k_style_steps
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        
        self.vae = DisentangledVAE(
            num_classes=num_classes,
            input_shape=input_shape,
            in_channels=in_channels,
            out_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            num_residual_blocks=num_residual_blocks,
            use_fft=use_fft,
            fft_sigma=fft_sigma,
            latent_channels=latent_channels,
            style_channels=style_channels,
            spatial_dims=len(input_shape),
            normalization=('batch', {}),
            activation=('swish', {}),
            use_residual_block=True,
            learnable_interpolation=True,
            attention_type='none',
        )

        # ── 3. Classifieur de domaine CONTENT sur z_c ─────────────────────────
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.content_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_channels, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, num_classes)
        )
        
        # ── 4. Classifieur de domaine STYLE sur z_s ──────────────────────────
        self.style_classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, num_classes)
        )

    
    def _vae_parameters(self):
        return list(self.vae.parameters())

    def _classifier_parameters(self):
        return (
            list(self.style_classifier.parameters())
            + list(self.content_classifier.parameters())
        )

    def _encoder_parameters(self):
        return list(self.vae.encoder.parameters())

    def _normalize(self, suv: torch.Tensor) -> torch.Tensor:
        log = torch.log1p(suv)
        return 2.0 * (log.clamp(0, self.suv_global_log_max) / self.suv_global_log_max) - 1.0
    
    def _denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        log = 0.5 * (x_norm.clamp(-1, 1) + 1.0) * self.suv_global_log_max
        return torch.expm1(log)

    def _kl_confusion_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """KL(pred || uniforme) — pousse le classifieur vers l'incertitude maximale."""
        uniform = torch.full_like(logits, 1.0 / self.num_classes)
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
    # schedulers
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
    
    def _scheduled_rec_weight(self) -> float:
        """Warmup progressif de la reconstruction : de 0.1 à rec_weight sur warmup_epochs."""
        if self.warmup_epochs > 0 and self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            return 0.1 + progress * (self.rec_weight - 0.1)
        return self.rec_weight

    def _is_warmup(self) -> bool:
        if isinstance(self.warmup_epochs, float):
            iters_per_epoch = self.trainer.num_training_batches
            return self.global_step < int(self.warmup_epochs * iters_per_epoch)
        return self.current_epoch < self.warmup_epochs

    # def on_train_epoch_start(self):
    #     """Met à jour beta et lr_classifiers au début de chaque époque du stage 2."""
    #     if self._is_warmup():
    #         return

    #     _, opt_style_clf, opt_content_clf, opt_unlearn = self.optimizers()
        
    #     new_lr_clf = self._cosine_lr(self.lrs["classifiers"], self.hparams.lr_classifiers_final)
    #     for opt in [opt_style_clf, opt_content_clf]:
    #         for pg in opt.param_groups:
    #             pg["lr"] = new_lr_clf

    #     new_lr_unlearn = self._cosine_lr(self.lrs["unlearn"], self.hparams.lr_unlearn_final)
    #     for pg in opt_unlearn.param_groups:
    #         pg["lr"] = new_lr_unlearn

    #     self.log("debug/lr_classifiers", new_lr_clf,    on_step=False, on_epoch=True)
    #     self.log("debug/lr_unlearn",     new_lr_unlearn, on_step=False, on_epoch=True)   


    def _shared_step(self, batch, prefix: str):
        suv = batch["source"][tio.DATA].float()
        domain_labels = batch["domain_id"]

        x = self._normalize(suv)
        bs = x.shape[0]

        x_hat, mu_c, logvar_c, mu_s, logvar_s, z_c, z_s = self.vae.forward(x)

        logits_content = self.content_classifier(self.pool(z_c))
        logits_style   = self.style_classifier(z_s)

        loss_clf_content = F.cross_entropy(logits_content, domain_labels)
        loss_clf_style = F.cross_entropy(logits_style, domain_labels)
        loss_clf = loss_clf_content + loss_clf_style

        loss_kl_content = kl_loss_spatial(mu_c, logvar_c).mean()
        loss_kl_style = kl_loss_1d(mu_s, logvar_s).mean()
        loss_kl = loss_kl_content + loss_kl_style

        # Reconstruction loss
        x_target_01 = (x.clamp(-1, 1) + 1.0) / 2.0
        x_hat_01 = (x_hat.clamp(-1, 1) + 1.0) / 2.0
        
        loss_l1 = F.l1_loss(x_hat, x)
        loss_ssim = 1.0 - self.ssim(x_hat_01, x_target_01)
        loss_rec = loss_l1 + self.ssim_weight * loss_ssim


        # Warmup progressif : de 0.1 à rec_weight sur warmup_epochs
        if self.warmup_epochs > 0 and self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            current_rec_weight = 0.1 + progress * (self.rec_weight - 0.1)
        else:
            current_rec_weight = self.rec_weight
        
        # Total loss
        loss = self.clf_weight * loss_clf + self.kld_weight * loss_kl + current_rec_weight * loss_rec

        # Accuracy content et style
        acc_content = (logits_content.argmax(1) == domain_labels).float().mean()
        acc_style = (logits_style.argmax(1) == domain_labels).float().mean()
        
        self.log(f"{prefix}/ce_content", loss_clf_content, batch_size=bs, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/ce_style", loss_clf_style, batch_size=bs, sync_dist=True)
        self.log(f"{prefix}/kl_loss", loss_kl, batch_size=bs, sync_dist=True)
        self.log(f"{prefix}/rec_loss", loss_rec, batch_size=bs, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/rec_weight", current_rec_weight, batch_size=bs)
        self.log(f"{prefix}/acc_content", acc_content, batch_size=bs, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/acc_style", acc_style, batch_size=bs, prog_bar=True, sync_dist=True)

        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, prefix="train")
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, prefix="val")
    
    def configure_optimizers(self):
        clf_params = list(self.content_classifier.parameters()) + list(self.style_classifier.parameters())
        # Tous les autres paramètres (Encodeur, Décodeur, etc.)
        clf_param_ids = [id(p) for p in clf_params]
        vae_params = [p for p in self.parameters() if id(p) not in clf_param_ids]

        return torch.optim.AdamW([
            {'params': vae_params, 'lr': self.lr_vae},
            {'params': clf_params, 'lr': self.lr_clf}
        ], weight_decay=self.weight_decay)


if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = "bdc8857f9d6f7010cff35bcdc0ae9413e05c75e1"
    
    cfg = CONFIG
    set_seed(cfg["seed"], workers=True)

    # ── DataModule ────────────────────────────────────────────────────────
    datamodule = MultiDomainUnlearningDataModule(**cfg["datamodule"])

    # ── Modèle ────────────────────────────────────────────────────────────
    model = UnlearningVAE(
        num_classes=cfg["num_domains"],
        input_shape=cfg["input_shape"],
        in_channels=cfg["in_channels"],
        hidden_channels=cfg["hidden_channels"],
        kernel_sizes=cfg["kernel_sizes"],
        strides=cfg["strides"],
        num_residual_blocks=cfg["num_residual_blocks"],
        latent_channels=cfg["latent_channels"],
        style_channels=cfg["style_channels"],
        use_fft=cfg["use_fft"],
        fft_sigma=cfg["fft_sigma"],
        lr_vae=cfg["lr_vae"],
        lr_clf=cfg["lr_clf"],
        # lr_unlearn=cfg["lr_unlearn"],
        # lr_classifiers_final=cfg["lr_classifiers_final"],
        # lr_unlearn_final=cfg["lr_unlearn_final"],
        weight_decay=cfg["weight_decay"],
        clf_weight=cfg["clf_weight"],
        kld_weight=cfg["kld_weight"],
        rec_weight=cfg["rec_weight"],
        ssim_weight=cfg["ssim_weight"],
        warmup_epochs=cfg["warmup_epochs"],
        k_style_steps=cfg["k_style_steps"],
        suv_global_log_max=cfg["suv_global_log_max"]
    )

    # ── Logger WandB ──────────────────────────────────────────────────────
    wb_logger = WandbLogger(
        save_dir=cfg["save_dir"],
        project=cfg["project_name"],
        name=cfg["run_name"],
        config=cfg,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg["save_dir"],
        filename="best-classifier-{epoch:02d}-{val/acc_content:.3f}",
        monitor="val/acc_content",
        mode="max",
        save_last=True,
        save_top_k=1,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        logger=wb_logger,
        callbacks=[checkpoint_callback],
        precision=cfg["precision"],
        accelerator="gpu",
        devices=1,
        max_epochs=cfg["max_epochs"],
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        limit_train_batches=cfg["limit_train_batches"],
        limit_val_batches=cfg["limit_val_batches"],
    )

    logger.info("Lancement classification pure des centres TEP 🔬")
    logger.info(f"  FFT: {'ON' if cfg['use_fft'] else 'OFF'} (learnable={cfg['fft_learnable']})")
    logger.info(f"  Input channels: {cfg['in_channels']} → {cfg['in_channels'] * 2 if cfg['use_fft'] else cfg['in_channels']}")
    logger.info(f"  Patch size: {cfg['datamodule']['patch_size']}")

    trainer.fit(model, datamodule=datamodule)

