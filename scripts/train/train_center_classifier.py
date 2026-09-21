import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import wandb
from typing import List, Tuple, Union

from pet_harmonization.models.base import (
    BasicBlock,
    BasicDown,
    UnetBasicBlock,
    UnetResBlock,
    SequentialEmb,
)
from pet_harmonization.models.attention import Attention

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from pet_harmonization.data import MultiDomainUnlearningDataModule
from pet_harmonization.models.fft import LearnableFFTHighPassFilter
from pet_harmonization.models.harmonization_vae import (
    StyleEmbedder,
    StyleConditionedDecoder,
    kl_loss_spatial,
    kl_loss_1d,
    reparameterize
)
from pet_harmonization.utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy("file_system")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG (modifiable ici directement)
# ═══════════════════════════════════════════════════════════════════════════════

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
    "architecture": "vae_encoder",  # Test de l'encodeur + KLD (sans décodeur)
    "in_channels": 1,               # 1 = image seule, 2 = image + FFT
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
    "latent_channels": 64,    # Élargissement du goulot pour aider la généralisation du classifieur
    "clf_weight": 1.0,         # Poids de la classification
    "kld_weight": 5e-7,        # Retour à 5e-7
    "rec_weight": 1.0,         # Poids final de la reconstruction
    "ssim_weight": 0.5,        # Poids du SSIM
    "warmup_epochs": 40,       # Epochs pour le warmup progressif de la reconstruction (0.1 -> 1.0)
    
    # WandB / Sauvegarde
    "project_name": "federated-pet",
    "run_name": "Center Classifier (VAE Encoder) — 64 Channels + Decoupled + Vibrant z_c",
    "save_dir": "runs/site_classifier/",
}


class BifurcatedContentStyleEncoder(nn.Module):
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



# ═══════════════════════════════════════════════════════════════════════════════
# ResNet3D léger pour classification de patchs
# ═══════════════════════════════════════════════════════════════════════════════

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.act = nn.SiLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + self.shortcut(x))
        return out


class PatchResNet3D(nn.Module):
    """
    Petit ResNet3D adapté aux patchs anisotropiques (16×64×64).
    Strides anisotropiques pour ne pas écraser la dimension Z trop vite.
    """
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.SiLU(inplace=True),
        )
        # Strides anisotropiques : (Z, H, W)
        # 16×64×64 → 16×32×32 → 8×16×16 → 4×8×8 → 2×4×4
        self.layer1 = ResBlock3D(32, 64,   stride=(1, 2, 2))
        self.layer2 = ResBlock3D(64, 128,  stride=(2, 2, 2))
        self.layer3 = ResBlock3D(128, 256, stride=(2, 2, 2))
        self.layer4 = ResBlock3D(256, 256, stride=(2, 2, 2))

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.classifier(x)


class GlobalVAEClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, use_fft: bool = False, fft_sigma: float = 7.5, latent_channels: int = 64):
        super().__init__()
        
        hidden_channels = [32, 64, 128, 256]
        
        # ── 1. Encodeur complet ───────────────────────────────────────────────
        self.encoder = BifurcatedContentStyleEncoder(
            input_shape=(16, 64, 64),
            fft_sigma=fft_sigma,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_sizes=[3, 3, 3, 3],
            strides=[[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2]],
            latent_channels=latent_channels,
            style_channels=256,
            spatial_dims=3,
            use_fft=use_fft,
            normalization=('batch', {})
        )
        
        # ── 2. Décodeur et conditionnement de style ───────────────────────────
        self.content_norm = nn.InstanceNorm3d(latent_channels, affine=False)
        self.style_embedder = StyleEmbedder(
            style_channels=256,
            style_embedding_dim=256,
        )
        self.decoder = StyleConditionedDecoder(
            latent_channels=latent_channels,
            out_channels=in_channels,
            style_embedding_dim=256,
            hidden_channels=hidden_channels,
            kernel_sizes=[3, 3, 3, 3],
            strides=[[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2]],
            spatial_dims=3,
            normalization=('instance', {})
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
        
    def forward(self, x):
        # 1. Encodage (la classe locale retourne aussi h_c, on l'ignore ici)
        kl_vars_c, kl_vars_s, _h_c = self.encoder(x)
        mu_c, logvar_c = kl_vars_c
        mu_s, logvar_s = kl_vars_s
        
        # 2. Reparamétrisation
        z_c = reparameterize(mu_c, logvar_c)
        z_s = reparameterize(mu_s, logvar_s)
        
        # 3. Normalisation du contenu & Embedding du style
        z_c_norm = self.content_norm(z_c)
        style_emb = self.style_embedder(z_s)
        
        # 4. Reconstruction
        x_hat = self.decoder(z_c_norm, style_emb)
        
        # 5. Classification content sur z_c
        pooled = self.pool(z_c)
        logits_content = self.content_classifier(pooled)
        
        # 6. Classification style sur z_s
        logits_style = self.style_classifier(z_s)
        
        return logits_content, logits_style, x_hat, mu_c, logvar_c, mu_s, logvar_s


# ═══════════════════════════════════════════════════════════════════════════════
# Module Lightning
# ═══════════════════════════════════════════════════════════════════════════════

class CenterClassifier(LightningModule):
    def __init__(
        self,
        architecture: str = "resnet",
        num_domains: int = 5,
        in_channels: int = 1,
        use_fft: bool = True,
        fft_sigma: float = 7.5,
        fft_learnable: bool = False,
        input_shape: tuple = (16, 64, 64),
        lr_vae: float = 1e-4,
        lr_clf: float = 1e-5,
        weight_decay: float = 1e-5,
        suv_global_log_max: float = 6.0,
        latent_channels: int = 256,
        clf_weight: float = 1.0,     # Base : 1.0. A tester : 10.0, 50.0...
        kld_weight: float = 5e-7,
        rec_weight: float = 1.0,     # Poids de la reconstruction
        ssim_weight: float = 0.5,    # Poids du SSIM dans la reconstruction
        warmup_epochs: int = 0,      # Base : 0. A tester : 5, 10...
    ):
        super().__init__()
        self.save_hyperparameters()
        self.suv_global_log_max = suv_global_log_max
        self.lr_vae = lr_vae
        self.lr_clf = lr_clf
        self.weight_decay = weight_decay
        self.use_fft = use_fft
        self.architecture = architecture
        self.clf_weight = clf_weight
        self.kld_weight = kld_weight
        self.rec_weight = rec_weight
        self.ssim_weight = ssim_weight
        self.warmup_epochs = warmup_epochs

        if architecture == "vae_encoder":
            from torchmetrics.image import StructuralSimilarityIndexMeasure
            self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        # Pour le ResNet, on doit l'ajouter explicitement ici si demandé.
        actual_in_channels = in_channels
        if use_fft and architecture == "resnet":
            self.fft_filter = LearnableFFTHighPassFilter(
                input_shape=input_shape,
                in_channels=in_channels,
                learnable=fft_learnable,
                sigma=fft_sigma,
                spatial_dims=3,
            )
            if not fft_learnable:
                for p in self.fft_filter.parameters():
                    p.requires_grad = False
            actual_in_channels = in_channels * 2

        if architecture == "resnet":
            self.model = PatchResNet3D(
                in_channels=actual_in_channels,
                num_classes=num_domains,
            )
        elif architecture == "vae_encoder":
            self.model = GlobalVAEClassifier(
                in_channels=in_channels,
                num_classes=num_domains,
                use_fft=use_fft,
                fft_sigma=fft_sigma,
                latent_channels=latent_channels
            )
        else:
            raise ValueError(f"Architecture inconnue : {architecture}")

    def _normalize(self, suv: torch.Tensor) -> torch.Tensor:
        log = torch.log1p(suv)
        return 2.0 * (log.clamp(0, self.suv_global_log_max) / self.suv_global_log_max) - 1.0

    def forward(self, x):
        if self.use_fft and self.architecture == "resnet":
            fft_x = self.fft_filter(x)
            x = torch.cat([x, fft_x], dim=1)
        return self.model(x)

    def _shared_step(self, batch, prefix: str):
        suv = batch["source"][tio.DATA].float()
        domain_labels = batch["domain_id"]

        # Suppression du squeeze(1) qui transformait (B, 1, D, H, W) en (B, D, H, W).
        # PyTorch Conv3d interprétait (B, D, H, W) comme un tenseur non-batché (C, D, H, W)
        # ce qui faisait que C devenait égal au batch size (ex: 16) !

        x = self._normalize(suv)
        bs = x.shape[0]

        if self.architecture == "vae_encoder":
            logits_content, logits_style, x_hat, mu_c, logvar_c, mu_s, logvar_s = self(x)
            # Classification content (sur z_c) et style (sur z_s)
            loss_clf_content = F.cross_entropy(logits_content, domain_labels)
            loss_clf_style = F.cross_entropy(logits_style, domain_labels)
            loss_clf = loss_clf_content + loss_clf_style
            
            # KLD loss
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
            
        else:
            # Mode ResNet standard
            logits = self(x)
            loss_clf = F.cross_entropy(logits, domain_labels)
            loss = loss_clf
            acc_content = (logits.argmax(1) == domain_labels).float().mean()
            self.log(f"{prefix}/ce_loss", loss_clf, batch_size=bs, prog_bar=True, sync_dist=True)
            self.log(f"{prefix}/acc_content", acc_content, batch_size=bs, prog_bar=True, sync_dist=True)

        self.log(f"{prefix}/loss", loss, batch_size=bs, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        if self.architecture == "vae_encoder":
            # Séparation des paramètres du VAE et des Classifieurs
            clf_params = list(self.model.content_classifier.parameters()) + list(self.model.style_classifier.parameters())
            # Tous les autres paramètres (Encodeur, Décodeur, etc.)
            clf_param_ids = [id(p) for p in clf_params]
            vae_params = [p for p in self.parameters() if id(p) not in clf_param_ids]

            return torch.optim.AdamW([
                {'params': vae_params, 'lr': self.lr_vae},
                {'params': clf_params, 'lr': self.lr_clf}
            ], weight_decay=self.weight_decay)
        else:
            return torch.optim.AdamW(
                self.parameters(),
                lr=self.lr_vae,
                weight_decay=self.weight_decay,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import os
    os.environ["WANDB_API_KEY"] = "bdc8857f9d6f7010cff35bcdc0ae9413e05c75e1"
    
    cfg = CONFIG
    set_seed(cfg["seed"], workers=True)

    # ── DataModule ────────────────────────────────────────────────────────
    datamodule = MultiDomainUnlearningDataModule(**cfg["datamodule"])

    # ── Modèle ────────────────────────────────────────────────────────────
    model = CenterClassifier(
        architecture=cfg["architecture"],
        num_domains=cfg["num_domains"],
        in_channels=cfg["in_channels"],
        use_fft=cfg["use_fft"],
        fft_sigma=cfg["fft_sigma"],
        fft_learnable=cfg["fft_learnable"],
        input_shape=tuple(cfg["datamodule"]["patch_size"]),
        lr_vae=cfg["lr_vae"],
        lr_clf=cfg["lr_clf"],
        weight_decay=cfg["weight_decay"],
        suv_global_log_max=cfg["suv_global_log_max"],
        latent_channels=cfg["latent_channels"],
        clf_weight=cfg["clf_weight"],
        kld_weight=cfg["kld_weight"],
        rec_weight=cfg["rec_weight"],
        ssim_weight=cfg["ssim_weight"],
        warmup_epochs=cfg["warmup_epochs"],
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
        gradient_clip_val=1.0,  # 🚨 FIX NaN: Empêche l'explosion des gradients de l'AdaIN
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


if __name__ == "__main__":
    main()

