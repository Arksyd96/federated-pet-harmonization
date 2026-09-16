import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import wandb

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from pet_harmonization.data import MultiDomainUnlearningDataModule
from pet_harmonization.models.fft import LearnableFFTHighPassFilter
from pet_harmonization.models.harmonization_vae import BifurcatedContentStyleEncoder, DisentangledHarmonizationVAE
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
    "architecture": "vae_encoder", # 'resnet' ou 'vae_encoder'
    "in_channels": 1,       # 1 = image seule, 2 = image + FFT
    "use_fft": False,       # Concatène les features FFT en entrée
    "fft_sigma": 7.5,
    "fft_learnable": False, # False = filtre fixe (pas de gradient)

    # Entraînement
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "max_epochs": 100,
    "precision": "bf16-mixed",
    "limit_train_batches": 250,
    "limit_val_batches": 50,

    # WandB / Sauvegarde
    "project_name": "federated-pet",
    "run_name": "Center Classifier (VAE Encoder) — Baseline",
    "save_dir": "runs/site_classifier/",
}


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
    def __init__(self, in_channels: int, num_classes: int, use_fft: bool = False, fft_sigma: float = 7.5, latent_channels: int = 8):
        super().__init__()
        
        # Hyperparamètres identiques au YAML complet
        hidden_channels = [32, 64, 128, 256]
        strides = [[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2]]
        normalization = ('batch', {})
        
        # ── 1. VAE Global Complet (Encodeur bifurqué + Embedder + Décodeur AdaIN)
        self.vae = DisentangledHarmonizationVAE(
            input_shape=(16, 64, 64),
            fft_sigma=fft_sigma,
            in_channels=in_channels,
            out_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_sizes=[3, 3, 3, 3],
            strides=strides,
            latent_channels=latent_channels,
            style_channels=256,
            style_embedding_dim=256,
            spatial_dims=3,
            use_fft=use_fft,
            normalization=normalization
        )
        
        # ── 2. Classifieur de domaine sur mu_c ────────────────────────────────
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_channels, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        
    def forward(self, x):
        # Forward du VAE global (sample_posterior=False pour obtenir mu_c et mu_s déterministes)
        x_hat, kl_vars_c, kl_vars_s, z_content, z_style = self.vae(x, sample_posterior=False)
        mu_c, logvar_c = kl_vars_c
        
        # Classification du domaine à partir du contenu
        pooled = self.pool(mu_c)
        logits = self.classifier(pooled)
        
        return logits, x_hat


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
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        suv_global_log_max: float = 6.0,
        rec_weight: float = 1.0,     # Poids de la loss de reconstruction totale
        ssim_weight: float = 0.5,    # Poids du SSIM dans la rec_loss
    ):
        super().__init__()
        self.save_hyperparameters()
        self.suv_global_log_max = suv_global_log_max
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_fft = use_fft
        self.architecture = architecture
        self.rec_weight = rec_weight
        self.ssim_weight = ssim_weight

        if architecture == "vae_encoder":
            from torchmetrics.image import StructuralSimilarityIndexMeasure
            self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        # Note: L'architecture VAE Encoder gère sa propre FFT en interne via use_fft.
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
                latent_channels=8
            )
        else:
            raise ValueError(f"Architecture inconnue: {architecture}")

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

        if suv.ndim == 5 and suv.shape[1] == 1:
            suv = suv.squeeze(1)

        x = self._normalize(suv)
        bs = x.shape[0]

        if self.architecture == "vae_encoder":
            logits, x_hat = self(x)
            loss_clf = F.cross_entropy(logits, domain_labels)
            
            # La cible de reconstruction est x (1 canal). 
            # (La concaténation FFT est faite en interne du VAE mais il reconstruit 1 seul canal)
            x_target_01 = (x.clamp(-1, 1) + 1.0) / 2.0
            x_hat_01 = (x_hat.clamp(-1, 1) + 1.0) / 2.0
            
            loss_l1 = F.l1_loss(x_hat, x)
            loss_ssim = 1.0 - self.ssim(x_hat_01, x_target_01)
            loss_rec = loss_l1 + self.ssim_weight * loss_ssim
            
            loss = loss_clf + self.rec_weight * loss_rec
            
            self.log(f"{prefix}/ce_loss", loss_clf, batch_size=bs, prog_bar=True, sync_dist=True)
            self.log(f"{prefix}/rec_loss", loss_rec, batch_size=bs, prog_bar=True, sync_dist=True)
            self.log(f"{prefix}/l1_loss", loss_l1, batch_size=bs)
            self.log(f"{prefix}/ssim_loss", loss_ssim, batch_size=bs)
        else:
            logits = self(x)
            loss = F.cross_entropy(logits, domain_labels)
            self.log(f"{prefix}/ce_loss", loss, batch_size=bs, prog_bar=True, sync_dist=True)

        acc = (logits.argmax(dim=1) == domain_labels).float().mean()
        self.log(f"{prefix}/accuracy", acc, batch_size=bs, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
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
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        suv_global_log_max=cfg["suv_global_log_max"],
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
        filename="best-classifier-{epoch:02d}-{val/accuracy:.3f}",
        monitor="val/accuracy",
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

