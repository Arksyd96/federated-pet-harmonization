import os
import argparse
import logging
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torchmetrics.image import StructuralSimilarityIndexMeasure

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf

from pet_harmonization.data import SingleTargetPETDataModule
from pet_harmonization.models.unet import UNet
from pet_harmonization.models.fft import LearnableFFTHighPassFilter
from pet_harmonization.utils import set_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FourierAwareClassifier(nn.Module):
    """
    Classifieur qui prend l'image spatiale et la version filtrée (hautes fréquences) concaténées.
    Entrée: Tensor de forme (B, 2, H, W)
    """
    def __init__(self, in_channels: int = 2, num_classes: int = 5, spatial_dims: int = 2):
        super().__init__()
        self.spatial_dims = spatial_dims
        
        # Architecture CNN classique basique
        Conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        Pool = nn.MaxPool2d if spatial_dims == 2 else nn.MaxPool3d
        AdaptivePool = nn.AdaptiveAvgPool2d if spatial_dims == 2 else nn.AdaptiveAvgPool3d
        
        self.features = nn.Sequential(
            Conv(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if spatial_dims == 2 else nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),
            Pool(2),
            
            Conv(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if spatial_dims == 2 else nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            Pool(2),
            
            Conv(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if spatial_dims == 2 else nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            Pool(2),
            
            Conv(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if spatial_dims == 2 else nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            AdaptivePool(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class ResidualUnlearningSystem(LightningModule):
    def __init__(
        self,
        num_classes: int = 5,
        spatial_dims: int = 2,
        input_shape: tuple = (64, 64),
        
        # UNet Params (pour la Delta Map)
        unet_hid_chs: list = [64, 128, 256, 512],
        unet_kernel_sizes: list = [3, 3, 3, 3],
        unet_strides: list = [1, 2, 2, 2],
        
        # FFT Filter Params
        fft_sigma: float = 7.5,
        
        # Optim Params
        lr_unet: float = 1e-4,
        lr_clf: float = 1e-4,
        weight_decay: float = 1e-5,
        alpha_residual: float = 1.0,  # X_harm = X + alpha * Delta
        
        # Loss Weights
        lambda_l1: float = 1.0,
        lambda_ssim: float = 1.0,
        lambda_adv: float = 1.0,
        
        suv_global_log_max: float = 6.0
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # Optimisation manuelle cruciale pour l'adversarial
        
        # 1. Générateur : UNet
        self.unet = UNet(
            in_ch=1,
            out_ch=1,
            spatial_dims=spatial_dims,
            hid_chs=unet_hid_chs,
            kernel_sizes=unet_kernel_sizes,
            strides=unet_strides,
            temb_channels=None, # Pas de conditionnement temporel
            use_attention='none',
            num_res_blocks=2
        )
        
        # 2. Extracteur de Fréquences (Filtre Passe-Haut)
        self.fft_filter = LearnableFFTHighPassFilter(
            input_shape=input_shape,
            in_channels=1,
            learnable=False,  # On peut le geler au début pour stabiliser
            sigma=fft_sigma,
            spatial_dims=spatial_dims
        )
        
        # 3. Classifieur de Domaine (Critique)
        # in_channels=2 car [X, HPF(X)]
        self.classifier = FourierAwareClassifier(
            in_channels=2, 
            num_classes=num_classes, 
            spatial_dims=spatial_dims
        )
        
        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0)
        
    def _normalize(self, suv: torch.Tensor) -> torch.Tensor:
        log = torch.log1p(suv)
        return 2.0 * (log.clamp(0, self.hparams.suv_global_log_max) / self.hparams.suv_global_log_max) - 1.0
        
    def _confusion_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Maximise l'entropie du classifieur."""
        p = F.softmax(logits, dim=1)
        return -torch.log(p + 1e-8).mean()

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.unet.parameters(), 
            lr=self.hparams.lr_unet, 
            weight_decay=self.hparams.weight_decay
        )
        opt_d = torch.optim.AdamW(
            self.classifier.parameters(), 
            lr=self.hparams.lr_clf, 
            weight_decay=self.hparams.weight_decay
        )
        return [opt_g, opt_d]

    def _prepare_classifier_input(self, x_spatial: torch.Tensor) -> torch.Tensor:
        """Filtre l'image spatialement via FFT et concatène."""
        x_hpf = self.fft_filter(x_spatial)
        return torch.cat([x_spatial, x_hpf], dim=1)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        
        suv_source = batch["source"][tio.DATA].float()
        domain_labels = batch["domain_id"]  # LongTensor [B]
        
        if self.hparams.spatial_dims == 2 and suv_source.ndim == 5:
            suv_source = suv_source.squeeze(1)
            
        x = self._normalize(suv_source)
        
        # --- Passage Forward ---
        delta_x = self.unet(x, t=None, condition=None)
        x_harm = x + self.hparams.alpha_residual * delta_x
        
        # =========================================================
        # PHASE 1 : Entraînement du Classifieur (Discriminator)
        # =========================================================
        # On détache x_harm pour ne pas propager le gradient dans le UNet
        x_harm_detached = x_harm.detach()
        
        # Préparation des entrées [Spatiale + Fréquentielle]
        real_input = self._prepare_classifier_input(x)
        fake_input = self._prepare_classifier_input(x_harm_detached)
        
        # Prédictions
        logits_real = self.classifier(real_input)
        logits_fake = self.classifier(fake_input)
        
        # Losses (Cross-Entropy classique : le classifieur DOIT trouver le domaine)
        loss_d_real = F.cross_entropy(logits_real, domain_labels)
        loss_d_fake = F.cross_entropy(logits_fake, domain_labels)
        loss_d = (loss_d_real + loss_d_fake) / 2.0
        
        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()
        
        # Métrique
        acc_real = (logits_real.argmax(dim=1) == domain_labels).float().mean()
        acc_fake_d = (logits_fake.argmax(dim=1) == domain_labels).float().mean()
        
        # =========================================================
        # PHASE 2 : Entraînement du Générateur (UNet)
        # =========================================================
        # Le UNet doit tromper le classifieur mis à jour ET préserver l'anatomie
        fake_input_for_g = self._prepare_classifier_input(x_harm)
        logits_fake_for_g = self.classifier(fake_input_for_g)
        
        # 1. Confusion Loss (Adversarial)
        loss_g_adv = self._confusion_loss(logits_fake_for_g)
        
        # 2. Content Loss (L1 sur Delta + SSIM)
        loss_l1 = F.l1_loss(delta_x, torch.zeros_like(delta_x))
        
        # SSIM demande des inputs dans [0, 1]
        x_01 = (x.clamp(-1, 1) + 1.0) / 2.0
        x_harm_01 = (x_harm.clamp(-1, 1) + 1.0) / 2.0
        loss_ssim = 1.0 - self.ssim_loss(x_harm_01, x_01)
        
        # Loss Totale Générateur
        loss_g = (
            self.hparams.lambda_adv * loss_g_adv +
            self.hparams.lambda_l1 * loss_l1 +
            self.hparams.lambda_ssim * loss_ssim
        )
        
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        opt_g.step()
        
        acc_fake_g = (logits_fake_for_g.argmax(dim=1) == domain_labels).float().mean()
        
        # --- Loggings ---
        self.log("train_D/loss", loss_d, prog_bar=True)
        self.log("train_D/acc_real", acc_real, prog_bar=True)
        self.log("train_D/acc_fake", acc_fake_d)
        
        self.log("train_G/loss_total", loss_g, prog_bar=True)
        self.log("train_G/loss_adv", loss_g_adv)
        self.log("train_G/loss_l1", loss_l1)
        self.log("train_G/loss_ssim", loss_ssim)
        self.log("train_G/acc_fake_post_G", acc_fake_g) # Devrait chuter vers 1/N
        

def main():
    # --- Configuration Bac à Sable (Modifiez ici pour vos tests) ---
    cfg = {
        'SEED': 42,
        
        # Datamodule params (à ajuster selon votre setup local)
        'datamodule': {
            'data_dir': 'data/processed/',
            'batch_size': 4,
            'num_workers': 4,
            'patch_size': (64, 64),
            'spatial_dims': 2,
            # 'target_name': 'earl', # si SingleTargetPETDataModule le requiert
        },
        
        # Modèle params
        'unlearning_model': {
            'num_classes': 5,          # Nombre de centres/scanners
            'spatial_dims': 2,         
            'input_shape': (64, 64),
            'unet_hid_chs': [32, 64, 128, 256],
            'unet_kernel_sizes': [3, 3, 3, 3],
            'unet_strides': [1, 2, 2, 2],
            'fft_sigma': 7.5,
            
            'lr_unet': 1e-4,
            'lr_clf': 1e-4,
            'weight_decay': 1e-5,
            
            'alpha_residual': 1.0,     # Facteur multiplicatif du résidu
            
            'lambda_l1': 1.0,          # Poids pour préserver l'image (petite magnitude)
            'lambda_ssim': 1.0,        # Poids pour préserver l'anatomie (SSIM)
            'lambda_adv': 1.0,         # Poids de la confusion (Adversarial)
            
            'suv_global_log_max': 6.0
        },
        
        # Trainer params
        'trainer': {
            'max_epochs': 100,
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices': 1,
            'log_every_n_steps': 1,
            # 'fast_dev_run': True, # Utile pour tester si le code compile
        }
    }
    # --------------------------------------------------------------
    
    set_seed(cfg.get('SEED', 42), workers=True)
    
    save_dir = "./runs/residual_unlearning/"
    os.makedirs(save_dir, exist_ok=True)
    
    datamodule = SingleTargetPETDataModule(**cfg.get('datamodule', {}))
    model = ResidualUnlearningSystem(**cfg.get('unlearning_model', {}))
    
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(save_dir, "checkpoints"),
            filename="epoch={epoch:03d}-loss_g={train_G/loss_total:.4f}",
            monitor="train_G/loss_total",
            mode="min",
            save_last=True,
            save_top_k=3,
            auto_insert_metric_name=False
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Intégration des paramètres recommandés similaires à l'Unlearning VAE
    trainer_kwargs = {
        'max_epochs': cfg['trainer'].get('max_epochs', 100),
        'accelerator': cfg['trainer'].get('accelerator', 'gpu' if torch.cuda.is_available() else 'cpu'),
        'devices': cfg['trainer'].get('devices', 1),
        'log_every_n_steps': 1,
        'check_val_every_n_epoch': 1,
        'num_sanity_val_steps': 0,
    }
    
    # On ajoute d'éventuels paramètres optionnels (limit_train_batches, precision...) s'ils existent dans config
    for k, v in cfg['trainer'].items():
        if k not in trainer_kwargs:
            trainer_kwargs[k] = v
            
    trainer = Trainer(
        default_root_dir=save_dir,
        callbacks=callbacks,
        **trainer_kwargs
    )
    
    logger.info("Lancement de l'entraînement Bac à Sable Residual Unlearning 🚀")
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()

