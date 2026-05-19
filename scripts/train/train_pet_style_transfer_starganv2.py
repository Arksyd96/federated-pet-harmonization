import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import argparse
import logging
import os
from datetime import datetime

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.pet_harmonization.data import MultiDomainUnlearningDataModule
from src.pet_harmonization.models.unet import UNet
from src.pet_harmonization.models.starganv2 import (
    StarGANv2, StyleEncoder, StarGANv2Discriminator, StarGANv2Generator, StyleEmbedder
)
from src.pet_harmonization.utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")


def main(args):
    # ── Config ────────────────────────────────────────────────────────────────
    config = OmegaConf.load(args.config_file)
    config = OmegaConf.to_container(config, resolve=True)
    config["DEBUG"] = args.debug

    if args.resume_checkpoint:
        config["ckpt_path"] = args.resume_checkpoint

    set_seed(config.get("SEED", 42), workers=True)

    # ── Dossiers et logger WandB ──────────────────────────────────────────────
    save_dir  = None
    wb_logger = None

    if not config.get("DEBUG"):
        current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        save_dir = os.path.join(os.path.curdir, config.get("dir_name"), current_time)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(save_dir, "config.yaml"))

        wb_logger = WandbLogger(
            project=config.get("project_name"),
            name=config.get("name"),
            save_dir=save_dir,
            config=config,
        )
    else:
        save_dir = "./runs/temporary/"
        logger.info("Mode DEBUG activé : pas de sauvegarde ni de logging WandB.")

    # ── DataModule ────────────────────────────────────────────────────────────
    datamodule = MultiDomainUnlearningDataModule(**config.get("datamodule", {}))

    # ── Composants du modèle ──────────────────────────────────────────────────
    pipeline_cfg       = config.get("pipeline", {})
    
    style_encoder = StyleEncoder(**config.get("style_encoder", {}))
    style_embedder = StyleEmbedder(
        style_channels=pipeline_cfg["style_dim"],
        style_embedding_dim=pipeline_cfg["style_embedding_dim"],
    )

    # Generator et Discriminator
    generator = StarGANv2Generator(**config.get("generator", {}))
    discriminator = StarGANv2Discriminator(
        num_domains=pipeline_cfg["num_domains"],
        **config.get("discriminator", {}),
    )

    # ── Pipeline Lightning ────────────────────────────────────────────────────
    if args.resume_checkpoint:
        logger.info(f"Reprise depuis : {args.resume_checkpoint}")
        pipeline = StarGANv2.load_from_checkpoint(
            args.resume_checkpoint,
            generator=generator,
            style_encoder=style_encoder,
            style_embedder=style_embedder,
            discriminator=discriminator,
            strict=False,
            **pipeline_cfg,
        )
    else:
        pipeline = StarGANv2(
            generator=generator,
            style_encoder=style_encoder,
            style_embedder=style_embedder,
            discriminator=discriminator,
            **pipeline_cfg,
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(save_dir, "checkpoints"),
            **config.get("model_checkpoint", {}),
        )
    ]

    if not config.get("DEBUG"):
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        logger=wb_logger if not config.get("DEBUG") else False,
        default_root_dir=save_dir,
        callbacks=callbacks,
        **config.get("trainer", {}),
    )

    # ── Lancement ─────────────────────────────────────────────────────────────
    logger.info("Lancement de l'entraînement StarGANv2 PET 🚀")
    logger.info(f"  Stage 1 (warmup)     : {pipeline_cfg['warmup_epochs']} epochs")
    logger.info(f"  Stage 2 (adversarial): {config['trainer']['max_epochs'] - pipeline_cfg['warmup_epochs']} epochs")
    logger.info(f"  Style dim            : {pipeline_cfg['style_dim']}")
    logger.info(f"  Style embedding dim  : {pipeline_cfg['style_embedding_dim']}")
    logger.info(f"  Num domains          : {pipeline_cfg['num_domains']}")

    trainer.fit(
        pipeline,
        datamodule=datamodule,
        ckpt_path=config.get("ckpt_path", None),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entraînement StarGAN v2 — Harmonisation PET multi-sites"
    )
    parser.add_argument(
        "-c", "--config-file", type=str, required=True,
        help="Chemin vers le fichier de configuration YAML",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true",
        help="Mode debug : pas de sauvegarde ni de logging WandB",
    )
    parser.add_argument(
        "-r", "--resume-checkpoint", type=str, default=None,
        help="Chemin vers un checkpoint pour reprendre l'entraînement",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"Config introuvable : {args.config_file}")

    main(args)