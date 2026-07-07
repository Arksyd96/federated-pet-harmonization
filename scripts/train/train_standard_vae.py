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

from pet_harmonization.data import MultiDomainUnlearningDataModule
from pet_harmonization.models.harmonization_vae import DisentangledHarmonizationVAE, StandardHarmonizationVAE
from pet_harmonization.utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")


def main(args):
    config = OmegaConf.load(args.config_file)
    config = OmegaConf.to_container(config, resolve=True)
    config["DEBUG"] = args.debug

    if args.resume_checkpoint:
        config["ckpt_path"] = args.resume_checkpoint

    set_seed(config.get("SEED", 42), workers=True)

    save_dir  = None
    wb_logger = None

    if not config.get("DEBUG"):
        current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        save_dir = os.path.join(os.path.curdir, config.get("dir_name"), current_time)
        os.makedirs(save_dir, exist_ok=True)

        # Sauvegarde de la config complète dans le dossier run
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

    datamodule = MultiDomainUnlearningDataModule(**config.get("datamodule", {}))
    vae = DisentangledHarmonizationVAE(**config.get("vae", {}))

    if args.resume_checkpoint:
        logger.info(f"Reprise depuis : {args.resume_checkpoint}")
        pipeline = StandardHarmonizationVAE.load_from_checkpoint(
            args.resume_checkpoint,
            vae=vae,
            strict=False,
            **config.get("pipeline", {}),
        )
    else:
        pipeline = StandardHarmonizationVAE(
            vae=vae,
            **config.get("pipeline", {}),
        )

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(save_dir, "./checkpoints"),
            **config.get('model_checkpoint', {})
        )
    ]

    if not config.get("DEBUG"):
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer = Trainer(
        logger=wb_logger if not config.get('DEBUG') else False,
        default_root_dir=save_dir,
        callbacks=callbacks,
        **config.get('trainer', {})
    )

    logger.info("Lancement de l'entraînement StandardHarmonizationVAE (Ablation) 🚀")
    logger.info(f"  Objectif : MAE + KLD (Sans désapprentissage adversarial)")
    logger.info(f"  Époques prévues : {config['trainer'].get('max_epochs')}")

    trainer.fit(pipeline, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement harmonisation PET — Étude d'ablation (Standard VAE)")
    parser.add_argument("-c", "--config-file", type=str, required=True, help="Chemin vers le fichier de configuration YAML",)
    parser.add_argument("-d", "--debug", action="store_true", help="Mode debug : pas de sauvegarde ni de logging WandB",)
    parser.add_argument("-r", "--resume-checkpoint", type=str, default=None, help="Chemin vers un checkpoint pour reprendre l'entraînement",)

    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"Config introuvable : {args.config_file}")

    main(args)