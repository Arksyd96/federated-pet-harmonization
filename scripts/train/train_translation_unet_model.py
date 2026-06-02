import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import argparse
import os
import logging
from datetime import datetime
from omegaconf import OmegaConf

import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer

from pet_harmonization.data import SingleTargetPETDataModule, MultiTargetPETDataModule
from pet_harmonization.models.unet import TranslationUNet
from pet_harmonization.utils import set_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    # 1. Configuration
    config = OmegaConf.load(args.config_file)
    config = OmegaConf.to_container(config, resolve=True)

    config['DEBUG'] = args.debug
    if args.resume_checkpoint:
        config['ckpt_path'] = args.resume_checkpoint

    set_seed(config.get('SEED', 42), workers=True)

    # 2. Gestion des Dossiers et Logger
    save_dir = None
    wb_logger = None

    if not config.get('DEBUG'):
        current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        save_dir = os.path.join(os.path.curdir, config.get('dir_name'), str(current_time))
        os.makedirs(save_dir, exist_ok=True)
        
        # Sauvegarde de la config
        OmegaConf.save(config, os.path.join(save_dir, "config.yaml"))
        
        wb_logger = WandbLogger(
            project=config.get('project_name'),
            name=config.get('name'),
            save_dir=save_dir,
            config=config,
            offline=True
        )
    else:
        save_dir = "./runs/temporary/"
        logger.info("Mode DEBUG activé : Aucune sauvegarde sur disque.")

    # 3. DataModule & Modèles
    # datamodule = MultiTargetPETDataModule(**config.get('datamodule', {}))
    datamodule = SingleTargetPETDataModule(**config.get('datamodule', {}))

    model = None
    if args.resume_checkpoint:
        logger.info(f"Reprise depuis : {args.resume_checkpoint}")
        model = TranslationUNet.load_from_checkpoint(
            args.resume_checkpoint,
            strict=False
        )
    else:
        model = TranslationUNet(**config.get('translation_model', {}))

    # 4. Callbacks
    callbacks = [
        # RichProgressBar(), # TODO: Activer si besoin (bug)
        ModelCheckpoint(
            dirpath=os.path.join(save_dir, "./checkpoints"),
            filename="{epoch:02d}",
            **config.get('model_checkpoint', {})
        )
    ]

    if not config.get('DEBUG'):
        callbacks.append(LearningRateMonitor(logging_interval='step'))

    # 5. Trainer
    trainer = Trainer(
        logger=wb_logger if not config.get('DEBUG') else False,
        default_root_dir=save_dir, # Sera None en debug (utilise /tmp ou courant sans écrire)
        callbacks=callbacks,
        **config.get('trainer', {})
    )

    # 6. Lancement
    logger.info("Lancement de l'entraînement 🚀")
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", type=str, required=True)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-r", "--resume-checkpoint", type=str, default=None)
    
    args = parser.parse_args()
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"Config introuvable : {args.config_file}")

    main(args)