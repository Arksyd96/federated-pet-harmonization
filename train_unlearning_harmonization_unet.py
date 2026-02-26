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

from modules.data import MultiDomainUnlearningDataModule
from modules.models.unet import UNetWithIntermediateFeatures, UnlearningUNet
from modules.models.domain_classifier import MultiLevelDomainClassifier
from modules.utils import set_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):
    config = OmegaConf.load(args.config_file)
    config = OmegaConf.to_container(config, resolve=True)

    config['DEBUG'] = args.debug
    if args.resume_checkpoint:
        config['ckpt_path'] = args.resume_checkpoint

    set_seed(config.get('SEED', 42), workers=True)

    # 2. Gestion des dossiers et logger
    save_dir = None
    wb_logger = None

    if not config.get('DEBUG'):
        current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        save_dir = os.path.join(os.path.curdir, config.get('dir_name'), str(current_time))
        os.makedirs(save_dir, exist_ok=True)

        OmegaConf.save(config, os.path.join(save_dir, "config.yaml"))

        wb_logger = WandbLogger(
            project=config.get('project_name'),
            name=config.get('name'),
            save_dir=save_dir,
            config=config
        )
    else:
        save_dir = "./runs/temporary/"
        logger.info("Mode DEBUG activé : Aucune sauvegarde sur disque.")

    # 3. DataModule
    datamodule = MultiDomainUnlearningDataModule(**config.get('datamodule', {}))

    # 4. Modèles
    unet = UNetWithIntermediateFeatures(**config.get('unet', {}))

    # feature_channels doit correspondre exactement aux canaux exposés par
    # UNetWithIntermediateFeatures : [hid_chs[0], hid_chs[1], ..., hid_chs[-1] (latent)]
    domain_classifier = MultiLevelDomainClassifier(**config.get('domain_classifier', {}))

    # 5. Pipeline Lightning
    if args.resume_checkpoint:
        logger.info(f"Reprise depuis : {args.resume_checkpoint}")
        pipeline = UnlearningUNet.load_from_checkpoint(
            args.resume_checkpoint,
            model=unet,
            domain_classifier=domain_classifier,
            strict=False,
        )
    else:
        pipeline = UnlearningUNet(
            model=unet,
            domain_classifier=domain_classifier,
            **config.get('pipeline', {}),
        )

    # 6. Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(save_dir, "./checkpoints"),
            filename="{epoch:02d}",
            **config.get('model_checkpoint', {})
        )
    ]

    if not config.get('DEBUG'):
        callbacks.append(LearningRateMonitor(logging_interval='step'))

    # 7. Trainer
    trainer = Trainer(
        logger=wb_logger if not config.get('DEBUG') else False,
        default_root_dir=save_dir,
        callbacks=callbacks,
        **config.get('trainer', {})
    )

    # 8. Lancement
    logger.info("Lancement de l'entraînement UnlearningUNet 🚀")
    trainer.fit(pipeline, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement harmonisation de domaine — UnlearningUNet")
    parser.add_argument("-c", "--config-file", type=str, required=True,
                        help="Chemin vers le fichier de configuration YAML")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Mode debug : pas de sauvegarde ni de logging WandB")
    parser.add_argument("-r", "--resume-checkpoint", type=str, default=None,
                        help="Chemin vers un checkpoint pour reprendre l'entraînement")

    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"Config introuvable : {args.config_file}")

    main(args)