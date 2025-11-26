import argparse
import importlib
import os
from datetime import datetime
from omegaconf import OmegaConf

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import wandb as wandb_logger
from pytorch_lightning.trainer import Trainer

from modules.data import PETToEARLDataLoader
from modules.diffusion import TranslationDiffusionPipeline
from modules.models.unet import UNet
from modules.scheduler import GaussianNoiseScheduler
from modules.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", type=str, default=None, required=True)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-r", "--resume-checkpoint", type=str, default=None)
    args = parser.parse_args()

    try:
        config = OmegaConf.load(args.config_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {args.config_file}")

    config['DEBUG'] = args.debug
    set_seed(config['SEED'], deterministic=False)

    os.environ["WANDB_API_KEY"] = config['WANDB_API_KEY']

    if not config['DEBUG']:
        # --------------- Settings --------------------
        current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        config['save_dir'] = os.path.join(os.path.curdir, config['dir_name'], str(current_time))
        os.makedirs(config['save_dir'], exist_ok=True)

        # --------------- Logger --------------------
        logger = wandb_logger.WandbLogger(
            project="cancer-outcome-prediction",
            name=config['name'],
            save_dir=config['save_dir'],
        )

    # datamodule
    datamodule = BurdenkoSignedDistanceDataLoader(**config['datamodule'], dtype=torch.float32)

    # --------------- Denoiser --------------------
    denoiser = UNet(
        cond_embedder=BurdenkoClinicalEmbedder(embed_dim=config['denoiser']['embed_dim'], 
                                               max_period=config['denoiser']['max_cond_period']),
        **config['denoiser']
    )
    
    # --------------- Diffusion Pipeline --------------------
    noise_scheduler = GaussianNoiseScheduler(**config['scheduler'])

    if args.resume_checkpoint is None:
        diffuser = SequenceDiffusionPipeline(
            noise_scheduler = noise_scheduler,
            noise_estimator = denoiser,
            **config['diffuser']
        )
    else:
        # Load the config from the checkpoint
        config['diffuser']['resuming_from'] = args.resume_checkpoint
        diffuser = SequenceDiffusionPipeline.load_from_checkpoint(
            args.resume_checkpoint,
            noise_scheduler = noise_scheduler,
            noise_estimator = denoiser,
            **config['diffuser']
        )

    # -------------- Training Initialization ---------------
    config['model_checkpoint']["dirpath"] = config['save_dir'] if not config['DEBUG'] else "./runs/temporary/"
    checkpointing = ModelCheckpoint(**config['model_checkpoint'])

    # --------------- Trainer --------------------
    config['trainer']['default_root_dir'] = config['save_dir'] if not config['DEBUG'] else "./"
    trainer = Trainer(
        logger = logger if not config['DEBUG'] else False,
        **config['trainer'],
        callbacks=[checkpointing]
    )

    if not config['DEBUG']:
        # --------------- Save Config --------------------
        OmegaConf.save(config, f"{config['save_dir']}/config.yaml")
    
    # Modify trainer call
    trainer.fit(diffuser, datamodule=datamodule)

