import os
import argparse
from datetime import datetime
from omegaconf import OmegaConf

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import wandb as wandb_logger
from pytorch_lightning.trainer import Trainer

from modules.data import FederatedPETDataLoader
from modules.models.autoencoders import VariationalAutoencoder
from modules.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", type=str, default=None, required=True)
    parser.add_argument("-d", "--debug", action="store_true")
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
            project=config['project_name'],
            name=config['name'],
            save_dir=config['save_dir'],
        )

    # datamodule
    datamodule = FederatedPETDataLoader(**config['datamodule'], dtype=torch.float32)

    # Initialize model
    model = VariationalAutoencoder(**config['model'])

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
    trainer.fit(model, datamodule=datamodule)  # Changed from 'diffuser' to 'model'
