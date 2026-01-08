import torch
import torch.nn as nn
import torchio as tio
from modules.data import PETTranslationDataModule
from modules.diffusion import TranslationDiffusionPipeline
from modules.models.unet import UNet
from modules.scheduler import GaussianNoiseScheduler
from modules.utils import set_seed
from omegaconf import OmegaConf

# --- CONFIGURATION ---
config = OmegaConf.load('./configs/pet_earl_translation.yaml')
config = OmegaConf.to_container(config, resolve=True)
set_seed(config.get('SEED', 42), workers=True)

denoiser = UNet(cond_embedder=None, **config.get('denoiser', {}))
noise_scheduler = GaussianNoiseScheduler(**config.get('scheduler', {}))
diffuser = TranslationDiffusionPipeline.load_from_checkpoint(
    './runs/pet-earl-translation-diffusion/2025_12_08_105029/checkpoints/last.ckpt',
    noise_estimator=denoiser,
    noise_scheduler=noise_scheduler,
    strict=False
)

denoiser, noise_scheduler, diffuser = denoiser.to('cuda'), noise_scheduler.to('cuda'), diffuser.to('cuda')
diffuser.eval()

print('model loaded')
print('seed: {}'.format(config['SEED']))

datamodule = PETTranslationDataModule(
    root_dir='./data/PET-EARL/Rennes_Nifti_resampled/',
    batch_size=1,
    train_ratio=0.0, # gather same evaluation data as in training
    patch_size=(64, 64, 64),
    num_workers=32,
    queue_max_length=10, # whole volumes are returned so no need for large queue
    samples_per_volume=1
)

datamodule.prepare_data()
datamodule.setup()
batch = next(iter(datamodule.test_dataloader()))

print(batch['source'][tio.DATA].shape)
