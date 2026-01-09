import torch
import torch.nn as nn
import torchio as tio
from tqdm import tqdm
import os
import SimpleITK as sitk
from modules.data import PETTranslationDataModule
from modules.diffusion import TranslationDiffusionPipeline
from modules.models.unet import UNet
from modules.scheduler import GaussianNoiseScheduler
from modules.utils import set_seed
from omegaconf import OmegaConf

from modules.data import robust_patch_normalization, robust_patch_denormalization

# --- CONFIGURATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = OmegaConf.load('./configs/pet_earl_translation.yaml')
config = OmegaConf.to_container(config, resolve=True)
set_seed(config.get('SEED', 42), workers=True)

denoiser = UNet(cond_embedder=None, **config.get('denoiser', {}))
noise_scheduler = GaussianNoiseScheduler(**config.get('scheduler', {}))
diffuser = TranslationDiffusionPipeline.load_from_checkpoint(
    './runs/2d-to-3d-pet-earl-translation-diffusion/2026_01_08_131448/checkpoints/last.ckpt',
    noise_estimator=denoiser,
    noise_scheduler=noise_scheduler,
    strict=False
)

denoiser, noise_scheduler, diffuser = denoiser.to(device), noise_scheduler.to(device), diffuser.to(device)
diffuser.eval()

print('model loaded')
print('seed: {}'.format(config['SEED']))

datamodule = PETTranslationDataModule(**config.get('datamodule', {}))

datamodule.prepare_data()
datamodule.setup()
batch = next(iter(datamodule.test_dataloader()))

print('Treating subject: {}'.format(batch['subject_id'][0]))


# --- 1. Chargement des Données ---
results = {}
source, target = batch['source'][tio.DATA], batch['target'][tio.DATA]
source, target = source.squeeze(1), target.squeeze(1)  # Remove extra dim introduced by torchIO

# --- 2. Préparation des Tenseurs ---
source, target = source.to(device), target.to(device)
b, d_dim, h_dim, w_dim = source.shape
# batch is always 1 in inference

output_volume = torch.zeros((d_dim, h_dim, w_dim), device=device)
count_map = torch.zeros((d_dim, h_dim, w_dim), device=device)

# Fonction pour générer les indices de départ sans dépasser
def get_start_indices(dim_size, patch_size, stride):
    indices = []
    i = 0
    while i + patch_size <= dim_size:
        indices.append(i)
        i += stride
    # Ajouter le dernier patch collé au bord si on n'est pas tombé pile poil
    if indices[-1] + patch_size < dim_size:
        indices.append(dim_size - patch_size)
    return sorted(list(set(indices))) # set pour éviter doublons si ça tombe pile

z_patch_size = 3
y_patch_size = 64
x_patch_size = 64
overlap = 1  # recouvrement de 1 voxel

z_starts = get_start_indices(d_dim, z_patch_size, z_patch_size - overlap)
y_starts = get_start_indices(h_dim, y_patch_size, y_patch_size - overlap)
x_starts = get_start_indices(w_dim, x_patch_size, x_patch_size - overlap)

total_patches = len(z_starts) * len(y_starts) * len(x_starts)
print(f"Volume: {d_dim}x{h_dim}x{w_dim} | Patchs à traiter : {total_patches}")

# --- 4. Boucle d'Inférence ---
pbar = tqdm(total=total_patches, desc="Inférence par Patch")

with torch.no_grad():
    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
                
                # A. Extraction du Patch Source
                patch_src = source[:, z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size]
                patch_tgt = target[:, z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size]
                
                # B. Prédiction (Appel à ta pipeline de diffusion)
                norm_patch_src, norm_patch_tgt, norm_factors = robust_patch_normalization(patch_src, patch_tgt, percentiles=(0.0, 99.9), clone=True)
                target_delta = norm_patch_tgt - norm_patch_src # => [-2, 2]
                
                # steps=50 (ou moins pour aller plus vite en test)
                with torch.no_grad():
                    delta = diffuser.sample(
                        norm_patch_src,
                        condition=None,
                        steps=10,
                        use_ddim=True,
                        verbose=False
                    )

                norm_patch_pred = norm_patch_src + delta
                patch_pred_suv, _ = robust_patch_denormalization(norm_patch_pred, norm_patch_pred, norm_factors)

                # print(norm_patch_pred.min().item(), norm_patch_pred.max().item(), norm_patch_pred.mean().item())
                # print(patch_pred_suv.min().item(), patch_pred_suv.max().item(), patch_pred_suv.mean().item())
                # print(norm_factors)
                
                # C. Accumulation
                output_volume[z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size] += patch_pred_suv.squeeze(0)
                count_map[z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size] += 1.0
                
                pbar.update(1)

pbar.close()

# --- 5. Normalisation et Sauvegarde ---
final_prediction = output_volume / count_map

# Retour sur CPU pour sauvegarde
final_prediction = final_prediction.cpu()
final_prediction = final_prediction.squeeze().permute(2, 1, 0)  # Suppression des dimensions batch et channel

output_sitk = sitk.GetImageFromArray(final_prediction)
s_meta = sitk.ReadImage(batch['source']['path'][0])

orient_filter = sitk.DICOMOrientImageFilter()
orient_filter.SetDesiredCoordinateOrientation("LPS")
output_sitk = orient_filter.Execute(s_meta)

output_sitk.SetDirection(s_meta.GetDirection())
output_sitk.SetOrigin(s_meta.GetOrigin())
output_sitk.SetSpacing(s_meta.GetSpacing())

output_path = os.path.join(os.path.dirname(batch['source']['path'][0]), f'predicted_EARL.nii.gz')
sitk.WriteImage(output_sitk, output_path)

print(f'Prediction saved at: {output_path}')

