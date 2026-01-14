import numpy as np
import torch
import torch.nn as nn
import torchio as tio
from tqdm import tqdm
import os
import SimpleITK as sitk
from modules.data import PETTranslationDataModule
from modules.models.unet import TranslationUNet
from modules.utils import set_seed
from omegaconf import OmegaConf
from modules.data import robust_patch_denormalization, robust_patch_normalization

# --- CONFIGURATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = OmegaConf.load('./configs/pet_earl_translation.yaml')
config = OmegaConf.to_container(config, resolve=True)
set_seed(config.get('SEED', 42), workers=True)

ckpt_path = './runs/2d-to-3d-pet-earl-translation-unet/2026_01_13_174241/checkpoints/epoch=25.ckpt'
model = TranslationUNet.load_from_checkpoint(ckpt_path)
model.to(device)
model.eval()

# Récupération des hyperparamètres stockés dans le modèle (plus sûr que la config yaml)
SUV_LOG_MAX = model.hparams.suv_global_log_max
ALPHA = model.hparams.alpha

print('model loaded')
print('seed: {}'.format(config['SEED']))

datamodule = PETTranslationDataModule(**config.get('datamodule', {}))
datamodule.prepare_data()
datamodule.setup()

loader = datamodule.test_dataloader()
batch = next(iter(loader))

print('Treating subject: {}'.format(batch['subject_id'][0]))

# --- 1. Chargement des Données ---
results = {}
suv_source, suv_target = batch['source'][tio.DATA], batch['target'][tio.DATA]
suv_source, suv_target = suv_source.float().squeeze(1), suv_target.float().squeeze(1)

# --- 2. Préparation des Tenseurs ---
suv_source, suv_target = suv_source.to(device), suv_target.to(device)
b, d_dim, h_dim, w_dim = suv_target.shape

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

z_patch_size = 5
y_patch_size = 64
x_patch_size = 64
overlap = 1  # recouvrement de 1 voxel

z_starts = get_start_indices(d_dim, z_patch_size, z_patch_size - overlap - 1)
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
                patch_src = suv_source[:, z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size]
                patch_tgt = suv_target[:, z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size]

                log_source = torch.log1p(patch_src)
                log_target = torch.log1p(patch_tgt)

                # scale to [-1, 1+eps] (i don't clip here, but could be an option)
                normalized_log_source = 2.0 * (log_source / SUV_LOG_MAX) - 1.0
                normalized_log_target = 2.0 * (log_target / SUV_LOG_MAX) - 1.0
            
                with torch.no_grad():
                    predicted_residual = model.forward(normalized_log_source)

                normalized_log_prediction = normalized_log_source + (predicted_residual / ALPHA)
                log_prediction = 0.5 * (normalized_log_prediction + 1.0) * SUV_LOG_MAX
                suv_prediction = torch.expm1(log_prediction)

                # Accumulation
                output_volume[z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size] += suv_prediction.squeeze(0)
                count_map[z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size] += 1.0
                
                pbar.update(1)

pbar.close()

# --- Normalisation et Sauvegarde ---
final_prediction = output_volume / count_map

# Retour sur CPU pour sauvegarde
final_prediction = final_prediction.cpu()
final_prediction = final_prediction.squeeze().permute(2, 1, 0)  # Suppression des dimensions batch et channel

final_prediction = final_prediction.numpy()
final_prediction = np.flip(final_prediction, axis=2)
final_prediction = np.flip(final_prediction, axis=1) # to match sitk orientation

output_sitk = sitk.GetImageFromArray(final_prediction)
s_meta = sitk.ReadImage(batch['source']['path'][0])
output_sitk.CopyInformation(s_meta)

output_path = os.path.join(os.path.dirname(batch['source']['path'][0]), f'predicted_EARL_unet.nii.gz')
sitk.WriteImage(output_sitk, output_path)

print(f'Prediction saved at: {output_path}')

