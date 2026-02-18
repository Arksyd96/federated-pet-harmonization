import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchio as tio
from tqdm import tqdm
from omegaconf import OmegaConf

import SimpleITK as sitk
from modules.data import MultiDomainUnlearningDataModule
from modules.models.unet import UNet
from modules.models.iffn import ImageFrequencyFusionModel
from modules.models.domain_classifier import DomainClassifier
from modules.scheduler import GaussianNoiseScheduler
from modules.diffusion import UnlearningHarmonizationDiffusionPipeline
from modules.utils import set_seed

def get_start_indices(dim_size, patch_size, stride):
    """
    Génère les indices de départ pour les patchs sans dépasser les dimensions.
    Garantit que le dernier patch couvre bien la fin du volume.
    """
    indices = []
    i = 0
    while i + patch_size <= dim_size:
        indices.append(i)
        i += stride
    # Ajouter le dernier patch collé au bord si on n'est pas tombé pile poil
    if indices[-1] + patch_size < dim_size:
        indices.append(dim_size - patch_size)
    return sorted(list(set(indices))) # set pour éviter doublons


def process_subject(diffuser, batch, device, filename, check_if_exists, curr_idx, length_loader):
    print('Treating subject: {} ({}/{})'.format(batch['subject_name'][0], curr_idx, length_loader))
    
    # --- 1. Chargement des images ---
    suv_source = batch['source'][tio.DATA]
    suv_source = suv_source.to(device)
            
    if suv_source.ndim == 5:  # (B, 1, D, H, W) -> (B, D, H, W)
        suv_source = suv_source.squeeze(1)

    # --- 2. Préparation des Tenseurs ---
    b, d_dim, h_dim, w_dim = suv_source.shape
    output_volume = torch.zeros((d_dim, h_dim, w_dim), device=device)
    count_map = torch.zeros((d_dim, h_dim, w_dim), device=device)

    z_patch_size = 5
    y_patch_size = 64
    x_patch_size = 64
    overlap = 1  # recouvrement de 1 voxel

    z_starts = get_start_indices(d_dim, z_patch_size, z_patch_size - overlap)[40:42]
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
                    suv_patch_src = suv_source[:, z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size]
                    if suv_patch_src.mean() < 1e-3:
                            pbar.update(1)
                            continue
                    
                    # Normalisation scale to [-1, 1]
                    log_suv_patch_src = torch.log1p(suv_patch_src)
                    norm_log_patch_src = 2.0 * (log_suv_patch_src.clamp(0, diffuser.suv_global_log_max) / diffuser.suv_global_log_max) - 1.0
                    
                    # steps=50 (ou moins pour aller plus vite en test)
                    with torch.no_grad():
                        norm_patch_log_pred = diffuser.sample(
                            norm_log_patch_src,
                            steps=50,
                            use_ddim=True,
                            verbose=False
                        )

                    norm_patch_log_pred = norm_patch_log_pred.clamp(-1, 1)
                    log_suv_patch_pred = 0.5 * (norm_patch_log_pred + 1.0) * diffuser.suv_global_log_max
                    suv_patch_pred = torch.expm1(log_suv_patch_pred)  # Inverse de log1p pour revenir à 
                    
                    display_max = max(
                        5.0, 
                        suv_patch_pred.max().item()
                    )
                    
                    suv_patch_pred = suv_patch_pred.clamp(0, display_max)  # Clamp pour éviter valeurs extrêmes
                    
                    # C. Accumulation
                    output_volume[z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size] += suv_patch_pred.squeeze(0)
                    count_map[z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size] += 1.0
                    
                    pbar.update(1)

    pbar.close()

    # --- 5. Normalisation et Sauvegarde ---
    final_prediction = output_volume / count_map

    # Retour sur CPU pour sauvegarde
    final_prediction = final_prediction.cpu()
    final_prediction = final_prediction.squeeze().permute(2, 1, 0)  # Suppression des dimensions batch et channel
    
    final_prediction = final_prediction.numpy()
    final_prediction = np.flip(final_prediction, axis=2) # Flip Z
    final_prediction = np.flip(final_prediction, axis=1) # Flip Y (Correction orientation)

    output_sitk = sitk.GetImageFromArray(final_prediction)
    
    s_path = batch['source']['path'][0]
    s_meta = sitk.ReadImage(s_path)
    output_sitk.SetDirection(s_meta.GetDirection())
    output_sitk.SetOrigin(s_meta.GetOrigin())
    output_sitk.SetSpacing(s_meta.GetSpacing())

    # Ecriture disque
    output_dir = os.path.dirname(s_path)
    output_path = os.path.join(output_dir, filename)
    sitk.WriteImage(output_sitk, output_path)
    print(f'Prediction saved at: {output_path}')



def predict_patch_wise_earl(args):
    config = OmegaConf.load(args.config_file)
    config = OmegaConf.to_container(config, resolve=True)
    set_seed(config.get('SEED', 42), workers=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    print(f"Loading model from: {args.ckpt_path}")
    feature_extractor = ImageFrequencyFusionModel(**config.get('feature_extractor', {}))
    domain_classifier = DomainClassifier(**config.get('domain_classifier', {}))
    denoiser = UNet(cond_embedder=None, **config.get('denoiser', {}))
    noise_scheduler = GaussianNoiseScheduler(**config.get('scheduler', {}))
    diffuser = UnlearningHarmonizationDiffusionPipeline.load_from_checkpoint(
        args.ckpt_path,
        feature_extractor=feature_extractor,
        domain_classifier=domain_classifier,
        noise_estimator=denoiser,
        noise_scheduler=noise_scheduler,
        strict=False
    )
    
    for module in [feature_extractor, domain_classifier, denoiser, noise_scheduler, diffuser]:
        module.to(device)
        module.eval()
    
    print('Model loaded successfully.')
    
    # --- CHARGEMENT DU MODELE ---
    datamodule = MultiDomainUnlearningDataModule(**config.get('datamodule', {}))
    datamodule.prepare_data()
    datamodule.setup()

    loader = datamodule.test_dataloader()
    for idx, batch in enumerate(loader):
        process_subject(diffuser, batch, device, args.filename, args.check, idx + 1, len(loader))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch-wise Prediction with UNet Model")
    parser.add_argument('--config-file', '-c', type=str, required=True, help='Path to the config yaml file.')
    parser.add_argument('--ckpt-path', '-m', type=str, required=True, help='Path to the model checkpoint (.ckpt).')
    parser.add_argument('--check', action='store_true', help='Check if predictions already exist before processing.')
    parser.add_argument('--filename', '-f', type=str, required=True, default='harmonized_PET', help='Filename to process (if needed).')
    args = parser.parse_args()
    
    predict_patch_wise_earl(args)