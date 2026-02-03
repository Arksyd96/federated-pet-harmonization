import argparse
import os
import numpy as np
import torch
import torchio as tio
import SimpleITK as sitk
from tqdm import tqdm
from omegaconf import OmegaConf

# Tes modules personnalisés
from modules.data import MultiTargetPETDataModule
from modules.models.unet import TranslationUNet
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


def process_subject(model, batch, device, filename, curr_idx, length_loader):
    # Récupération des paramètres du modèle
    SUV_LOG_MAX = model.hparams.suv_global_log_max
    ALPHA = model.hparams.alpha

    print('Treating subject: {} ({}/{})'.format(batch['subject_id'][0], curr_idx, length_loader))
    
    # code temporaire pour debug
    # check si les prédictions existent déjà pour ce patient :
    # source_path = batch['source']['path'][0]
    # output_dir = os.path.dirname(source_path)
    # for suffix in ['_EARL1.nii.gz', '_EARL2.nii.gz']:
    #     output_path = os.path.join(output_dir, f'{filename}{suffix}')
    #     if os.path.exists(output_path):
    #         print(f'Predictions already exist at: {output_path}. Skipping subject.')
    #         return

    # --- 1. Préparation des Tenseurs ---
    # Récupération des données brutes (batch de taille 1)
    suv_source = batch['source'][tio.DATA].float().squeeze(1) # (B, D, H, W)
    suv_source = suv_source.to(device)
    
    b, d_dim, h_dim, w_dim = suv_source.shape

    # Initialisation des volumes de sortie
    output_earl_1_volume = torch.zeros((d_dim, h_dim, w_dim), device=device)
    output_earl_2_volume = torch.zeros((d_dim, h_dim, w_dim), device=device)
    count_map = torch.zeros((d_dim, h_dim, w_dim), device=device)

    # --- 2. Configuration des Patchs ---
    z_patch_size = 5
    y_patch_size = 64
    x_patch_size = 64
    overlap = 2  # recouvrement de 1 voxel sur x et y
    
    z_starts = get_start_indices(d_dim, z_patch_size, z_patch_size - 1)
    y_starts = get_start_indices(h_dim, y_patch_size, y_patch_size - overlap)
    x_starts = get_start_indices(w_dim, x_patch_size, x_patch_size - overlap)

    total_patches = len(z_starts) * len(y_starts) * len(x_starts)
    print(f"Volume: {d_dim}x{h_dim}x{w_dim} | Patchs à traiter : {total_patches}")

    # --- 3. Boucle d'Inférence ---
    pbar = tqdm(total=total_patches, desc="Inférence par Patch")

    with torch.no_grad():
        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    # Extraction
                    patch_src = suv_source[:, z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size]
                    if patch_src.mean() < 1e-3:
                        # Patch vide, on skip
                        pbar.update(1)
                        continue

                    # Normalisation & Log Transform
                    log_source = torch.log1p(patch_src)

                    normalized_log_source = 2.0 * (log_source / SUV_LOG_MAX) - 1.0
                    
                    # Prédiction
                    predicted_residual = model.forward(normalized_log_source)
                    pr_earl1, pr_earl2 = torch.chunk(predicted_residual, 2, dim=1)

                    # Reconstruction inverse
                    normalized_log_pred_earl1 = normalized_log_source + (pr_earl1 / ALPHA)
                    normalized_log_pred_earl2 = normalized_log_source + (pr_earl2 / ALPHA)

                    log_pred_earl1 = 0.5 * (normalized_log_pred_earl1 + 1.0) * SUV_LOG_MAX
                    log_pred_earl2 = 0.5 * (normalized_log_pred_earl2 + 1.0) * SUV_LOG_MAX
                    
                    suv_pred_earl1 = torch.expm1(log_pred_earl1)
                    suv_pred_earl2 = torch.expm1(log_pred_earl2)

                    # Accumulation
                    output_earl_1_volume[z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size] += suv_pred_earl1.squeeze(0)
                    output_earl_2_volume[z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size] += suv_pred_earl2.squeeze(0)
                    count_map[z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size] += 1.0
                    
                    pbar.update(1)

    pbar.close()

    # --- 4. Normalisation finale et Sauvegarde ---
    final_pred_earl1 = output_earl_1_volume / count_map
    final_pred_earl2 = output_earl_2_volume / count_map

    # Post-processing pour format Nifti
    final_pred_earl1 = final_pred_earl1.cpu()
    final_pred_earl2 = final_pred_earl2.cpu()
    
    final_pred_earl1 = final_pred_earl1.squeeze().permute(2, 1, 0)  # (W, H, D) -> Permute pour match ITK
    final_pred_earl2 = final_pred_earl2.squeeze().permute(2, 1, 0)  # (W, H, D) -> Permute pour match ITK

    final_pred_earl1 = final_pred_earl1.numpy()
    final_pred_earl1 = np.flip(final_pred_earl1, axis=2) # Flip Z
    final_pred_earl1 = np.flip(final_pred_earl1, axis=1) # Flip Y (Correction orientation)
    
    final_pred_earl2 = final_pred_earl2.numpy()
    final_pred_earl2 = np.flip(final_pred_earl2, axis=2) # Flip Z
    final_pred_earl2 = np.flip(final_pred_earl2, axis=1) # Flip Y (Correction orientation)

    # Création image SimpleITK
    output_sitk_earl_1 = sitk.GetImageFromArray(final_pred_earl1)
    output_sitk_earl_2 = sitk.GetImageFromArray(final_pred_earl2)
    
    # Copie métadonnées source
    source_path = batch['source']['path'][0]
    s_meta = sitk.ReadImage(source_path)
    output_sitk_earl_1.CopyInformation(s_meta)
    output_sitk_earl_2.CopyInformation(s_meta)

    # Ecriture disque
    output_dir = os.path.dirname(source_path)
    output_filename = f'{filename}'
    
    for output_sitk, suffix in zip([output_sitk_earl_1, output_sitk_earl_2], ['_EARL1.nii.gz', '_EARL2.nii.gz']):
        output_path = os.path.join(output_dir, f'{output_filename}{suffix}')
        sitk.WriteImage(output_sitk, output_path)
        print(f'Prediction saved at: {output_path}')


def predict_patch_wise_earl(args):
    config = OmegaConf.load(args.config_file)
    config = OmegaConf.to_container(config, resolve=True)
    set_seed(config.get('SEED', 42), workers=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    print(f"Loading model from: {args.ckpt_path}")
    model = TranslationUNet.load_from_checkpoint(args.ckpt_path)
    model.to(device)
    model.eval()
    print('Model loaded successfully.')

    datamodule = MultiTargetPETDataModule(**config.get('datamodule', {}))
    datamodule.prepare_data()
    datamodule.setup()

    loader = datamodule.test_dataloader()
    for idx, batch in enumerate(loader):
        process_subject(model, batch, device, args.filename, idx + 1, len(loader))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch-wise Prediction with UNet Model")
    parser.add_argument('--config-file', '-c', type=str, required=True, help='Path to the config yaml file.')
    parser.add_argument('--ckpt-path', '-m', type=str, required=True, help='Path to the model checkpoint (.ckpt).')
    parser.add_argument('--filename', '-f', type=str, required=False, default='predicted_EARL_unet', help='Filename to process (if needed).')
    args = parser.parse_args()
    
    predict_patch_wise_earl(args)
