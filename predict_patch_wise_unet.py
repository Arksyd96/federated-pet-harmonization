import argparse
import os
import numpy as np
import torch
import torchio as tio
import SimpleITK as sitk
from tqdm import tqdm
from omegaconf import OmegaConf

# Tes modules personnalisés
from modules.data import PETTranslationDataModule
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


def process_subject(model, batch, device, filename):
    # Récupération des paramètres du modèle
    SUV_LOG_MAX = model.hparams.suv_global_log_max
    ALPHA = model.hparams.alpha

    print('Treating subject: {}'.format(batch['subject_id'][0]))

    # --- 1. Préparation des Tenseurs ---
    # Récupération des données brutes (batch de taille 1)
    suv_source = batch['source'][tio.DATA].float().squeeze(1) # (B, D, H, W)
    suv_target = batch['target'][tio.DATA].float().squeeze(1)

    suv_source = suv_source.to(device)
    suv_target = suv_target.to(device)
    
    b, d_dim, h_dim, w_dim = suv_target.shape

    # Initialisation des volumes de sortie
    output_volume = torch.zeros((d_dim, h_dim, w_dim), device=device)
    count_map = torch.zeros((d_dim, h_dim, w_dim), device=device)

    # --- 2. Configuration des Patchs ---
    z_patch_size = 5
    y_patch_size = 64
    x_patch_size = 64
    overlap = 1  # recouvrement de 1 voxel sur x et y
    

    z_starts = get_start_indices(d_dim, z_patch_size, 2)
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
                    patch_tgt = suv_target[:, z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size]

                    # Normalisation & Log Transform
                    log_source = torch.log1p(patch_src)
                    log_target = torch.log1p(patch_tgt) # (Calculé mais non utilisé pour l'inférence, hérité du code original)

                    normalized_log_source = 2.0 * (log_source / SUV_LOG_MAX) - 1.0
                    
                    # Prédiction
                    predicted_residual = model.forward(normalized_log_source)

                    # Reconstruction inverse
                    normalized_log_prediction = normalized_log_source + (predicted_residual / ALPHA)
                    log_prediction = 0.5 * (normalized_log_prediction + 1.0) * SUV_LOG_MAX
                    suv_prediction = torch.expm1(log_prediction)

                    # Accumulation
                    output_volume[z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size] += suv_prediction.squeeze(0)
                    count_map[z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size] += 1.0
                    
                    pbar.update(1)

    pbar.close()

    # --- 4. Normalisation finale et Sauvegarde ---
    final_prediction = output_volume / count_map

    # Post-processing pour format Nifti
    final_prediction = final_prediction.cpu()
    final_prediction = final_prediction.squeeze().permute(2, 1, 0)  # (W, H, D) -> Permute pour match ITK

    final_prediction = final_prediction.numpy()
    final_prediction = np.flip(final_prediction, axis=2) # Flip Z
    final_prediction = np.flip(final_prediction, axis=1) # Flip Y (Correction orientation)

    # Création image SimpleITK
    output_sitk = sitk.GetImageFromArray(final_prediction)
    
    # Copie métadonnées source
    source_path = batch['source']['path'][0]
    s_meta = sitk.ReadImage(source_path)
    output_sitk.CopyInformation(s_meta)

    # Ecriture disque
    output_dir = os.path.dirname(source_path)
    output_filename = f'{filename}.nii.gz'
    output_path = os.path.join(output_dir, output_filename)
    
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

    datamodule = PETTranslationDataModule(**config.get('datamodule', {}))
    datamodule.prepare_data()
    datamodule.setup()

    loader = datamodule.test_dataloader()
    for batch in loader:
        process_subject(model, batch, device, args.filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch-wise Prediction with UNet Model")
    parser.add_argument('--config-file', '-c', type=str, required=True, help='Path to the config yaml file.')
    parser.add_argument('--ckpt-path', '-m', type=str, required=True, help='Path to the model checkpoint (.ckpt).')
    parser.add_argument('--filename', '-f', type=str, required=False, default='predicted_EARL_unet', help='Filename to process (if needed).')
    args = parser.parse_args()
    
    predict_patch_wise_earl(args)