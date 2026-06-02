import argparse
import os
import math
import numpy as np
import torch
import torchio as tio
import SimpleITK as sitk
from tqdm import tqdm
from omegaconf import OmegaConf

# Tes modules personnalisés
from pet_harmonization.data import SingleTargetPETDataModule
from pet_harmonization.models.unet import TranslationUNet
from pet_harmonization.utils import set_seed

def get_start_indices(dim_size, patch_size, stride):
    indices = []
    i = 0
    while i + patch_size <= dim_size:
        indices.append(i)
        i += stride
    # Ajouter le dernier patch collé au bord si on n'est pas tombé pile poil
    if indices[-1] + patch_size < dim_size:
        indices.append(dim_size - patch_size)
    return sorted(list(set(indices))) # set pour éviter doublons

def get_uniform_starts(dim_size: int, patch_size: int, min_overlap_ratio: float = 0.5) -> list[int]:
    if dim_size <= patch_size:
        return [0]
    max_stride = max(patch_size - int(patch_size * min_overlap_ratio), 1)
    n_patches  = math.ceil((dim_size - patch_size) / max_stride) + 1
    stride_f   = (dim_size - patch_size) / max(n_patches - 1, 1)
    return [round(i * stride_f) for i in range(n_patches)]

def make_gaussian_weight_map(patch_size: tuple[int, int, int], sigma_ratio: float = 0.6) -> torch.Tensor:
    maps = []
    for size in patch_size:
        coords = torch.linspace(-1, 1, size)
        g      = torch.exp(-0.5 * (coords / sigma_ratio) ** 2)
        maps.append(g / g.max())
    return maps[0][:, None, None] * maps[1][None, :, None] * maps[2][None, None, :]

def process_subject(model, batch, device, filename, output_dir, curr_idx, length_loader, voi_filename=None):
    SUV_LOG_MAX = model.hparams.suv_global_log_max
    ALPHA = model.hparams.alpha
    
    subject_name = batch['subject_id'][0]
    print(f"Treating subject: {subject_name} ({curr_idx}/{length_loader})")
    
    # assert that voi is not empty if --voi-only is activated
    if voi_filename is not None:
        voi_tensor = batch['voi'][tio.DATA]
        if voi_tensor.sum() < 1e-3:
            print(f"⚠️  Skipping {subject_name} due to empty VOI mask.")
            return
    
    # dir
    region_name = voi_filename.split('.')[0] if voi_filename is not None else "whole_body"
    subj_out_dir = os.path.join(output_dir, subject_name, region_name)
    os.makedirs(subj_out_dir, exist_ok=True)

    # --- 1. Préparation des Tenseurs ---
    # Récupération des données brutes (batch de taille 1)
    suv_source = batch['source'][tio.DATA].float().to(device)
    if suv_source.ndim == 5:
        suv_source = suv_source.squeeze(1) # (D, H, W)

    _, d_dim, h_dim, w_dim = suv_source.shape
    orig_d, orig_h, orig_w = d_dim, h_dim, w_dim
    
    # padding dynamique pour assurer les dimensions du patch d'entrée
    z_patch_size, y_patch_size, x_patch_size = 5, 64, 64
    overlap = 2

    # Initialisation des volumes de sortie
    output_volume   = torch.zeros((d_dim, h_dim, w_dim), device=device)
    weight_sum      = torch.zeros((d_dim, h_dim, w_dim), device=device)
    
    z_starts = get_start_indices(d_dim, z_patch_size, z_patch_size - 1)
    y_starts = get_start_indices(h_dim, y_patch_size, y_patch_size - overlap)
    x_starts = get_start_indices(w_dim, x_patch_size, x_patch_size - overlap)
    
    total_patches = len(z_starts) * len(y_starts) * len(x_starts)
    gauss_w = make_gaussian_weight_map((z_patch_size, y_patch_size, x_patch_size), sigma_ratio=.5).to(device)

    # --- 3. Boucle d'Inférence ---
    pbar = tqdm(total=total_patches, desc="Inférence du pseudo-EARL")

    with torch.no_grad():
        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    patch_src = suv_source[:, z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size]
                    
                    if patch_src.mean() < 1e-3:
                        pbar.update( 1 )
                        continue

                    # Normalisation & Log Transform
                    log_source = torch.log1p(patch_src)
                    normalized_log_source = 2.0 * (log_source / SUV_LOG_MAX) - 1.0
                    
                    # Prédiction
                    predicted_residual = model.forward(normalized_log_source)

                    # Reconstruction inverse
                    normalized_log_prediction = normalized_log_source + (predicted_residual / ALPHA)
                    log_prediction = 0.5 * (normalized_log_prediction + 1.0) * SUV_LOG_MAX
                    suv_prediction = torch.expm1(log_prediction)

                    # Accumulation
                    output_volume[z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size] += suv_prediction.squeeze(0) * gauss_w
                    weight_sum[z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size] += gauss_w
                    
                    pbar.update(1)

    pbar.close()
    
    # Dé-padding et pondération
    recon_volume = (output_volume / weight_sum.clamp(min=1e-8)).cpu()
    crop_z_start = (d_dim - orig_d) // 2
    crop_y_start = (h_dim - orig_h) // 2
    crop_x_start = (w_dim - orig_w) // 2
    
    recon_volume = recon_volume[
        crop_z_start : crop_z_start + orig_d,
        crop_y_start : crop_y_start + orig_h,
        crop_x_start : crop_x_start + orig_w
   ]

    # Sauvegarde BIDS like
    source_path = batch['source']['path'][0]
    affine_matrix = batch['source'][tio.AFFINE][0]
    
    if voi_filename is not None:        
        # Sauvegarde source croppée
        source_cropped_path = os.path.join(subj_out_dir, "pet.nii.gz")
        if not os.path.exists(source_cropped_path):
            source_tensor = batch['source'][tio.DATA][0].cpu()
            source_tio = tio.ScalarImage(tensor=source_tensor, affine=affine_matrix)
            sitk.WriteImage(source_tio.as_sitk(), source_cropped_path)
            
        # Sauvegarde targets croppée (EARL1 et/ou 2)
        c_idx = 1
        for key in batch.keys():
            if key.startswith('target'):
                target_cropped_path = os.path.join(subj_out_dir, f"earl{c_idx}.nii.gz")
                if not os.path.exists(target_cropped_path):
                    target_tensor = batch[key][tio.DATA][0].cpu()
                    target_tio = tio.ScalarImage(tensor=target_tensor, affine=affine_matrix)
                    sitk.WriteImage(target_tio.as_sitk(), target_cropped_path)
                c_idx += 1
                   
        # Sauvegarde mask croppé
        mask_cropped_path = os.path.join(subj_out_dir, "mask.nii.gz")
        if not os.path.exists(mask_cropped_path):
            mask_tensor = batch['voi'][tio.DATA].cpu()
            if mask_tensor.shape[1] == 1:
                mask_tensor = mask_tensor.squeeze( 1 )
            mask_tio = tio.LabelMap(tensor=mask_tensor, affine=affine_matrix)
            sitk.WriteImage(mask_tio.as_sitk(), mask_cropped_path)
            
        # Sauvegarde prédiction pseudo-EARL
        if recon_volume.ndim == 3:
            recon_volume = recon_volume.unsqueeze(0)
            
        num_channels = recon_volume.shape[0]
        for c in range(num_channels):
            pred_path = os.path.join(subj_out_dir, f"{filename}{c + 1}.nii.gz")
            pred_tio = tio.ScalarImage(tensor=recon_volume[c].unsqueeze( 0 ), affine=affine_matrix)
            sitk.WriteImage(pred_tio.as_sitk(), pred_path)
        
    else:
        final_prediction = recon_volume.squeeze().permute(2, 1, 0).numpy()
        final_prediction = np.flip(final_prediction, axis=2) # Flip Z
        final_prediction = np.flip(final_prediction, axis=1) # Flip Y (Correction orientation)
        final_prediction = final_prediction.astype(np.float32) # ensure float32 for SimpleITK

        # Création image SimpleITK
        output_sitk = sitk.GetImageFromArray(final_prediction)
        output_sitk.CopyInformation(sitk.ReadImage(source_path))
        
        pred_path = os.path.join(subj_out_dir, f"{filename}.nii.gz")
        sitk.WriteImage(output_sitk, pred_path)
        print(f"✅ Whole-body prediction saved at: {pred_path}")

        
def predict_patch_wise_earl(args):
    config = OmegaConf.load(args.config_file)
    config = OmegaConf.to_container(config, resolve=True)
    set_seed(config.get('SEED', 42), workers=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initialisation | Modèle : TranslationUNET | Device : {device}")
    
    model = TranslationUNet.load_from_checkpoint(args.ckpt_path)
    model.to(device)
    model.eval()
    print('Model loaded successfully.')

    # Passage du voi_filename dans les kwargs du DataModule
    datamodule_kwargs = config.get('datamodule', {})
    datamodule_kwargs['voi_filename'] = args.voi_filename
    
    datamodule = SingleTargetPETDataModule(**datamodule_kwargs)
    datamodule.prepare_data()
    datamodule.setup()

    loader = datamodule.voi_dataloader(min_voi_crop_shape=(32, 192, 192)) if args.voi_only else datamodule.test_dataloader()
    
    for idx, batch in enumerate(loader):
        process_subject(
            model=model, 
            batch=batch, 
            device=device, 
            filename=args.filename, 
            output_dir=args.output, 
            curr_idx=idx + 1, 
            length_loader=len(loader),
            voi_filename=args.voi_filename if args.voi_only else None
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch-wise Prediction Translation UNet (pseudo-EARL)")
    parser.add_argument('--config-file', '-c', type=str, required=True, help='Path to the config yaml file.')
    parser.add_argument('--ckpt-path', '-m', type=str, required=True, help='Path to the model checkpoint (.ckpt).')
    parser.add_argument('--output', '-o', type=str, required=True, help='ex: outputs/pseudoEARL.')
    parser.add_argument('--filename', '-f', type=str, required=False, default='pseudo-earl', help='Filename to process.')
    parser.add_argument('--voi-only', action='store_true', help='Activer le recadrage sur l\'organe ciblé.')
    parser.add_argument('--voi-filename', type=str, default=None, help='Nom du masque (ex: liver.nii.gz) si --voi-only est activé.')
    args = parser.parse_args()
    
    predict_patch_wise_earl(args)