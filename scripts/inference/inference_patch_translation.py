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

def process_subject(
    model, 
    batch, 
    device, 
    filename, 
    output_dir, 
    override, 
    include_only,
    curr_idx, 
    length_loader
    ):
    SUV_LOG_MAX = model.hparams.suv_global_log_max
    ALPHA = model.hparams.alpha
    
    subject_name = batch['subject_id'][0]
    print(f"Treating subject: {subject_name} ({curr_idx}/{length_loader})")
    
    if include_only is not None and subject_name not in include_only:
        print(f"⚠️  Subject {subject_name} not in include_only list. Skipping...")
        return
    
    # check if file already exists
    subj_out_dir = os.path.join(output_dir, subject_name)
    os.makedirs(subj_out_dir, exist_ok=True)
    if not override and os.path.exists(os.path.join(subj_out_dir, f"{filename}.nii.gz")):
        print(f"⚠️  Prediction already exists for {subject_name} at {os.path.join(subj_out_dir, f'{filename}.nii.gz')}. Skipping...")
        return

    # Récupération des données brutes (batch de taille 1)
    suv_source = batch['source'][tio.DATA].float().to(device)
    if suv_source.ndim == 5:
        suv_source = suv_source.squeeze(1) # (D, H, W)

    _, d_dim, h_dim, w_dim = suv_source.shape
    
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
    
    # pondération
    recon_volume = (output_volume / weight_sum.clamp(min=1e-8)).cpu()
    source_path = batch['source']['path'][0]

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
    
    datamodule = SingleTargetPETDataModule(**datamodule_kwargs)
    datamodule.prepare_data()
    datamodule.setup()

    loader = datamodule.test_dataloader()
    
    for idx, batch in enumerate(loader):
        process_subject(
            model=model, 
            batch=batch, 
            device=device, 
            filename=args.filename, 
            output_dir=args.output, 
            include_only=args.include_only,
            override=args.override,
            curr_idx=idx + 1, 
            length_loader=len(loader)
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch-wise Prediction Translation UNet (pseudo-EARL)")
    parser.add_argument('--config-file', '-c', type=str, required=True, help='Path to the config yaml file.')
    parser.add_argument('--ckpt-path', '-m', type=str, required=True, help='Path to the model checkpoint (.ckpt).')
    parser.add_argument('--output', '-o', type=str, required=True, help='ex: outputs/pseudoEARL.')
    parser.add_argument('--include-only', '-i', type=str, nargs='*', default=None, help='List of subject IDs to include (default: all).')
    parser.add_argument('--filename', '-f', type=str, required=False, default='pseudo-earl', help='Filename to process.')
    parser.add_argument('--override', '-r', action='store_true', help='Whether to override existing predictions.')
    args = parser.parse_args()
    
    predict_patch_wise_earl(args)