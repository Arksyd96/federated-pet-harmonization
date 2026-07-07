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


def save_prediction(recon_volume: torch.Tensor, source_path: str, pred_path: str):
    """Fonction modulaire pour formater et sauvegarder le volume SimpleITK."""
    final_prediction = recon_volume.squeeze().permute(2, 1, 0).numpy()
    final_prediction = np.flip(final_prediction, axis=2) # Flip Z
    final_prediction = np.flip(final_prediction, axis=1) # Flip Y (Correction orientation)
    final_prediction = final_prediction.astype(np.float32) # ensure float32 for SimpleITK

    output_sitk = sitk.GetImageFromArray(final_prediction)
    output_sitk.CopyInformation(sitk.ReadImage(source_path))
    sitk.WriteImage(output_sitk, pred_path)
    print(f"✅ Prediction saved at: {pred_path}")


def process_subject(
    model, 
    batch, 
    device, 
    filename, 
    output_dir, 
    override, 
    include_only,
    curr_idx, 
    length_loader,
    num_standards
    ):
    SUV_LOG_MAX = model.hparams.suv_global_log_max
    ALPHA = model.hparams.alpha
    
    subject_name = batch['subject_id'][0]
    print(f"Treating subject: {subject_name} ({curr_idx}/{length_loader})")
    
    if include_only is not None and subject_name not in include_only:
        print(f"⚠️  Subject {subject_name} not in include_only list. Skipping...")
        return
    
    # Check if files already exist
    subj_out_dir = os.path.join(output_dir, subject_name)
    os.makedirs(subj_out_dir, exist_ok=True)
    
    # Détermination dynamique des noms de fichiers
    out_filenames = [f"{filename}.nii.gz"] if num_standards == 1 else [f"{filename}{i + 1}.nii.gz" for i in range(num_standards)]
    all_exist = all(os.path.exists(os.path.join(subj_out_dir, f)) for f in out_filenames)
    
    if not override and all_exist:
        print(f"⚠️  Predictions already exist for {subject_name}. Skipping...")
        return

    # Récupération des données brutes (batch de taille 1)
    suv_source = batch['source'][tio.DATA].float().to(device)
    if suv_source.ndim == 5:
        suv_source = suv_source.squeeze(1) # (D, H, W)

    _, d_dim, h_dim, w_dim = suv_source.shape
    
    # padding dynamique pour assurer les dimensions du patch d'entrée
    z_patch_size, y_patch_size, x_patch_size = 5, 64, 64
    overlap = 2

    # Initialisation des volumes de sortie (Une liste contenant 1 ou 2 volumes)
    output_volumes  = [torch.zeros((d_dim, h_dim, w_dim), device=device) for _ in range(num_standards)]
    
    # Un seul weight_sum suffit car les patchs sont accumulés aux mêmes endroits
    weight_sum      = torch.zeros((d_dim, h_dim, w_dim), device=device)
    
    z_starts = get_start_indices(d_dim, z_patch_size, z_patch_size - 1)
    y_starts = get_start_indices(h_dim, y_patch_size, y_patch_size - overlap)
    x_starts = get_start_indices(w_dim, x_patch_size, x_patch_size - overlap)
    
    total_patches = len(z_starts) * len(y_starts) * len(x_starts)
    gauss_w = make_gaussian_weight_map((z_patch_size, y_patch_size, x_patch_size), sigma_ratio=.5).to(device)

    # --- 3. Boucle d'Inférence ---
    pbar = tqdm(total=total_patches, desc="Inférence patch-wise")

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
                    
                    # Prédiction du/des résidus (peut être 5 canaux ou 10 canaux)
                    predicted_residual = model.forward(normalized_log_source)

                    # Duplication de la source pour l'addition si on a plusieurs standards (ex: (1, 10, 64, 64))
                    src_repeated = normalized_log_source.repeat(1, num_standards, 1, 1)

                    # Reconstruction inverse
                    normalized_log_prediction = src_repeated + (predicted_residual / ALPHA)
                    log_prediction = 0.5 * (normalized_log_prediction + 1.0) * SUV_LOG_MAX
                    suv_prediction = torch.expm1(log_prediction)

                    # Split des canaux prédits selon les standards (morceaux de taille 5)
                    chunks = torch.chunk(suv_prediction, num_standards, dim=1)

                    # Accumulation séparée pour chaque standard
                    for i, chunk in enumerate(chunks):
                        output_volumes[i][z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size] += chunk.squeeze(0) # * gauss_w
                    
                    weight_sum[z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size] += 1
                    pbar.update(1)

    pbar.close()
    
    # Pondération et sauvegarde
    source_path = batch['source']['path'][0]
    weight_clamped = weight_sum.clamp(min=1e-8)
    
    for i, out_filename in enumerate(out_filenames):
        recon_volume = (output_volumes[i] / weight_clamped).cpu()
        pred_path = os.path.join(subj_out_dir, out_filename)
        save_prediction(recon_volume, source_path, pred_path)

        
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

    print(f"⚙️  Configuration : Génération de {args.num_standards} standard(s) cible(s).")

    # On utilise toujours le SingleTargetPETDataModule pour l'inférence
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
            length_loader=len(loader),
            num_standards=args.num_standards
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch-wise Prediction Translation UNet (pseudo-EARL)")
    parser.add_argument('--config-file', '-c', type=str, required=True, help='Path to the config yaml file.')
    parser.add_argument('--ckpt-path', '-m', type=str, required=True, help='Path to the model checkpoint (.ckpt).')
    parser.add_argument('--output', '-o', type=str, required=True, help='ex: outputs/pseudoEARL.')
    parser.add_argument('--include-only', '-i', type=str, nargs='*', default=None, help='List of subject IDs to include (default: all).')
    parser.add_argument('--filename', '-f', type=str, required=False, default='pseudo-earl', help='Filename to process.')
    parser.add_argument('--override', '-r', action='store_true', help='Whether to override existing predictions.')
    parser.add_argument('--num-standards', '-n', type=int, default=1, help='Number of target standards to generate (ex: 1 or 2).')
    args = parser.parse_args()
    
    predict_patch_wise_earl(args)
