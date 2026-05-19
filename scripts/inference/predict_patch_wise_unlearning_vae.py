import argparse
import math
import os
import numpy as np
import torch
import torchio as tio
import SimpleITK as sitk
from tqdm import tqdm
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter

from src.pet_harmonization.data import MultiDomainUnlearningDataModule, Float32Lambda
from src.pet_harmonization.models.harmonization_vae import DisentangledHarmonizationVAE, UnlearningVAE
from src.pet_harmonization.utils import set_seed


# =============================================================================
# Utilitaires patchs
# =============================================================================

def get_uniform_starts(dim_size: int, patch_size: int, min_overlap_ratio: float = 0.5) -> list[int]:
    if dim_size <= patch_size:
        return [0]
    max_stride = max(patch_size - int(patch_size * min_overlap_ratio), 1)
    n_patches  = math.ceil((dim_size - patch_size) / max_stride) + 1
    stride_f   = (dim_size - patch_size) / max(n_patches - 1, 1)
    return [round(i * stride_f) for i in range(n_patches)]


def make_gaussian_weight_map(
    patch_size: tuple[int, int, int],
    sigma_ratio: float = 0.6,
) -> torch.Tensor:
    maps = []
    for size in patch_size:
        coords = torch.linspace(-1, 1, size)
        g      = torch.exp(-0.5 * (coords / sigma_ratio) ** 2)
        maps.append(g / g.max())
    return maps[0][:, None, None] * maps[1][None, :, None] * maps[2][None, None, :]


# =============================================================================
# Extraction du z_style moyen depuis un volume de référence NIfTI
# =============================================================================

def extract_mean_z_style(
    ref_nifti_path: str,
    model,
    device: torch.device,
    y_patch_size: int = 64,
    x_patch_size: int = 64,
    z_patch_size: int = 5,
) -> torch.Tensor:
    """
    Charge le volume de référence exactement comme le test_dataloader
    (Float32Lambda + ToCanonical), puis extrait le z_style moyen patch par patch.

    Returns
    -------
    z_style_mean : (1, style_channels) — à passer à harmonize(z_style_fixed=...)
    """
    print(f"Extracting mean z_style from reference: {ref_nifti_path}")
    vae = model.vae

    # Même transform que test_dataloader — garantit la même orientation
    transform = tio.Compose([
        Float32Lambda(),
        tio.ToCanonical(),
    ])

    ref_subject = tio.Subject(
        source=tio.Image(ref_nifti_path, type=tio.INTENSITY),
    )
    ref_subject = transform(ref_subject)

    # (1, D, H, W) — même format que batch["source"][tio.DATA] en inférence
    ref_tensor = ref_subject["source"][tio.DATA].float().to(device)
    _, d_dim, h_dim, w_dim = ref_tensor.shape

    z_starts = get_uniform_starts(d_dim, z_patch_size, min_overlap_ratio=0.1)
    y_starts = get_uniform_starts(h_dim, y_patch_size, min_overlap_ratio=0.1)
    x_starts = get_uniform_starts(w_dim, x_patch_size, min_overlap_ratio=0.1)

    z_style_list = []

    with torch.no_grad():
        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    patch = ref_tensor[
                        :,
                        z:z + z_patch_size,
                        y:y + y_patch_size,
                        x:x + x_patch_size,
                    ]
                    if patch.mean() < 1e-3:
                        continue

                    patch_norm = model._normalize(patch)
                    _, _, mu_style, _ = vae.content_style_encoder(patch_norm)
                    z_style_list.append(mu_style.mean(dim=0, keepdim=True))  # (1, style_channels)

    if not z_style_list:
        raise RuntimeError("Aucun patch valide trouvé dans le volume de référence.")

    z_style_mean = torch.stack(z_style_list, dim=0).mean(dim=0)  # (1, style_channels)
    print(f"z_style extracted from {len(z_style_list)} patches — shape: {z_style_mean.shape}")
    return z_style_mean


# =============================================================================
# Traitement d'un sujet
# =============================================================================

def process_subject(
    model, 
    batch, 
    device, 
    filename, 
    curr_idx, 
    length_loader, 
    z_style_fixed=None,
    voi_filename: str = None
):
    vae = model.vae

    print(f"Treating subject: {batch['subject_name'][0]} ({curr_idx}/{length_loader})")

    # --- 1. Préparation du tenseur ---
    suv_source = batch["source"][tio.DATA].float().to(device)

    if suv_source.ndim == 5:
        suv_source = suv_source.squeeze(1)      # (B,1,D,H,W) → (B,D,H,W)

    _, d_dim, h_dim, w_dim = suv_source.shape
    orig_d, orig_h, orig_w = d_dim, h_dim, w_dim

    output_volume = torch.zeros((d_dim, h_dim, w_dim), device=device)
    weight_sum    = torch.zeros((d_dim, h_dim, w_dim), device=device)

    # --- 2. Configuration dynamique des patchs ---
    z_patch_size = 5
    y_patch_size = 64
    x_patch_size = 64

    # Pad symétrique si le volume est plus petit que la taille minimale du patch
    if d_dim < z_patch_size:
        pad_z = z_patch_size - d_dim
        suv_source = torch.nn.functional.pad(suv_source, (0, 0, 0, 0, pad_z // 2, pad_z - pad_z // 2))
        d_dim += pad_z
    if h_dim < y_patch_size:
        pad_y = y_patch_size - h_dim
        suv_source = torch.nn.functional.pad(suv_source, (0, 0, pad_y // 2, pad_y - pad_y // 2, 0, 0))
        h_dim += pad_y
    if w_dim < x_patch_size:
        pad_x = x_patch_size - w_dim
        suv_source = torch.nn.functional.pad(suv_source, (pad_x // 2, pad_x - pad_x // 2, 0, 0, 0, 0))
        w_dim += pad_x

    output_volume = torch.zeros((d_dim, h_dim, w_dim), device=device)
    weight_sum    = torch.zeros((d_dim, h_dim, w_dim), device=device)

    z_starts = get_uniform_starts(d_dim, z_patch_size, min_overlap_ratio=0.6)
    y_starts = get_uniform_starts(h_dim, y_patch_size, min_overlap_ratio=0.25)
    x_starts = get_uniform_starts(w_dim, x_patch_size, min_overlap_ratio=0.25)

    total_patches = len(z_starts) * len(y_starts) * len(x_starts)
    style_mode    = "référence" if z_style_fixed is not None else "neutre (zéros)"
    print(
        f"Volume: {d_dim}x{h_dim}x{w_dim} | "
        f"Patches: {len(z_starts)}x{len(y_starts)}x{len(x_starts)} = {total_patches} | "
        f"Style: {style_mode}"
    )

    gauss_w = make_gaussian_weight_map(
        (z_patch_size, y_patch_size, x_patch_size), sigma_ratio=1.0
    ).to(device)

    # --- 3. Boucle d'inférence ---
    pbar = tqdm(total=total_patches, desc="Inférence par patch")

    with torch.no_grad():
        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    patch_src = suv_source[
                        :,
                        z:z + z_patch_size,
                        y:y + y_patch_size,
                        x:x + x_patch_size,
                    ]

                    if patch_src.mean() < 1e-3:
                        pbar.update(1)
                        continue

                    patch_norm      = model._normalize(patch_src)
                    patch_pred_norm = vae.harmonize(
                        patch_norm,
                        x_style_ref=None,
                        z_style_fixed=z_style_fixed, # None → style neutre
                        style_dropout_p=0.95
                    )
                    # patch_pred_norm, *_ = vae.forward(patch_norm)
                    patch_pred = model._denormalize(patch_pred_norm).squeeze(0)  # (D, H, W)

                    output_volume[
                        z:z + z_patch_size,
                        y:y + y_patch_size,
                        x:x + x_patch_size,
                    ] += patch_pred * gauss_w
                    weight_sum[
                        z:z + z_patch_size,
                        y:y + y_patch_size,
                        x:x + x_patch_size,
                    ] += gauss_w

                    pbar.update(1)

    pbar.close()

    recon_volume = (output_volume / weight_sum.clamp(min=1e-8)).cpu()
    
    # 🎯 CORRECTION 2 : On crop le volume reconstruit pour virer le padding et retrouver la taille initiale de l'organe
    crop_z_start = (d_dim - orig_d) // 2
    crop_y_start = (h_dim - orig_h) // 2
    crop_x_start = (w_dim - orig_w) // 2
    
    recon_volume = recon_volume[
        crop_z_start : crop_z_start + orig_d,
        crop_y_start : crop_y_start + orig_h,
        crop_x_start : crop_x_start + orig_w
    ]

    source_path = batch['source']['path'][0]

    if voi_filename is not None:
        if z_style_fixed is None:
            np_arr = gaussian_filter(recon_volume.numpy(), sigma=1.0)
            recon_volume = torch.from_numpy(np_arr)

        affine_matrix = batch['source'][tio.AFFINE][0]

        # pred_tio fait maintenant exactement la taille attendue (orig_d, orig_h, orig_w)
        pred_tio = tio.ScalarImage(tensor=recon_volume.unsqueeze(0), affine=affine_matrix)
        output_sitk = pred_tio.as_sitk()

        output_path = os.path.join(os.path.dirname(source_path), f"{filename}_voi_only.nii.gz")
        sitk.WriteImage(output_sitk, output_path)
        print(f"Prediction (Cropped VOI) saved at: {output_path}")

        mask_tensor = batch['voi'][tio.DATA].cpu() 
        if mask_tensor.shape[1] == 1:
            mask_tensor = mask_tensor.squeeze(1)
        
        mask_tio = tio.LabelMap(tensor=mask_tensor, affine=affine_matrix)
        mask_sitk = mask_tio.as_sitk()

        mask_base_name = voi_filename.split('.')[0]
        mask_output_path = os.path.join(os.path.dirname(source_path), f"{mask_base_name}_cropped.nii.gz")
        sitk.WriteImage(mask_sitk, mask_output_path)
        print(f"Matching cropped mask saved at: {mask_output_path}")

    else:
        # CAS DU VOLUME ENTIER
        final_prediction = recon_volume.squeeze().permute(2, 1, 0).numpy()
        final_prediction = np.flip(final_prediction, axis=2)
        final_prediction = np.flip(final_prediction, axis=1)
        
        if z_style_fixed is None:
            final_prediction = gaussian_filter(final_prediction, sigma=1.0)

        output_sitk = sitk.GetImageFromArray(final_prediction)
        output_sitk.CopyInformation(sitk.ReadImage(source_path))

        output_path = os.path.join(os.path.dirname(source_path), f"{filename}.nii.gz")
        sitk.WriteImage(output_sitk, output_path)
        print(f"Whole-body prediction saved at: {output_path}")


# =============================================================================
# Entrée principale
# =============================================================================

def predict_patch_wise_vae(args):
    config = OmegaConf.load(args.config_file)
    config = OmegaConf.to_container(config, resolve=True)
    set_seed(config.get('SEED', 42), workers=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    print(f"Loading model from: {args.ckpt_path}")
    vae = DisentangledHarmonizationVAE(**config.get('vae', {}))
    model = UnlearningVAE.load_from_checkpoint(
        args.ckpt_path,
        vae=vae,
        **config.get('pipeline', {}),
    )
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # Extraction du z_style de référence si un volume est fourni
    z_style_fixed = None
    if args.style_ref:
        z_style_fixed = extract_mean_z_style(
            ref_nifti_path=args.style_ref,
            model=model,
            device=device,
        )

    datamodule = MultiDomainUnlearningDataModule(**config.get('datamodule', {}), voi_filename=args.voi_filename)
    datamodule.prepare_data()
    datamodule.setup()

    loader = datamodule.test_dataloader() if not args.voi_only else datamodule.voi_dataloader()
    for idx, batch in enumerate(loader):
        process_subject(
            model, batch, device, args.filename,
            idx + 1, len(loader),
            z_style_fixed=z_style_fixed,
            voi_filename=args.voi_filename if args.voi_only else None
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inférence patch-wise avec DisentangledHarmonizationVAE"
    )
    parser.add_argument('--config-file', '-c', type=str, required=True,
                        help='Chemin vers le fichier config YAML.')
    parser.add_argument('--ckpt-path',   '-m', type=str, required=True,
                        help='Chemin vers le checkpoint (.ckpt).')
    parser.add_argument('--style-ref',   '-s', type=str, required=False, default=None,
                        help='(Optionnel) Chemin vers un volume NIfTI de référence de style. '
                             'Si absent → style neutre (z_style = 0).')
    parser.add_argument('--voi-only', action='store_true', help='Si activé, ne traite que les patchs contenant la VOI (ex. cerveau).')
    parser.add_argument('--voi-filename', type=str, default=None, help='Chemin vers la VOI (masque binaire) à utiliser si --voi-only est activé.')
    parser.add_argument('--filename',    '-f', type=str, required=False,
                        default='harmonized_PET_vae',
                        help='Nom du fichier de sortie (sans extension).')
    args = parser.parse_args()

    predict_patch_wise_vae(args)
