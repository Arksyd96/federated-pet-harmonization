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
from src.pet_harmonization.models.starganv2 import (
    StarGANv2, StyleEncoder, StarGANv2Discriminator,
    StarGANv2Generator, StyleEmbedder,
)
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
# Chargement du style pré-extrait
# =============================================================================

def load_style(style_path: str, device: torch.device) -> torch.Tensor:
    """
    Charge le fichier .pt produit par extract_style_stargan.py.

    Returns
    -------
    style : (1, style_dim) sur device
    """
    checkpoint = torch.load(style_path, map_location=device)
    style      = checkpoint["style"].to(device)   # (1, style_dim)

    print(f"Style loaded from : {style_path}")
    print(f"  mode       : {checkpoint.get('mode', 'N/A')}")
    print(f"  n_patients : {checkpoint.get('n_patients', 'N/A')}")
    print(f"  style_dim  : {checkpoint.get('style_dim', style.shape[-1])}")
    print(f"  repo_ref   : {checkpoint.get('repo_ref', 'N/A')}")

    return style


# =============================================================================
# Traitement d'un sujet
# =============================================================================

def process_subject(
    model,
    batch: dict,
    device: torch.device,
    filename: str,
    curr_idx: int,
    length_loader: int,
    z_style_fixed: torch.Tensor,
    voi_filename: str = None
):
    print(f"Treating subject: {batch['subject_name']} ({curr_idx}/{length_loader})")

    # --- 1. Préparation du tenseur ---
    suv_source = batch["source"][tio.DATA].float().to(device)
        
    if suv_source.ndim == 5:
        suv_source = suv_source.squeeze(1)

    b, d_dim, h_dim, w_dim = suv_source.shape
    
    # 🎯 CRUCIAL : On sauvegarde les dimensions d'origine de l'organe
    orig_d, orig_h, orig_w = d_dim, h_dim, w_dim

    # --- 2. Configuration dynamique des patchs & PADDING ---
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

    # 🎯 CORRECTION 1 : On initialise les volumes vides APRÈS avoir mis à jour d_dim, h_dim, w_dim
    output_volume = torch.zeros((d_dim, h_dim, w_dim), device=device)
    weight_sum    = torch.zeros((d_dim, h_dim, w_dim), device=device)

    z_starts = get_uniform_starts(d_dim, z_patch_size, min_overlap_ratio=0.6)
    y_starts = get_uniform_starts(h_dim, y_patch_size, min_overlap_ratio=0.25)
    x_starts = get_uniform_starts(w_dim, x_patch_size, min_overlap_ratio=0.25)

    total_patches = len(z_starts) * len(y_starts) * len(x_starts)
    style_mode    = "pré-extrait" if z_style_fixed is not None else "neutre (zéros)"
    print(
        f"Volume (Padded): {d_dim}x{h_dim}x{w_dim} (Original: {orig_d}x{orig_h}x{orig_w}) | "
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
                    patch_pred_norm = model.harmonize(
                        patch_norm,
                        z_style_fixed=z_style_fixed,
                    )
                    patch_pred = model._denormalize(patch_pred_norm).squeeze(0)

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

    # --- 4. Post-processing et sauvegarde ---
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

def predict_patch_wise_stargan(args):
    config = OmegaConf.load(args.config_file)
    config = OmegaConf.to_container(config, resolve=True)
    set_seed(config.get('SEED', 42), workers=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    pipeline_cfg  = config.get("pipeline", {})

    # ── Chargement du modèle ──────────────────────────────────────────────────
    print(f"Loading model from: {args.ckpt_path}")
    style_encoder  = StyleEncoder(**config.get("style_encoder", {}))
    style_embedder = StyleEmbedder(
        style_channels=pipeline_cfg["style_dim"],
        style_embedding_dim=pipeline_cfg["style_embedding_dim"],
    )
    generator     = StarGANv2Generator(**config.get("generator", {}))
    discriminator = StarGANv2Discriminator(
        num_domains=pipeline_cfg["num_domains"],
        **config.get("discriminator", {}),
    )

    model = StarGANv2.load_from_checkpoint(
        args.ckpt_path,
        generator=generator,
        style_encoder=style_encoder,
        style_embedder=style_embedder,
        discriminator=discriminator,
        strict=False,
        **pipeline_cfg,
    )
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # ── Chargement du style ───────────────────────────────────────────────────
    z_style_fixed = None
    if args.style_ref:
        z_style_fixed = load_style(args.style_ref, device)
    else:
        print("⚠️  Aucun style fourni — utilisation du vecteur nul.")
        z_style_fixed = torch.zeros(1, pipeline_cfg["style_dim"], device=device)

    # ── DataModule ────────────────────────────────────────────────────────────
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
        description="Inférence patch-wise StarGAN v2 — harmonisation PET"
    )
    parser.add_argument('--config-file', '-c', type=str, required=True, help='Chemin vers le fichier config YAML.')
    parser.add_argument('--ckpt-path', '-m', type=str, required=True, help='Chemin vers le checkpoint (.ckpt).')
    parser.add_argument('--style-ref', '-s', type=str, required=False, 
                        default=None, help='Fichier .pt de style pré-extrait. Si absent → vecteur nul.')
    parser.add_argument('--voi-only', action='store_true', help='Si activé, ne traite que les patchs contenant la VOI (ex. cerveau).')
    parser.add_argument('--voi-filename', type=str, default=None, help='Chemin vers la VOI (masque binaire) à utiliser si --voi-only est activé.')
    parser.add_argument('--filename', '-f', type=str, required=False, default='harmonized_PET_stargan',
                        help='Nom du fichier de sortie (sans extension).')
    args = parser.parse_args()

    predict_patch_wise_stargan(args)