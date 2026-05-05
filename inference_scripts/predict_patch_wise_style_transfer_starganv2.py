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

from modules.data import MultiDomainUnlearningDataModule, Float32Lambda
from modules.models.starganv2 import StarGANv2, StyleEncoder, StarGANv2Discriminator, StarGANv2Generator, StyleEmbedder
from modules.utils import set_seed


# =============================================================================
# Utilitaires patchs (identiques aux autres scripts d'inférence)
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
# Extraction du style moyen depuis un volume de référence NIfTI
# =============================================================================

def extract_mean_style(
    ref_nifti_path: str,
    model: StarGANv2,
    device: torch.device,
    z_patch_size: int = 5,
    y_patch_size: int = 64,
    x_patch_size: int = 64,
) -> torch.Tensor:
    """
    Charge le volume de référence (Float32Lambda + ToCanonical),
    extrait le style_code sur chaque patch valide et retourne la moyenne.

    Returns
    -------
    style_mean : (1, style_dim) — à passer à harmonize(z_style_fixed=...)
    """
    print(f"Extracting mean style from reference: {ref_nifti_path}")

    transform = tio.Compose([Float32Lambda(), tio.ToCanonical()])
    ref_subject = tio.Subject(source=tio.Image(ref_nifti_path, type=tio.INTENSITY))
    ref_subject = transform(ref_subject)

    ref_tensor = ref_subject["source"][tio.DATA].float().to(device)  # (1, D, H, W)
    _, d_dim, h_dim, w_dim = ref_tensor.shape

    # Overlap minimal pour l'extraction de style — on veut juste diversité spatiale
    z_starts = get_uniform_starts(d_dim, z_patch_size, min_overlap_ratio=0.1)
    y_starts = get_uniform_starts(h_dim, y_patch_size, min_overlap_ratio=0.1)
    x_starts = get_uniform_starts(w_dim, x_patch_size, min_overlap_ratio=0.1)

    style_list = []

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

                    patch_norm   = model._normalize(patch)
                    style_code   = model.style_encoder(patch_norm)   # (1, style_dim)
                    style_list.append(style_code)

    if not style_list:
        raise RuntimeError("Aucun patch valide trouvé dans le volume de référence.")

    style_mean = torch.stack(style_list, dim=0).mean(dim=0)  # (1, style_dim)
    print(f"Style extracted from {len(style_list)} patches — shape: {style_mean.shape}")
    return style_mean


# =============================================================================
# Traitement d'un sujet
# =============================================================================

def process_subject(
    model: StarGANv2,
    batch: dict,
    device: torch.device,
    filename: str,
    curr_idx: int,
    length_loader: int,
    z_style_fixed: torch.Tensor,
):
    print(f"Treating subject: {batch['subject_name'][0]} ({curr_idx}/{length_loader})")

    # --- 1. Préparation du tenseur ---
    suv_source = batch["source"][tio.DATA].float().to(device)
    if suv_source.ndim == 5:
        suv_source = suv_source.squeeze(1)      # (B,1,D,H,W) → (B,D,H,W)

    b, d_dim, h_dim, w_dim = suv_source.shape

    output_volume = torch.zeros((d_dim, h_dim, w_dim), device=device)
    weight_sum    = torch.zeros((d_dim, h_dim, w_dim), device=device)

    # --- 2. Configuration dynamique des patchs ---
    z_patch_size = 5
    y_patch_size = 64
    x_patch_size = 64

    z_starts = get_uniform_starts(d_dim, z_patch_size, min_overlap_ratio=0.6)
    y_starts = get_uniform_starts(h_dim, y_patch_size, min_overlap_ratio=0.25)
    x_starts = get_uniform_starts(w_dim, x_patch_size, min_overlap_ratio=0.25)

    total_patches = len(z_starts) * len(y_starts) * len(x_starts)
    print(
        f"Volume: {d_dim}x{h_dim}x{w_dim} | "
        f"Patches: {len(z_starts)}x{len(y_starts)}x{len(x_starts)} = {total_patches}"
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

                    patch_norm = model._normalize(patch_src)

                    # Inférence via harmonize — z_style_fixed précalculé
                    patch_pred_norm = model.harmonize(
                        patch_norm,
                        z_style_fixed=z_style_fixed,
                    )

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

    # --- 4. Normalisation finale et post-processing ---
    final_prediction = (output_volume / weight_sum.clamp(min=1e-8)).cpu()
    final_prediction = final_prediction.squeeze().permute(2, 1, 0).numpy()
    final_prediction = np.flip(final_prediction, axis=2)    # Flip Z
    final_prediction = np.flip(final_prediction, axis=1)    # Flip Y
    # final_prediction = gaussian_filter(final_prediction, sigma=1.0)

    # --- 5. Sauvegarde ---
    output_sitk = sitk.GetImageFromArray(final_prediction)
    source_path = batch['source']['path'][0]
    output_sitk.CopyInformation(sitk.ReadImage(source_path))

    output_path = os.path.join(os.path.dirname(source_path), f"{filename}.nii.gz")
    sitk.WriteImage(output_sitk, output_path)
    print(f"Prediction saved at: {output_path}")


# =============================================================================
# Entrée principale
# =============================================================================

def predict_patch_wise_stargan(args):
    config = OmegaConf.load(args.config_file)
    config = OmegaConf.to_container(config, resolve=True)
    set_seed(config.get('SEED', 42), workers=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    pipeline_cfg = config.get("pipeline", {})

    # ── Instanciation des composants ─────────────────────────────────────────
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

    # ── Extraction du style de référence ─────────────────────────────────────
    z_style_fixed = extract_mean_style(
        ref_nifti_path=args.style_ref,
        model=model,
        device=device,
    )

    # ── DataModule ────────────────────────────────────────────────────────────
    datamodule = MultiDomainUnlearningDataModule(**config.get('datamodule', {}))
    datamodule.prepare_data()
    datamodule.setup()

    loader = datamodule.test_dataloader()
    for idx, batch in enumerate(loader):
        process_subject(
            model, batch, device, args.filename,
            idx + 1, len(loader),
            z_style_fixed=z_style_fixed,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inférence patch-wise avec StarGANv2 — harmonisation PET"
    )
    parser.add_argument('--config-file', '-c', type=str, required=True, help='Chemin vers le fichier config YAML.')
    parser.add_argument('--ckpt-path', '-m', type=str, required=True, help='Chemin vers le checkpoint (.ckpt).')
    parser.add_argument('--style-ref', '-s', type=str, required=True, help='Chemin vers le volume NIfTI de réf')
    parser.add_argument('--filename', '-f', type=str, required=False, default='harmonized_PET_stargan')
    args = parser.parse_args()

    if not os.path.exists(args.style_ref):
        raise FileNotFoundError(f"Volume de référence introuvable : {args.style_ref}")

    predict_patch_wise_stargan(args)