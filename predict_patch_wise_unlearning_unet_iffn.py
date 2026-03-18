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

from modules.data import MultiDomainUnlearningDataModule
from modules.models.unet import UnlearningUNet, UNet
from modules.models.domain_classifier import DomainClassifier
from modules.models.iffn import ImageFrequencyFusionModel
from modules.utils import set_seed


# =============================================================================
# Utilitaires patchs
# =============================================================================

def get_uniform_starts(dim_size: int, patch_size: int, min_overlap_ratio: float = 0.5) -> list[int]:
    """
    Distribue les indices de départ de façon parfaitement uniforme sur la dimension.
    Stride calculé depuis le ratio d'overlap, puis réparti sans arrondi cumulatif.
    """
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
    """
    Carte de poids gaussienne 3D (D, H, W), valeurs ∈ (0, 1].
    sigma_ratio=0.6 : doux, bords contribuent encore ~17% du centre.
    """
    maps = []
    for size in patch_size:
        coords = torch.linspace(-1, 1, size)
        g      = torch.exp(-0.5 * (coords / sigma_ratio) ** 2)
        maps.append(g / g.max())
    return maps[0][:, None, None] * maps[1][None, :, None] * maps[2][None, None, :]


# =============================================================================
# Traitement d'un sujet
# =============================================================================

def process_subject(model, batch, device, filename, curr_idx, length_loader):
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

    z_starts = get_uniform_starts(d_dim, z_patch_size, min_overlap_ratio=0.4)
    y_starts = get_uniform_starts(h_dim, y_patch_size, min_overlap_ratio=0.1)
    x_starts = get_uniform_starts(w_dim, x_patch_size, min_overlap_ratio=0.1)

    total_patches = len(z_starts) * len(y_starts) * len(x_starts)
    print(
        f"Volume: {d_dim}x{h_dim}x{w_dim} | "
        f"Patches: {len(z_starts)}x{len(y_starts)}x{len(x_starts)} = {total_patches}"
    )

    gauss_w = make_gaussian_weight_map((z_patch_size, y_patch_size, x_patch_size), sigma_ratio=2.5).to(device)

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

                    feature_map = model.feature_extractor(patch_norm)  # [B, C, H', W']
                    patch_pred = model.model.forward(feature_map, t=None)
                    
                    patch_pred = model._denormalize(patch_pred).squeeze(0)  # (D, H, W)

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

    # --- 4. Normalisation finale ---
    final_prediction = (output_volume / weight_sum.clamp(min=1e-8)).cpu()
    final_prediction = final_prediction.squeeze().permute(2, 1, 0).numpy()
    final_prediction = np.flip(final_prediction, axis=2)    # Flip Z
    final_prediction = np.flip(final_prediction, axis=1)    # Flip Y
    final_prediction = gaussian_filter(final_prediction, sigma=0.1)  # Lissage léger pour atténuer les artefacts de patchs

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

def predict_patch_wise_unet(args):
    config = OmegaConf.load(args.config_file)
    config = OmegaConf.to_container(config, resolve=True)
    set_seed(config.get('SEED', 42), workers=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    print(f"Loading model from: {args.ckpt_path}")
    unet = UNet(**config.get('unet', {}))
    feature_extractor = ImageFrequencyFusionModel(**config.get('feature_extractor', {}))
    domain_classifier = DomainClassifier(**config.get('domain_classifier', {}))

    model = UnlearningUNet.load_from_checkpoint(
        args.ckpt_path,
        model=unet,
        domain_classifier=domain_classifier,
        feature_extractor=feature_extractor,
        strict=False,
        **config.get('pipeline', {}),
    )
    
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    datamodule = MultiDomainUnlearningDataModule(**config.get('datamodule', {}))
    datamodule.prepare_data()
    datamodule.setup()

    loader = datamodule.test_dataloader()
    for idx, batch in enumerate(loader):
        process_subject(model, batch, device, args.filename, idx + 1, len(loader))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inférence patch-wise avec UNetWithIntermediateFeatures"
    )
    parser.add_argument('--config-file', '-c', type=str, required=True,
                        help='Chemin vers le fichier config YAML.')
    parser.add_argument('--ckpt-path',   '-m', type=str, required=True,
                        help='Chemin vers le checkpoint (.ckpt).')
    parser.add_argument('--filename',    '-f', type=str, required=False,
                        default='harmonized_PET_unet',
                        help='Nom du fichier de sortie (sans extension).')
    args = parser.parse_args()

    predict_patch_wise_unet(args)