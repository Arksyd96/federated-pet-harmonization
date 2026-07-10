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

# Import des Datamodules
from pet_harmonization.data import MultiDomainUnlearningDataModule

# Import de TOUS les modèles
from pet_harmonization.models.starganv2 import StarGANv2, StyleEncoder, StarGANv2Discriminator, StarGANv2Generator, StyleEmbedder
from pet_harmonization.models.harmonization_vae import DisentangledHarmonizationVAE, UnlearningVAE, StandardHarmonizationVAE
from pet_harmonization.models.unet_v2_skip import UnlearningUNet as UnlearningUNetSkip, SpectralUNetWithIntermediateFeatures
from pet_harmonization.models.unet import UnlearningUNet as UnlearningUNetIFFN, UNet
from pet_harmonization.models.domain_classifier import DomainClassifier
from pet_harmonization.models.iffn import ImageFrequencyFusionModel

from pet_harmonization.utils import set_seed

# =============================================================================

def get_uniform_starts(dim_size: int, patch_size: int, min_overlap_ratio: float = 0.5) -> list[int]:
    if dim_size <= patch_size:
        return [ 0 ]
    max_stride = max(patch_size - int(patch_size * min_overlap_ratio), 1)
    n_patches  = math.ceil((dim_size - patch_size) / max_stride) + 1
    stride_f   = (dim_size - patch_size) / max(n_patches - 1, 1)
    return [ round(i * stride_f) for i in range(n_patches) ]

def make_gaussian_weight_map(patch_size: tuple[int, int, int], sigma_ratio: float = 0.6) -> torch.Tensor:
    maps = []
    for size in patch_size:
        coords = torch.linspace(-1, 1, size)
        g      = torch.exp(-0.5 * (coords / sigma_ratio) ** 2)
        maps.append(g / g.max())
    return maps[ 0 ][ :, None, None ] * maps[ 1 ][ None, :, None ] * maps[ 2 ][ None, None, : ]

def load_style(style_path: str, device: torch.device) -> torch.Tensor:
    checkpoint = torch.load(style_path, map_location=device)
    style      = checkpoint["style"].to(device)
    
    print(f"Style loaded from : {style_path}")
    print(f"  mode       : {checkpoint.get('mode', 'N/A')}")
    print(f"  n_patients : {checkpoint.get('n_patients', 'N/A')}")
    print(f"  style_dim  : {checkpoint.get('style_dim', style.shape[-1])}")
    print(f"  repo_ref   : {checkpoint.get('repo_ref', 'N/A')}")
    
    return style

# =============================================================================

def infer_patch(model_type: str, model, patch_src: torch.Tensor, style: torch.Tensor = None):
    patch_norm = model._normalize(patch_src)
    
    if model_type == "stargan":
        patch_pred_norm = model.harmonize(patch_norm, z_style_fixed=style)
        patch_pred = model._denormalize(patch_pred_norm)
        
    elif model_type == "vae":
        patch_pred_norm = model.vae.harmonize(
            patch_norm, 
            x_style_ref=None, 
            z_style_fixed=style, 
            style_dropout_p=0.95
        )
        patch_pred = model._denormalize(patch_pred_norm)
        
    elif model_type == "standard-vae":
        # Le Standard VAE n'a pas de conditionnement de style
        patch_pred_norm = model(patch_norm)
        patch_pred = model._denormalize(patch_pred_norm)
        
    elif model_type == "unet-skip":
        patch_pred_norm = model.model.forward(patch_norm, t=None)
        patch_pred = model._denormalize(patch_pred_norm)
        
    elif model_type == "unet-iffn":
        feature_map = model.feature_extractor(patch_norm)
        patch_pred_norm = model.model.forward(feature_map, t=None)
        patch_pred = model._denormalize(patch_pred_norm)
        
    else:
        raise ValueError(f"Modèle inconnu : {model_type}")
        
    return patch_pred.squeeze(0)

# =============================================================================

def process_subject(
    model_type: str,
    model,
    batch: dict,
    device: torch.device,
    curr_idx: int,
    length_loader: int,
    z_style_fixed: torch.Tensor = None
):
    subject_name = batch['subject_name'][0]
    source_path = batch['source']['path'][0]
    
    # 💡 MODIFICATION : Le dossier de sortie est directement le dossier parent de l'image source
    subj_out_dir = os.path.dirname(source_path)
    pred_path = os.path.join(subj_out_dir, f"harmonized-pet-{model_type}.nii.gz")
    
    print(f"Treating subject: {subject_name} ({curr_idx}/{length_loader})")
    
    # 💡 MODIFICATION : Condition de skip placée avant de préparer les tenseurs
    if os.path.exists(pred_path):
        print(f"⏩ Skip : Le fichier {os.path.basename(pred_path)} existe déjà dans {subj_out_dir}")
        return

    suv_source = batch["source"][tio.DATA].float().to(device)
    if suv_source.ndim == 5:
        suv_source = suv_source.squeeze(1)

    _, d_dim, h_dim, w_dim = suv_source.shape
    orig_d, orig_h, orig_w = d_dim, h_dim, w_dim

    z_patch_size, y_patch_size, x_patch_size = 5, 64, 64 
    
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
    
    z_starts = get_uniform_starts(d_dim, z_patch_size, min_overlap_ratio=0.60)
    y_starts = get_uniform_starts(h_dim, y_patch_size, min_overlap_ratio=0.25)
    x_starts = get_uniform_starts(w_dim, x_patch_size, min_overlap_ratio=0.25)

    total_patches = len(z_starts) * len(y_starts) * len(x_starts)
    gauss_w = make_gaussian_weight_map((z_patch_size, y_patch_size, x_patch_size), sigma_ratio=1.0).to(device)

    pbar = tqdm(total=total_patches, desc=f"Inférence ({model_type})")
    with torch.no_grad():
        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    patch_src = suv_source[:, z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size]

                    if patch_src.mean() < 1e-3: 
                        pbar.update(1)
                        continue
                    
                    patch_pred = infer_patch(model_type, model, patch_src, z_style_fixed)

                    output_volume[z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size] += patch_pred * gauss_w
                    weight_sum[z:z + z_patch_size, y:y + y_patch_size, x:x + x_patch_size] += gauss_w
                    pbar.update(1)
    pbar.close()

    recon_volume = (output_volume / weight_sum.clamp(min=1e-8)).cpu()
    crop_z_start = (d_dim - orig_d) // 2
    crop_y_start = (h_dim - orig_h) // 2
    crop_x_start = (w_dim - orig_w) // 2
    
    recon_volume = recon_volume[
        crop_z_start : crop_z_start + orig_d,
        crop_y_start : crop_y_start + orig_h,
        crop_x_start : crop_x_start + orig_w
    ]

    final_prediction = recon_volume.squeeze().permute(2, 1, 0).numpy()
    final_prediction = np.flip(final_prediction, axis=2)
    final_prediction = np.flip(final_prediction, axis=1)
    
    if z_style_fixed is None and model_type != "standard-vae":
        final_prediction = gaussian_filter(final_prediction, sigma=1.0)

    output_sitk = sitk.GetImageFromArray(final_prediction)
    output_sitk.CopyInformation(sitk.ReadImage(source_path))

    sitk.WriteImage(output_sitk, pred_path)
    print(f"✅ Whole-body prediction saved at: {pred_path}")


# =============================================================================
# Script Niveau 0
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inférence unifiée (Sauvegarde directement dans le dossier du patient)")
    parser.add_argument('--config-file', '-c', type=str, required=True, help='Chemin vers le fichier config YAML.')
    parser.add_argument('--ckpt-path', '-m', type=str, required=True, help='Chemin vers le checkpoint (.ckpt).')
    parser.add_argument('--model-type', type=str, required=True, choices=['stargan', 'vae', 'unet-skip', 'unet-iffn', 'standard-vae'], help='Le type d\'architecture à charger.')
    parser.add_argument('--style-ref', '-s', type=str, required=False, default=None, help='Fichier .pt de style pré-extrait (pour StarGAN et VAE).')
    args = parser.parse_args()

    config = OmegaConf.load(args.config_file)
    config = OmegaConf.to_container(config, resolve=True)
    set_seed(config.get('SEED', 42), workers=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initialisation de l'inférence | Modèle : {args.model_type.upper()} | Device : {device}")

    # 1. Chargement dynamique du modèle
    if args.model_type == 'stargan':
        pipeline_cfg = config.get("pipeline", {})
        style_encoder = StyleEncoder(**config.get("style_encoder", {}))
        style_embedder = StyleEmbedder(style_channels=pipeline_cfg["style_dim"], style_embedding_dim=pipeline_cfg["style_embedding_dim"])
        generator = StarGANv2Generator(**config.get("generator", {}))
        discriminator = StarGANv2Discriminator(num_domains=pipeline_cfg["num_domains"], **config.get("discriminator", {}))
        model = StarGANv2.load_from_checkpoint(args.ckpt_path, generator=generator, style_encoder=style_encoder, style_embedder=style_embedder, discriminator=discriminator, strict=False, **pipeline_cfg)
    
    elif args.model_type == 'vae':
        vae = DisentangledHarmonizationVAE(**config.get('vae', {}))
        model = UnlearningVAE.load_from_checkpoint(args.ckpt_path, vae=vae, strict=False, **config.get('pipeline', {}))
        
    elif args.model_type == 'standard-vae':
        vae = DisentangledHarmonizationVAE(**config.get('vae', {}))
        model = StandardHarmonizationVAE.load_from_checkpoint(args.ckpt_path, vae=vae, strict=False, **config.get('pipeline', {}))
    
    elif args.model_type == 'unet_skip':
        unet = SpectralUNetWithIntermediateFeatures(**config.get('unet', {}))
        model = UnlearningUNetSkip.load_from_checkpoint(args.ckpt_path, model=unet, strict=False, **config.get('pipeline', {}))
    
    elif args.model_type == 'unet_iffn':
        unet = UNet(**config.get('unet', {}))
        feature_extractor = ImageFrequencyFusionModel(**config.get('feature_extractor', {}))
        domain_classifier = DomainClassifier(**config.get('domain_classifier', {}))
        model = UnlearningUNetIFFN.load_from_checkpoint(args.ckpt_path, model=unet, domain_classifier=domain_classifier, feature_extractor=feature_extractor, strict=False, **config.get('pipeline', {}))

    model.to(device)
    model.eval()

    # 2. Gestion du Style (uniquement pour les modèles concernés)
    z_style_fixed = None
    if args.model_type in ['stargan', 'vae']:
        if args.style_ref:
            z_style_fixed = load_style(args.style_ref, device)
        else:
            if args.model_type == 'stargan':
                style_dim = config.get("pipeline", {}).get("style_dim", 64)
                print("⚠️ Aucun style fourni — utilisation d'un vecteur neutre (zéros).")
                z_style_fixed = torch.zeros(1, style_dim, device=device)
            elif args.model_type == 'vae':
                print("⚠️ Aucun style fourni — utilisation d'un vecteur neutre (à 0.95 du style original).")
                z_style_fixed = None

    # 3. Chargement des données
    datamodule = MultiDomainUnlearningDataModule(**config.get('datamodule', {}))
    datamodule.prepare_data()
    datamodule.setup()

    # Inférence forcée sur le dataloader global de test
    loader = datamodule.test_dataloader()
    
    # 4. Exécution
    for idx, batch in enumerate(loader):
        process_subject(
            model_type=args.model_type,
            model=model,
            batch=batch,
            device=device,
            curr_idx=idx + 1,
            length_loader=len(loader),
            z_style_fixed=z_style_fixed
        )