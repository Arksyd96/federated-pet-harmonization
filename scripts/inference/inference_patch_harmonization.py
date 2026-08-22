import argparse
import os
import torch
import torchio as tio
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np

from pet_harmonization.data import MultiDomainUnlearningDataModule
from pet_harmonization.models.starganv2 import StarGANv2, StyleEncoder, StarGANv2Discriminator, StarGANv2Generator, StyleEmbedder
from pet_harmonization.models.harmonization_vae import DisentangledHarmonizationVAE, UnlearningVAE, StandardHarmonizationVAE
from pet_harmonization.models.unet_v2_skip import UnlearningUNet as UnlearningUNetSkip, SpectralUNetWithIntermediateFeatures
from pet_harmonization.models.unet import UnlearningUNet as UnlearningUNetIFFN, UNet
from pet_harmonization.models.domain_classifier import DomainClassifier
from pet_harmonization.models.iffn import ImageFrequencyFusionModel
from pet_harmonization.utils import set_seed


def load_style(style_path: str, device: torch.device) -> torch.Tensor:
    checkpoint = torch.load(style_path, map_location=device)
    style = checkpoint["style"].to(device)
    
    print(f"Style loaded from : {style_path}")
    print(f"  mode       : {checkpoint.get('mode', 'N/A')}")
    print(f"  n_patients : {checkpoint.get('n_patients', 'N/A')}")
    print(f"  style_dim  : {checkpoint.get('style_dim', style.shape[-1])}")
    print(f"  repo_ref   : {checkpoint.get('repo_ref', 'N/A')}")
    
    return style


def infer_patch(model_type: str, model, patch_src: torch.Tensor, style: torch.Tensor = None):
    patch_norm = model._normalize(patch_src)
    
    if style is not None:
        style = style.expand(patch_norm.shape[0], -1)
    
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
        
    return patch_pred


def process_subject(
    model_type: str,
    model,
    batch: dict,
    device: torch.device,
    curr_idx: int,
    length_loader: int,
    z_style_fixed: torch.Tensor = None,
    patch_size: tuple = (64, 64, 64),
    patch_overlap: tuple = (32, 32, 32),
    filename: str = None,
    override: bool = False
):
    subject = tio.utils.get_subjects_from_batch(batch)[0]
    
    subject_name = subject['subject_name']
    subj_out_dir = batch['subject_path'][0]
    
    filename = f"{model_type}" if filename is None else filename
    pred_path = os.path.join(subj_out_dir, f"harmonized-pet-{filename}.nii.gz")
    
    print(f"\nTreating subject: {subject_name} ({curr_idx}/{length_loader})")
    
    if os.path.exists(pred_path):
        if not override:
            print(f"⏩ Skip : Le fichier {os.path.basename(pred_path)} existe déjà.")
            return
        else:
            print(f"⚠️ Override : Le fichier {os.path.basename(pred_path)} sera écrasé.")

    print(f"Patch size: {patch_size}, Overlap: {patch_overlap}")
    
    grid_sampler = tio.data.GridSampler(subject, patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=4, num_workers=0) 

    aggregator = tio.data.GridAggregator(grid_sampler, overlap_mode='hann')

    with torch.inference_mode():
        for patch_batch in tqdm(patch_loader, desc=f"Inférence ({model_type})"):
            locations = patch_batch[tio.LOCATION]
            
            patch_tio = patch_batch['source'][tio.DATA].to(device)
            patch_src = patch_tio.squeeze(1)

            if patch_src.mean() < 1e-3: 
                patch_pred_tio = torch.zeros_like(patch_tio)
                aggregator.add_batch(patch_pred_tio, locations)
                continue
            
            patch_pred = infer_patch(model_type, model, patch_src, z_style_fixed)
            patch_pred_tio = patch_pred.unsqueeze(1)
            
            aggregator.add_batch(patch_pred_tio, locations)

    recon_tensor = aggregator.get_output_tensor()
    output_image = tio.ScalarImage(tensor=recon_tensor, affine=subject['source'].affine)
    output_image.save(pred_path)
    
    print(f"✅ Whole-body prediction saved correctly at: {pred_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inférence unifiée")
    parser.add_argument('--config-file', '-c', type=str, required=True)
    parser.add_argument('--ckpt-path', '-m', type=str, required=True)
    parser.add_argument('--model-type', type=str, required=True, choices=['stargan', 'vae', 'unet-skip', 'unet-iffn', 'standard-vae'])
    parser.add_argument('--style-ref', '-s', type=str, required=False, default=None)
    parser.add_argument('--filename', '-f', type=str, required=False, default=None)
    parser.add_argument('--patch-overlap', '-o', type=int, nargs=3, required=False, default=(32, 32, 32))
    parser.add_argument('--override', action='store_true')
    args = parser.parse_args()

    config = OmegaConf.load(args.config_file)
    config = OmegaConf.to_container(config, resolve=True)
    set_seed(config.get('SEED', 42), workers=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initialisation de l'inférence | Modèle : {args.model_type.upper()} | Device : {device}")

    if args.model_type == 'stargan':
        pipeline_cfg = config.get("pipeline", {})
        style_encoder = StyleEncoder(**config.get("style_encoder", {}))
        style_embedder = StyleEmbedder(style_channels=pipeline_cfg["style_dim"], style_embedding_dim=pipeline_cfg["style_embedding_dim"])
        generator = StarGANv2Generator(**config.get("generator", {}))
        discriminator = StarGANv2Discriminator(num_domains=pipeline_cfg["num_domains"], **config.get("discriminator", {}))
        model = StarGANv2.load_from_checkpoint(args.ckpt_path, generator=generator, style_encoder=style_encoder, style_embedder=style_embedder, discriminator=discriminator, strict=False, **pipeline_cfg)
    
    elif args.model_type == 'vae':
        vae = DisentangledHarmonizationVAE(**config.get('vae', {}))
        model = UnlearningVAE.load_from_checkpoint(args.ckpt_path, vae=vae, strict=True, **config.get('pipeline', {}))
        
    elif args.model_type == 'standard-vae':
        vae = DisentangledHarmonizationVAE(**config.get('vae', {}))
        model = StandardHarmonizationVAE.load_from_checkpoint(args.ckpt_path, vae=vae, strict=True, **config.get('pipeline', {}))
    
    elif args.model_type == 'unet-skip':
        unet = SpectralUNetWithIntermediateFeatures(**config.get('unet', {}))
        model = UnlearningUNetSkip.load_from_checkpoint(args.ckpt_path, model=unet, strict=True, **config.get('pipeline', {}))
    
    elif args.model_type == 'unet-iffn':
        unet = UNet(**config.get('unet', {}))
        feature_extractor = ImageFrequencyFusionModel(**config.get('feature_extractor', {}))
        domain_classifier = DomainClassifier(**config.get('domain_classifier', {}))
        model = UnlearningUNetIFFN.load_from_checkpoint(args.ckpt_path, model=unet, domain_classifier=domain_classifier, feature_extractor=feature_extractor, strict=False, **config.get('pipeline', {}))
    
    else:
        raise ValueError(f"Modèle inconnu ou mal formaté : {args.model_type}")

    model.to(device)
    model.eval()

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

    datamodule = MultiDomainUnlearningDataModule(**config.get('datamodule', {}))
    datamodule.prepare_data()
    datamodule.setup()

    loader = datamodule.test_dataloader()
    
    for idx, batch in enumerate(loader):
        process_subject(
            model_type=args.model_type,
            model=model,
            batch=batch,
            device=device,
            curr_idx=idx + 1,
            length_loader=len(loader),
            z_style_fixed=z_style_fixed,
            patch_size=config.get('datamodule', {}).get('patch_size', (5, 64, 64)),
            patch_overlap=args.patch_overlap,
            filename=args.filename,
            override=args.override
        )