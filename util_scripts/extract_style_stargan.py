import argparse
import math
import os
import random

import numpy as np
import torch
import torchio as tio
from tqdm import tqdm
from omegaconf import OmegaConf

from modules.data import Float32Lambda

# from modules.data import Float32Lambda
from modules.models.starganv2 import (
    StarGANv2, StyleEncoder, StarGANv2Discriminator,
    StarGANv2Generator, StyleEmbedder,
)
from modules.utils import set_seed


# =============================================================================
# Utilitaires patchs
# =============================================================================

def get_uniform_starts(dim_size: int, patch_size: int, min_overlap_ratio: float = 0.1) -> list[int]:
    if dim_size <= patch_size:
        return [0]
    max_stride = max(patch_size - int(patch_size * min_overlap_ratio), 1)
    n_patches  = math.ceil((dim_size - patch_size) / max_stride) + 1
    stride_f   = (dim_size - patch_size) / max(n_patches - 1, 1)
    return [round(i * stride_f) for i in range(n_patches)]


# =============================================================================
# Extraction du style depuis un volume unique
# =============================================================================

def extract_style_from_volume(
    nifti_path: str,
    model: StarGANv2,
    device: torch.device,
    z_patch_size: int = 5,
    y_patch_size: int = 64,
    x_patch_size: int = 64,
) -> torch.Tensor:
    """
    Charge un volume, extrait le style code sur chaque patch valide,
    et retourne la moyenne des style codes du volume.

    Returns
    -------
    style_mean : (1, style_dim) sur GPU
    """
    transform = tio.Compose([Float32Lambda(), tio.ToCanonical()])
    subject   = tio.Subject(source=tio.Image(nifti_path, type=tio.INTENSITY))
    subject   = transform(subject)

    vol = subject["source"][tio.DATA].float().to(device)  # (1, D, H, W)
    _, d_dim, h_dim, w_dim = vol.shape

    z_starts = get_uniform_starts(d_dim, z_patch_size, min_overlap_ratio=0.1)
    y_starts = get_uniform_starts(h_dim, y_patch_size, min_overlap_ratio=0.1)
    x_starts = get_uniform_starts(w_dim, x_patch_size, min_overlap_ratio=0.1)

    style_list = []

    with torch.no_grad():
        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    patch = vol[
                        :,
                        z:z + z_patch_size,
                        y:y + y_patch_size,
                        x:x + x_patch_size,
                    ]
                    if patch.mean() < 1e-3:
                        continue
                    patch_norm = model._normalize(patch)
                    style_list.append(model.style_encoder(patch_norm))  # (1, style_dim)

    if not style_list:
        return None

    return torch.stack(style_list, dim=0).mean(dim=0)  # (1, style_dim)


# =============================================================================
# Agrégation multi-patients
# =============================================================================

def aggregate_styles(
    all_styles: torch.Tensor,   # (N, style_dim)
    mode: str,
    n_clusters: int = 1,
) -> torch.Tensor:
    """
    Agrège N vecteurs de style en un seul.

    Parameters
    ----------
    all_styles : (N, style_dim)
    mode       : 'mean' | 'median' | 'medoid' | 'cluster'
    n_clusters : utilisé uniquement si mode='cluster'

    Returns
    -------
    style : (1, style_dim)
    """
    if mode == "mean":
        return all_styles.mean(dim=0, keepdim=True)

    elif mode == "median":
        return all_styles.median(dim=0).values.unsqueeze(0)

    elif mode == "medoid":
        # Distance L2 de chaque style vers tous les autres
        # Medoid = style dont la somme des distances est minimale
        dists = torch.cdist(all_styles, all_styles, p=2)   # (N, N)
        medoid_idx = dists.sum(dim=1).argmin()
        return all_styles[medoid_idx].unsqueeze(0)

    elif mode == "cluster":
        # K-means simple sur GPU (implémentation légère, pas de dépendance sklearn)
        N, D = all_styles.shape
        k    = min(n_clusters, N)

        # Init aléatoire des centroïdes
        indices    = torch.randperm(N, device=all_styles.device)[:k]
        centroids  = all_styles[indices].clone()

        for _ in range(100):    # max 100 itérations
            # Assignation
            dists    = torch.cdist(all_styles, centroids, p=2)   # (N, k)
            labels   = dists.argmin(dim=1)                        # (N,)

            # Mise à jour des centroïdes
            new_centroids = torch.stack([
                all_styles[labels == i].mean(dim=0) if (labels == i).any()
                else centroids[i]
                for i in range(k)
            ])

            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        # Retourner le centroïde le plus central (medoid parmi les centroïdes)
        dists_c    = torch.cdist(centroids, centroids, p=2)
        best_idx   = dists_c.sum(dim=1).argmin()
        return centroids[best_idx].unsqueeze(0)

    else:
        raise ValueError(f"Mode inconnu : {mode}. Choisir parmi : mean, median, medoid, cluster")


# =============================================================================
# Entrée principale
# =============================================================================

def main(args):
    config = OmegaConf.load(args.config_file)
    config = OmegaConf.to_container(config, resolve=True)
    set_seed(config.get("SEED", 42), workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # ── Chargement du modèle ──────────────────────────────────────────────────
    pipeline_cfg  = config.get("pipeline", {})
    style_encoder = StyleEncoder(**config.get("style_encoder", {}))
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

    # ── Collecte des patients ─────────────────────────────────────────────────
    all_patients = sorted([
        p for p in os.listdir(args.repo_ref)
        if os.path.isdir(os.path.join(args.repo_ref, p))
    ])

    if args.num_samples is not None and args.num_samples < len(all_patients):
        all_patients = random.sample(all_patients, args.num_samples)
        print(f"Sampled {args.num_samples} patients from {len(all_patients)} available.")
    else:
        print(f"Using all {len(all_patients)} patients.")

    # ── Extraction des styles (1 par 1, GPU) ─────────────────────────────────
    style_codes = []

    pbar = tqdm(all_patients, desc=f"Extracting styles [{args.mode}]", unit="patient")
    for patient_name in pbar:
        nifti_path = os.path.join(args.repo_ref, patient_name, args.filename_ref)

        if not os.path.exists(nifti_path):
            pbar.write(f"  ⚠️  Fichier introuvable, skip : {nifti_path}")
            continue

        style = extract_style_from_volume(nifti_path, model, device)

        if style is None:
            pbar.write(f"  ⚠️  Aucun patch valide pour : {patient_name}")
            continue

        style_codes.append(style)    # (1, style_dim) sur GPU
        pbar.set_postfix({"valid": len(style_codes)})

    if not style_codes:
        raise RuntimeError("Aucun style extrait — vérifier repo_ref et filename_ref.")

    print(f"\n{len(style_codes)} patients valides sur {len(all_patients)} tentés.")

    # ── Agrégation ───────────────────────────────────────────────────────────
    all_styles = torch.cat(style_codes, dim=0)   # (N, style_dim) sur GPU
    print(f"Aggregating styles with mode='{args.mode}' — input shape: {all_styles.shape}")

    aggregated = aggregate_styles(all_styles, mode=args.mode, n_clusters=args.n_clusters)
    print(f"Aggregated style shape: {aggregated.shape}")

    # ── Sauvegarde ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({
        "style":       aggregated.cpu(),    # (1, style_dim)
        "mode":        args.mode,
        "n_patients":  len(style_codes),
        "style_dim":   aggregated.shape[-1],
        "repo_ref":    args.repo_ref,
        "filename_ref": args.filename_ref,
    }, args.output)

    print(f"\nStyle saved at: {args.output}")
    print(f"  mode       : {args.mode}")
    print(f"  n_patients : {len(style_codes)}")
    print(f"  style_dim  : {aggregated.shape[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extraction et agrégation du vecteur de style pour StarGAN v2"
    )
    parser.add_argument('--config-file',   '-c', type=str, required=True,
                        help='Chemin vers le fichier config YAML.')
    parser.add_argument('--ckpt-path',     '-m', type=str, required=True,
                        help='Chemin vers le checkpoint (.ckpt).')
    parser.add_argument('--repo-ref',      '-r', type=str, required=True,
                        help='Dossier racine contenant les patients de référence.')
    parser.add_argument('--filename-ref',  '-f', type=str, required=True,
                        help='Nom du fichier NIfTI dans chaque dossier patient (ex: PET.nii.gz).')
    parser.add_argument('--num-samples',   '-n', type=int, default=None,
                        help='Nombre de patients à échantillonner.')
    parser.add_argument('--mode',               type=str, default='medoid',
                        choices=['mean', 'median', 'medoid', 'cluster'],
                        help='Mode d\'agrégation (défaut: medoid).')
    parser.add_argument('--n-clusters',         type=int, default=1,
                        help='Nombre de clusters (uniquement si mode=cluster, défaut: 1).')
    parser.add_argument('--output',        '-o', type=str, required=True,
                        help='Chemin de sauvegarde du style agrégé (.pt).')
    args = parser.parse_args()

    if not os.path.exists(args.repo_ref):
        raise FileNotFoundError(f"Repo introuvable : {args.repo_ref}")

    main(args)