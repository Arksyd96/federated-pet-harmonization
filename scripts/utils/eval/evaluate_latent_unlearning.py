import hashlib
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchio as tio
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, silhouette_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from pet_harmonization.data import MultiDomainUnlearningDataModule
from pet_harmonization.models.unlearning_vae import UnlearningVAE
from pet_harmonization.utils import set_seed


def _subject_rng(subject_uid: str, base_seed: int) -> np.random.RandomState:
    """Seed deterministe par sujet pour extraire les memes patchs entre Stage 1 et Stage 2."""
    digest = hashlib.sha256(f"{base_seed}:{subject_uid}".encode("utf-8")).digest()
    return np.random.RandomState(int.from_bytes(digest[:4], "little"))


def _get_vae_split_lookup(
    root_dir: str,
    train_split_config: list,
    datamodule_seed: int = 42,
) -> Dict[str, str]:
    """Reconstruit le mapping (domain/subject_name -> 'train' | 'val') sans filtrer la cohorte d'eval."""
    dm = MultiDomainUnlearningDataModule(
        root_dir=root_dir,
        split_config=train_split_config,
        seed=datamodule_seed,
    )
    dm.setup()
    lookup: Dict[str, str] = {}
    for subj in dm.train_subjects:
        lookup[f"{subj['domain_name']}/{subj['subject_name']}"] = "train"
    for subj in dm.val_subjects:
        lookup[f"{subj['domain_name']}/{subj['subject_name']}"] = "val"
    return lookup


def _summarize_content_tensor(t: torch.Tensor, include_mean_std: bool = True) -> torch.Tensor:
    """
    Resume un tenseur spatial (B, C, ...) par canal.
    Apres InstanceNorm3d (`content_norm`), mean=0 et std=1 par construction :
    on extrait abs_mean, moment cubique, max et min pour capturer la forme spatiale.
    """
    spatial_axes = tuple(range(2, t.ndim))
    stats = []
    if include_mean_std:
        stats.append(t.mean(dim=spatial_axes))
        stats.append(t.std(dim=spatial_axes, unbiased=False))
    else:
        stats.append(t.abs().mean(dim=spatial_axes))
        stats.append((t ** 3).mean(dim=spatial_axes))
    stats.append(t.amax(dim=spatial_axes))
    stats.append(t.amin(dim=spatial_axes))
    return torch.cat(stats, dim=1)


def _sample_subject_patches(
    subject: tio.Subject,
    subject_uid: str,
    patch_size: Tuple[int, int, int],
    patch_overlap: Tuple[int, int, int],
    patches_per_patient: int,
    min_patch_mean: float,
    spatial_dims: int,
    base_seed: int,
) -> Optional[torch.Tensor]:
    """Collecte les patchs non-vides du sujet et en selectionne K de facon deterministe."""
    grid_sampler = tio.data.GridSampler(subject, patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=32, num_workers=0)

    valid_chunks: List[torch.Tensor] = []
    for patch_batch in patch_loader:
        patch_tio = patch_batch["source"][tio.DATA]
        patch_src = patch_tio.squeeze(1) if spatial_dims == 2 else patch_tio
        valid_mask = patch_src.view(patch_src.shape[0], -1).mean(dim=1) >= min_patch_mean
        if valid_mask.any():
            valid_chunks.append(patch_src[valid_mask])

    if not valid_chunks:
        return None

    all_valid = torch.cat(valid_chunks, dim=0)
    total_valid = all_valid.shape[0]
    if total_valid > patches_per_patient:
        rng = _subject_rng(subject_uid, base_seed)
        idx = np.sort(rng.choice(total_valid, size=patches_per_patient, replace=False))
        return all_valid[torch.from_numpy(idx).long()]
    return all_valid


def _encode_patches_for_model(
    model: UnlearningVAE,
    patches: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """Passe exactement les memes patchs dans le modele et retourne les 3 representations latentes."""
    buckets: Dict[str, List[torch.Tensor]] = defaultdict(list)
    with torch.inference_mode():
        for start in range(0, patches.shape[0], batch_size):
            batch_p = patches[start : start + batch_size].to(device)
            patch_norm = model._normalize(batch_p)
            mu_c, _, mu_s, _ = model.vae.encode(patch_norm)
            norm_mu_c = model.vae.content_norm(mu_c)

            buckets["z_style"].append(mu_s.detach().cpu())
            buckets["z_content_raw"].append(_summarize_content_tensor(mu_c, include_mean_std=True).detach().cpu())
            buckets["z_content_norm"].append(_summarize_content_tensor(norm_mu_c, include_mean_std=False).detach().cpu())

    return {k: torch.cat(v, dim=0).numpy() for k, v in buckets.items()}


def evaluate_representation(
    patch_bundle: dict,
    patient_df: pd.DataFrame,
    rep_key: str,
    domains: List[str],
    seed: int = 42,
) -> dict:
    """
    Calcule les metriques pour une representation (z_content_raw, z_content_norm, z_style) :
      1. Niveau patient (1 point = 1 patient) : Kruskal-Wallis H_med, ARI K-Means, Silhouette PCA.
      2. Probing externe (StratifiedGroupKFold sur Subject_UID) : Balanced Acc par patch et vote patient.
    """
    pat_sub = patient_df[patient_df["Domain"].isin(domains)].reset_index(drop=True)
    rep_cols = [c for c in pat_sub.columns if c.startswith(f"{rep_key}_")]
    X_pat = pat_sub[rep_cols].values
    y_pat_str = pat_sub["Domain"].values

    # 1. Kruskal-Wallis par dimension latente (1 patient = 1 observation)
    h_vals = []
    for j in range(X_pat.shape[1]):
        groups = [X_pat[y_pat_str == d, j] for d in domains]
        if all(len(g) > 1 for g in groups) and np.std(X_pat[:, j]) > 1e-12:
            h_vals.append(kruskal(*groups)[0])
    h_med = float(np.median(h_vals)) if h_vals else 0.0

    # 2. Clustering non supervise (PCA 90% variance + K-Means + Silhouette)
    X_pat_scaled = StandardScaler().fit_transform(X_pat)
    n_comp = min(20, X_pat_scaled.shape[0] - 1, X_pat_scaled.shape[1])
    pca = PCA(n_components=0.90 if n_comp >= 5 else max(1, n_comp), random_state=seed)
    X_pca = pca.fit_transform(X_pat_scaled)
    if X_pca.shape[1] < 2 and X_pat_scaled.shape[1] >= 2 and X_pat_scaled.shape[0] >= 3:
        X_pca = PCA(n_components=2, random_state=seed).fit_transform(X_pat_scaled)

    y_pat_int = LabelEncoder().fit_transform(y_pat_str)
    clusters = KMeans(n_clusters=len(domains), random_state=seed, n_init=10).fit_predict(X_pca)
    ari = float(adjusted_rand_score(y_pat_int, clusters))
    sil = float(silhouette_score(X_pca, y_pat_int)) if len(domains) > 1 else 0.0

    # 3. Probing externe au niveau patch (groupes par patient pour zero leakage)
    mask = np.isin(patch_bundle["Domain"], domains)
    X_patch = patch_bundle[rep_key][mask]
    y_patch_str = patch_bundle["Domain"][mask]
    groups = patch_bundle["Subject_UID"][mask]

    le = LabelEncoder()
    y_patch_int = le.fit_transform(y_patch_str)

    min_subj_per_dom = int(pd.Series(groups).groupby(y_patch_str).nunique().min())
    n_splits = min(5, min_subj_per_dom)

    if n_splits < 2:
        patch_bacc = float("nan")
        patient_bacc = float("nan")
    else:
        oof_probs = np.zeros((len(y_patch_int), len(le.classes_)), dtype=np.float64)
        sgkf = StratifiedGroupKFold(n_splits=n_splits)
        for tr_idx, te_idx in sgkf.split(X_patch, y_patch_int, groups=groups):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=seed),
            )
            clf.fit(X_patch[tr_idx], y_patch_int[tr_idx])
            oof_probs[te_idx] = clf.predict_proba(X_patch[te_idx])

        patch_bacc = float(balanced_accuracy_score(y_patch_int, oof_probs.argmax(axis=1)))

        log_probs = np.log(np.clip(oof_probs, 1e-8, 1.0))
        prob_df = pd.DataFrame(log_probs, columns=le.classes_)
        prob_df["Subject_UID"] = groups
        prob_df["y_true"] = y_patch_int
        agg_df = prob_df.groupby("Subject_UID", sort=False).mean()
        patient_bacc = float(
            balanced_accuracy_score(agg_df["y_true"].astype(int), agg_df[le.classes_].values.argmax(axis=1))
        )

    return {
        "n_patients": len(pat_sub),
        "n_patches": int(mask.sum()),
        "H_med": h_med,
        "ARI_kmeans": ari,
        "Silhouette_PCA": sil,
        "Probe_Patch_BAcc": patch_bacc,
        "Probe_Patient_BAcc": patient_bacc,
        "Chance": 1.0 / len(domains),
    }


def main():
    cfg = {
        "SEED": 101,
        "spatial_dims": 3,
        "patch_size": (16, 64, 64),
        "patch_overlap": (8, 16, 16),
        "patches_per_patient": 64,
        "min_patch_mean": 1e-3,
        "batch_size_encoder": 16,
        "output_dir": "./data/latent_eval/",

        # Renseigner les 2 checkpoints issus du MEME entrainement :
        "checkpoints": {
            "stage1": "runs/sandbox-unlearn-vae/2026_10_23_162550/checkpoints/stage1_final_epoch=039-rec=0.1122-style=0.365-content=0.901.ckpt",
            "stage2": "runs/sandbox-unlearn-vae/2026_10_23_162550/checkpoints/2-epoch=127-rec=0.0201-style=0.960-content=0.172.ckpt",
        },

        # Cohorte d'evaluation (300 patients = 50 par centre sur les 6 centres)
        "datamodule": {
            "root_dir": "./data/PET-EARL/",
            "split_config": [[0, 50], [0, 50], [0, 50], [0, 50], [0, 50], [0, 50]],
            "batch_size": 16,
            "patch_size": [16, 64, 64],
            "num_workers": 4,
            "queue_max_length": 4096,
            "samples_per_volume": 64,
            "seed": 42,
        },
        # Config d'entrainement uniquement pour taguer 'train' / 'val' / 'unseen' dans les CSV
        "train_split_config": [[60, 15], [60, 15], [0, 0], [60, 15], [60, 15], [60, 15]],
        "trained_domains": ["a100", "chb", "lacassagne", "nantes", "rennes"],
        "all_domains": ["a100", "chb", "curie", "lacassagne", "nantes", "rennes"],
    }

    set_seed(cfg["SEED"], workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["output_dir"], exist_ok=True)

    models: Dict[str, UnlearningVAE] = {}
    for stage_name, ckpt in cfg["checkpoints"].items():
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint '{stage_name}' introuvable : {ckpt}")
        m = UnlearningVAE.load_from_checkpoint(ckpt, strict=True)
        m.to(device)
        m.eval()
        models[stage_name] = m

    split_lookup = _get_vae_split_lookup(
        root_dir=cfg["datamodule"]["root_dir"],
        train_split_config=cfg["train_split_config"],
        datamodule_seed=cfg["datamodule"]["seed"],
    )

    datamodule = MultiDomainUnlearningDataModule(**cfg["datamodule"])
    datamodule.prepare_data()
    datamodule.setup()
    loader = datamodule.test_dataloader()
    id_to_domain = {v: k for k, v in datamodule.domain_to_id.items()}

    # Streaming en un seul passage : meme K patchs exacts pour Stage 1 et Stage 2, RAM minimale
    stage_patch_arrays: Dict[str, Dict[str, List[np.ndarray]]] = {s: defaultdict(list) for s in models}
    patch_meta_rows: List[Tuple[str, str, str, str]] = []
    stage_patient_rows: Dict[str, List[dict]] = {s: [] for s in models}

    for batch in tqdm(loader, desc="Extraction latente Stage 1 & Stage 2"):
        subject = tio.utils.get_subjects_from_batch(batch)[0]
        subj_name = str(subject["subject_name"])
        domain_id = int(batch["domain_id"][0])
        domain_name = id_to_domain.get(domain_id, f"center_{domain_id}")
        subject_uid = f"{domain_name}/{subj_name}"
        vae_split = split_lookup.get(subject_uid, "unseen")

        patches = _sample_subject_patches(
            subject=subject,
            subject_uid=subject_uid,
            patch_size=cfg["patch_size"],
            patch_overlap=cfg["patch_overlap"],
            patches_per_patient=cfg["patches_per_patient"],
            min_patch_mean=cfg["min_patch_mean"],
            spatial_dims=cfg["spatial_dims"],
            base_seed=cfg["SEED"],
        )
        if patches is None:
            continue

        n_patches = patches.shape[0]
        for _ in range(n_patches):
            patch_meta_rows.append((subject_uid, subj_name, domain_name, vae_split))

        for stage_name, model in models.items():
            latents = _encode_patches_for_model(
                model=model,
                patches=patches,
                device=device,
                batch_size=cfg["batch_size_encoder"],
            )
            for key, arr in latents.items():
                stage_patch_arrays[stage_name][key].append(arr)

            patient_row = {
                "Subject_UID": subject_uid,
                "Subject_ID": subj_name,
                "Domain": domain_name,
                "vae_split": vae_split,
                "n_patches": n_patches,
            }
            for rep_key, arr in latents.items():
                med = np.median(arr, axis=0)
                for j, val in enumerate(med):
                    patient_row[f"{rep_key}_{j}"] = float(val)
            stage_patient_rows[stage_name].append(patient_row)

    patch_meta = pd.DataFrame(patch_meta_rows, columns=["Subject_UID", "Subject_ID", "Domain", "vae_split"])
    summary_rows = []

    for stage_name in models:
        patient_df = pd.DataFrame(stage_patient_rows[stage_name])
        csv_path = os.path.join(cfg["output_dir"], f"{stage_name}_patient_latents.csv")
        patient_df.to_csv(csv_path, index=False)
        print(f"CSV patient sauvegarde : {csv_path} ({len(patient_df)} lignes)")

        patch_bundle = {k: np.concatenate(v, axis=0) for k, v in stage_patch_arrays[stage_name].items()}
        patch_bundle["Subject_UID"] = patch_meta["Subject_UID"].values
        patch_bundle["Domain"] = patch_meta["Domain"].values
        patch_bundle["vae_split"] = patch_meta["vae_split"].values

        for scope_name, doms in [
            ("5_trained_centers", cfg["trained_domains"]),
            ("6_all_with_curie", cfg["all_domains"]),
        ]:
            for rep_key in ["z_content_raw", "z_content_norm", "z_style"]:
                metrics = evaluate_representation(
                    patch_bundle=patch_bundle,
                    patient_df=patient_df,
                    rep_key=rep_key,
                    domains=doms,
                    seed=cfg["SEED"],
                )
                summary_rows.append({"Stage": stage_name, "Scope": scope_name, "Latent": rep_key, **metrics})

        # Verification dediee sur les sujets de validation VAE presents dans les 300 patients
        val_mask = patch_bundle["vae_split"] == "val"
        val_patient_df = patient_df[patient_df["vae_split"] == "val"].reset_index(drop=True)
        if val_mask.sum() > 0 and val_patient_df["Domain"].nunique() == len(cfg["trained_domains"]):
            val_patch_bundle = {
                k: v[val_mask] if isinstance(v, np.ndarray) else v
                for k, v in patch_bundle.items()
            }
            for rep_key in ["z_content_raw", "z_content_norm", "z_style"]:
                metrics = evaluate_representation(
                    patch_bundle=val_patch_bundle,
                    patient_df=val_patient_df,
                    rep_key=rep_key,
                    domains=cfg["trained_domains"],
                    seed=cfg["SEED"],
                )
                summary_rows.append({"Stage": stage_name, "Scope": "5_trained_VAL_ONLY", "Latent": rep_key, **metrics})

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(cfg["output_dir"], "stage1_vs_stage2_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print("BILAN STAGE 1 vs STAGE 2 — ESPACE LATENT")
    print("=" * 100)
    print(summary_df.round(3).to_string(index=False))
    print(f"\nResume sauvegarde dans : {summary_path}")


if __name__ == "__main__":
    main()
