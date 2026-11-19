import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


def _has_valid_pet_body_and_lesion(args: Tuple[str, str, str]) -> dict:
    """Verifie la presence de pet* + body* et si lesion.nii.gz est present et non vide."""
    domain, subj, root_dir = args
    sitk.ProcessObject.SetGlobalWarningDisplay(False)

    sdir = os.path.join(root_dir, domain, subj)
    files = os.listdir(sdir)
    has_pet = any(f.lower().startswith("pet") and f.endswith((".nii", ".nii.gz")) for f in files)
    has_body = any(f.lower().startswith("body") and f.endswith((".nii", ".nii.gz")) for f in files)

    has_lesion = False
    lesion_path = os.path.join(sdir, "lesion.nii.gz")
    if os.path.exists(lesion_path):
        mask_img = sitk.ReadImage(lesion_path)
        has_lesion = bool(np.any(sitk.GetArrayViewFromImage(mask_img) > 0))

    return {
        "domain": domain,
        "subject_id": subj,
        "valid_pet_body": has_pet and has_body,
        "has_lesion": has_lesion,
    }


def scan_dataset(root_dir: str, num_workers: int = 32) -> pd.DataFrame:
    """Scanne tous les domaines et retourne un DataFrame (domain, subject_id, valid_pet_body, has_lesion)."""
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Dossier racine introuvable : {root_dir}")

    domains = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    tasks = []
    for d in domains:
        dom_dir = os.path.join(root_dir, d)
        for s in sorted(os.listdir(dom_dir)):
            if os.path.isdir(os.path.join(dom_dir, s)):
                tasks.append((d, s, root_dir))

    rows = []
    workers = min(num_workers, max(1, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_has_valid_pet_body_and_lesion, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Scan dataset (lesions + PET/body)", unit="patient"):
            rows.append(fut.result())

    return pd.DataFrame(rows).sort_values(["domain", "subject_id"]).reset_index(drop=True)


def generate_manifest(
    scan_df: pd.DataFrame,
    domain_rules: Dict[str, dict],
    seed: int = 101,
) -> pd.DataFrame:
    """
    Genere un manifeste CSV (domain, subject_id, split, has_lesion) selon des regles par domaine :
      - train         : int obligatoire (peut etre 0, ex. curie ; jamais None)
      - test          : int ou None (None = tout le reste part en test)
      - train_lesions : quota de patients avec lesion inclus dans train
    """
    rng = np.random.RandomState(seed)
    manifest_rows = []

    for domain in sorted(domain_rules.keys()):
        rule = domain_rules[domain]
        n_train = rule.get("train")
        n_test = rule.get("test")
        n_train_lesions = rule.get("train_lesions", 0)

        if n_train is None:
            raise ValueError(f"Pour '{domain}', 'train' ne peut pas etre None (mettre 0 si exclu du train).")
        if n_train < 0 or n_train_lesions < 0 or (n_test is not None and n_test < 0):
            raise ValueError(f"Effectifs negatifs invalides pour '{domain}' : {rule}")

        dom_df = scan_df[(scan_df["domain"] == domain) & (scan_df["valid_pet_body"])].copy()
        if dom_df.empty:
            print(f"⚠️ Domaine '{domain}' vide ou absent.")
            continue

        lesion_ids = dom_df[dom_df["has_lesion"]]["subject_id"].tolist()
        clean_ids = dom_df[~dom_df["has_lesion"]]["subject_id"].tolist()
        rng.shuffle(lesion_ids)
        rng.shuffle(clean_ids)

        if n_train > len(dom_df):
            raise ValueError(f"'{domain}' : train={n_train} demande mais seulement {len(dom_df)} patients disponibles.")

        # 1. Quota de lesions en train
        take_les = min(n_train_lesions, n_train, len(lesion_ids))
        if take_les < n_train_lesions:
            print(f"⚠️ [{domain}] Quota train_lesions={n_train_lesions} impossible (dispo={len(lesion_ids)}), pris={take_les}.")

        train_les = lesion_ids[:take_les]
        rem_les = lesion_ids[take_les:]

        # 2. Completion du train avec des sans-lesion (fallback sur lesions restantes si insuffisant)
        need_clean = n_train - take_les
        take_clean = min(need_clean, len(clean_ids))
        train_clean = clean_ids[:take_clean]
        rem_clean = clean_ids[take_clean:]

        extra_les_for_train = []
        if take_clean < need_clean:
            deficit = need_clean - take_clean
            print(f"⚠️ [{domain}] Pas assez de patients sans lesion ({len(clean_ids)} dispo) : +{deficit} avec lesion ajoutes en train.")
            extra_les_for_train = rem_les[:deficit]
            rem_les = rem_les[deficit:]

        train_ids = train_les + train_clean + extra_les_for_train
        rng.shuffle(train_ids)

        # 3. Reste -> test (tronque a n_test si specifie)
        remaining_ids = rem_les + rem_clean
        rng.shuffle(remaining_ids)
        if n_test is None:
            test_ids = remaining_ids
        else:
            if n_test > len(remaining_ids):
                print(f"⚠️ [{domain}] test={n_test} demande mais seulement {len(remaining_ids)} restants.")
            test_ids = remaining_ids[:n_test]

        lesion_set = set(lesion_ids)
        for sid in train_ids:
            manifest_rows.append({"domain": domain, "subject_id": sid, "split": "train", "has_lesion": sid in lesion_set})
        for sid in test_ids:
            manifest_rows.append({"domain": domain, "subject_id": sid, "split": "test", "has_lesion": sid in lesion_set})

    return pd.DataFrame(manifest_rows)


def print_summary(manifest_df: pd.DataFrame, out_path: str, seed: int) -> None:
    print(f"\n=== Manifest (seed={seed}) -> {out_path} ===")
    for dom, g in manifest_df.groupby("domain", sort=True):
        tr = g[g["split"] == "train"]
        te = g[g["split"] == "test"]
        print(
            f"  {dom:12s} | Total={len(g):3d} | "
            f"Train={len(tr):3d} (lesions={int(tr['has_lesion'].sum()):2d}) | "
            f"Test={len(te):3d} (lesions={int(te['has_lesion'].sum()):2d})"
        )
    print(
        f"  {'TOTAL':12s} | Total={len(manifest_df):3d} | "
        f"Train={int((manifest_df['split'] == 'train').sum()):3d} "
        f"(lesions={int(manifest_df.loc[manifest_df['split'] == 'train', 'has_lesion'].sum()):3d}) | "
        f"Test={int((manifest_df['split'] == 'test').sum()):3d} "
        f"(lesions={int(manifest_df.loc[manifest_df['split'] == 'test', 'has_lesion'].sum()):3d})"
    )


def main():
    parser = argparse.ArgumentParser(description="Genere un manifeste CSV train/test fige pour PET-EARL.")
    parser.add_argument("--root-dir", type=str, default="./data/PET-EARL/")
    parser.add_argument("--output", "-o", type=str, default="./data/manifest.csv")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    domain_rules = {
        "a100":       {"train": 50, "test": None, "train_lesions": 5},
        "chb":        {"train": 50, "test": None, "train_lesions": 5},
        "curie":      {"train": 0,  "test": None, "train_lesions": 5},
        "lacassagne": {"train": 50, "test": None, "train_lesions": 5},
        "nantes":     {"train": 50, "test": None, "train_lesions": 5},
        "rennes":     {"train": 50, "test": None, "train_lesions": 5},
    }

    scan_df = scan_dataset(args.root_dir, num_workers=args.workers)
    manifest_df = generate_manifest(scan_df, domain_rules=domain_rules, seed=args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    manifest_df.to_csv(args.output, index=False)
    print_summary(manifest_df, args.output, args.seed)


if __name__ == "__main__":
    main()
