import gc
import inspect
import os

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm


def _filter_components_by_suvmax(
    mask_sitk: sitk.Image,
    pet_sitk: sitk.Image,
    min_suvmax: float,
) -> sitk.Image:
    """
    Post-traitement optionnel : conserve uniquement les composantes connexes
    dont le SUVmax >= min_suvmax (sans tronquer les bords de la tumeur).
    """
    binary = sitk.Cast(mask_sitk > 0, sitk.sitkUInt8)
    cc_img = sitk.ConnectedComponent(binary)

    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc_img, sitk.Cast(pet_sitk, sitk.sitkFloat32))

    keep_labels = [
        lbl for lbl in stats.GetLabels()
        if stats.GetMaximum(lbl) >= min_suvmax
    ]

    if not keep_labels:
        empty = sitk.Image(binary.GetSize(), sitk.sitkUInt8)
        empty.CopyInformation(binary)
        return empty

    cc_arr = sitk.GetArrayFromImage(cc_img)
    filtered_arr = np.isin(cc_arr, keep_labels).astype("uint8")
    out = sitk.GetImageFromArray(filtered_arr)
    out.CopyInformation(binary)
    return out


def main():
    import lionz

    # --- Configuration Segmentation LION (curie PET-only) ---
    cfg = {
        "root_dir": "./data/PET-EARL/",
        "domain": "curie",
        "input_filename": "harmonized-pet-gaussian.nii.gz",
        "output_filename": "lesion.nii.gz",
        "model": "fdg",
        "accelerator": "cuda",   # "cuda" | "cpu" | "mps"
        "override": False,
        
        # Mettre par ex. 5.0 pour supprimer uniquement les faux positifs dont SUVmax < 5.0.
        # Laisser à None pour sauvegarder la prédiction brute de LION.
        "min_component_suvmax": None,
    }

    domain_dir = os.path.join(cfg["root_dir"], cfg["domain"])
    if not os.path.isdir(domain_dir):
        raise FileNotFoundError(f"Dossier introuvable : {domain_dir}")

    subjects = sorted([
        s for s in os.listdir(domain_dir)
        if os.path.isdir(os.path.join(domain_dir, s))
    ])
    print(f"🔎 {len(subjects)} dossiers patients détectés dans {domain_dir}")

    # Vérification dynamique de la signature de lionz.lion selon la version installée
    sig = inspect.signature(lionz.lion)
    lion_kwargs = {}
    if "accelerator" in sig.parameters:
        lion_kwargs["accelerator"] = cfg["accelerator"]

    print(f"🚀 Lancement LION (model='{cfg['model']}', kwargs={lion_kwargs}) | min_component_suvmax={cfg['min_component_suvmax']}")

    processed, skipped, missing, failed = 0, 0, 0, 0

    for subj in tqdm(subjects, desc=f"Segmentation LION ({cfg['domain']})"):
        subj_dir = os.path.join(domain_dir, subj)
        in_path = os.path.join(subj_dir, cfg["input_filename"])
        out_path = os.path.join(subj_dir, cfg["output_filename"])

        if not os.path.exists(in_path):
            missing += 1
            continue

        if os.path.exists(out_path) and not cfg["override"]:
            skipped += 1
            continue

        try:
            img_sitk = sitk.ReadImage(in_path)
            seg_raw = lionz.lion(img_sitk, cfg["model"], **lion_kwargs)

            if not isinstance(seg_raw, sitk.Image):
                seg_sitk = sitk.GetImageFromArray(seg_raw)
                seg_sitk.CopyInformation(img_sitk)
            else:
                seg_sitk = seg_raw

            seg_binary = sitk.Cast(seg_sitk > 0, sitk.sitkUInt8)
            if seg_binary.GetSize() == img_sitk.GetSize():
                seg_binary.CopyInformation(img_sitk)

            if cfg["min_component_suvmax"] is not None:
                seg_binary = _filter_components_by_suvmax(
                    seg_binary, img_sitk, cfg["min_component_suvmax"]
                )

            sitk.WriteImage(seg_binary, out_path, useCompression=True)
            processed += 1

        except Exception as e:
            failed += 1
            print(f"\n❌ Erreur sur {subj} : {e}")

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(
        f"\n✅ Terminé | Traités: {processed} | Ignorés (déjà présents): {skipped} | "
        f"Entrée manquante: {missing} | Erreurs: {failed}"
    )


if __name__ == "__main__":
    main()
