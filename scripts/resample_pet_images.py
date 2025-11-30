import os
import argparse
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


def resample_image(sitk_image, new_spacing=[2.0, 2.0, 2.0], interpolator=sitk.sitkBSpline):
    original_size = sitk_image.GetSize()
    original_spacing = sitk_image.GetSpacing()
    original_origin = sitk_image.GetOrigin()
    original_direction = sitk_image.GetDirection()

    # Calculer la nouvelle taille (Size = PhysicalSize / Spacing)
    # On arrondit à l'entier le plus proche
    new_size = [
        int(round(osz * osp / nsp))
        for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)
    ]

    # 3. Configuration du filtre de resampling
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(original_origin)  # On garde l'origine physique exacte
    resampler.SetOutputDirection(original_direction)  # On garde l'orientation exacte
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)  # Valeur pour les pixels hors champ (air)

    # 4. Exécution
    return resampler.Execute(sitk_image)


def process_dataset(input_root, output_root, target_spacing):
    # Création du dossier racine de sortie
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"Dossier créé : {output_root}")

    # Lister tous les sujets (sous-dossiers)
    subjects = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    subjects.sort()

    print(f"Début du resampling pour {len(subjects)} sujets vers {target_spacing} mm...")

    # Barre de progression
    for subject in tqdm(subjects, desc="Traitement"):
        subject_input_path = os.path.join(input_root, subject)
        subject_output_path = os.path.join(output_root, subject)

        # Recréer le dossier sujet dans la destination
        os.makedirs(subject_output_path, exist_ok=True)

        # Lister les fichiers NIfTI (PT et EARL)
        files = [
            f
            for f in os.listdir(subject_input_path)
            if f.startswith("PT_")
            and f.endswith((".nii", ".nii.gz"))
            and "MIP" not in f
        ]

        for filename in files:
            input_file = os.path.join(subject_input_path, filename)
            output_file = os.path.join(subject_output_path, filename)

            # Optimisation : Si le fichier existe déjà, on le saute (utile si le script plante au milieu)
            if os.path.exists(output_file):
                continue

            # Lecture
            image = sitk.ReadImage(input_file)

            # Resampling
            # Note: Pour les images PET (valeurs continues), B-Spline est idéal.
            # Si vous aviez des masques binaires, il faudrait utiliser sitkNearestNeighbor.
            resampled_image = resample_image(image, new_spacing=target_spacing, interpolator=sitk.sitkBSpline)

            # Sauvegarde
            # On force le format compressé .nii.gz ou non .nii selon l'extension d'origine
            sitk.WriteImage(resampled_image, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resampling isotrope de dataset PET/EARL via SimpleITK."
    )

    # Arguments par défaut basés sur votre demande
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="./datasets/EARL/Ano_Nifti",
        help="Chemin du dataset original.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./datasets/EARL/Ano_Nifti_resampled",
        help="Chemin du dataset de sortie.",
    )
    parser.add_argument(
        "--spacing",
        "-s",
        type=float,
        nargs=3,
        default=[2.0, 2.0, 2.0],
        help="Nouveau spacing cible (x y z).",
    )

    args = parser.parse_args()

    # Lancement
    process_dataset(args.input, args.output, args.spacing)

    print("\n✅ Terminé avec succès.")
