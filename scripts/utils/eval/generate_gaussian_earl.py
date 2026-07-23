import os
import argparse
import numpy as np
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def apply_gaussian_blur(input_path, output_path, fwhm_mm=8.0):
    """Applique un flou Gaussien sur une image NIfTI (basé sur le FWHM) et force le fond à zéro absolu."""
    try:
        if not os.path.exists(input_path):
            return False, f"Fichier source introuvable : {os.path.basename(input_path)}"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Conversion FWHM (mm) vers Sigma (mm)
        sigma_mm = fwhm_mm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        
        image = sitk.ReadImage(input_path)
        gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian_filter.SetSigma(sigma_mm)
        gaussian_filter.SetNormalizeAcrossScale(False)
        blurred_image = gaussian_filter.Execute(image)
        
        arr_orig = sitk.GetArrayFromImage(image)
        arr_blurred = sitk.GetArrayFromImage(blurred_image)
        
        arr_blurred[arr_orig < 1e-4] = 0.0
        
        final_image = sitk.GetImageFromArray(arr_blurred)
        final_image.CopyInformation(image)
        
        # 6. Sauvegarde (écrase silencieusement le fichier s'il existe déjà)
        sitk.WriteImage(final_image, output_path)
        
        return True, "Succès"
    except Exception as e:
        return False, str(e)

def process_subject(args_tuple):
    """Fonction worker traitant un sujet unique."""
    subj_id, subj_in_dir, subj_out_dir, filename, target_name, fwhm = args_tuple
    
    # Gestion de l'extension .nii.gz pour l'entrée
    if not filename.endswith((".nii", ".nii.gz")):
        filename += ".nii.gz"
        
    # Gestion de l'extension .nii.gz pour la sortie
    if not target_name.endswith((".nii", ".nii.gz")):
        target_name += ".nii.gz"
        
    input_path = os.path.join(subj_in_dir, filename)
    output_path = os.path.join(subj_out_dir, target_name)
    
    success, message = apply_gaussian_blur(input_path, output_path, fwhm)
    return subj_id, success, message

def main():
    parser = argparse.ArgumentParser(description="Générateur pseudo-EARL (Miroir + Sélection de sujets).")
    
    parser.add_argument("--input", type=str, required=True, help="Repo source")
    parser.add_argument("--output", type=str, required=True, help="Repo destination")
    parser.add_argument("--filename", type=str, required=True, help="Nom exact du fichier d'entrée (ex: pet_std)")
    parser.add_argument("--output-filename", type=str, default='gaussian-earl.nii.gz', help="Nom du fichier de sortie")
    parser.add_argument("--include-only", nargs='*', default=None, help="Liste des IDs de sujets à traiter (ex: S01 S05)")
    parser.add_argument("--num-workers", type=int, default=multiprocessing.cpu_count() - 2, help="Nombre de processus")
    parser.add_argument("--fwhm", type=float, default=8.0, help="FWHM en mm (ex: 8.0 pour un pseudo-EARL1)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logging.error(f"Le dossier source {args.input} n'existe pas.")
        return

    # Liste initiale de tous les dossiers sujets
    all_subjects = [s for s in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, s))]
    
    # Filtrage selon include-only
    if args.include_only is not None:
        subjects = [s for s in all_subjects if s in args.include_only]
        logging.info(f"Filtre activé : {len(subjects)} sujets sélectionnés sur {len(all_subjects)}.")
    else:
        subjects = all_subjects

    if not subjects:
        logging.warning("Aucun sujet ne correspond aux critères de traitement.")
        return

    # Préparation des tâches
    tasks = [
        (s, os.path.join(args.input, s), os.path.join(args.output, s), args.filename, args.output_filename, args.fwhm)
        for s in subjects
    ]

    logging.info(f"🚀 Traitement lancé pour {len(subjects)} sujets avec un FWHM de {args.fwhm} mm.")
    success_count = 0
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = { executor.submit(process_subject, t): t for t in tasks }
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Génération Pseudo-EARL"):
            res = future.result()
            
            subj_id = res[0]
            is_ok = res[1]
            msg = res[2]
            
            if is_ok:
                success_count += 1
            else:
                logging.warning(f"[{subj_id}] {msg}")

    logging.info(f"✅ Terminé : {success_count}/{len(subjects)} images générées dans {args.output}")

if __name__ == "__main__":
    main()