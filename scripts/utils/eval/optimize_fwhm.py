import os
import argparse
import random
import logging
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import label
import concurrent.futures
from tqdm import tqdm

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dictionnaire global servant de mémoire tampon partagée entre les threads
MEMORY_POOL = {}

def align_mask_to_reference(mask_img, ref_img):
    """Aligne spatialement le masque sur l'image de référence avec une interpolation au plus proche voisin."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img) # Sets Size, Spacing, Origin, Direction
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(mask_img)

def precompute_subject_data(subj_dir, subj_id, args):
    """Charge les images, aligne le masque, et pré-calcule les VOI50 pour toutes les lésions."""
    std_path = os.path.join(subj_dir, args.std_filename)
    earl_path = os.path.join(subj_dir, args.earl_filename)
    mask_path = os.path.join(subj_dir, args.mask_filename)
    
    # Lecture des images via SimpleITK
    std_img = sitk.ReadImage(std_path, sitk.sitkFloat32)
    earl_img = sitk.ReadImage(earl_path, sitk.sitkFloat32)
    mask_img = sitk.ReadImage(mask_path, sitk.sitkUInt8)
    
    # Alignement du masque sur le PET Standard
    aligned_mask_img = align_mask_to_reference(mask_img, std_img)
    
    # Conversion en tableaux NumPy
    earl_arr = sitk.GetArrayFromImage(earl_img)
    mask_arr = sitk.GetArrayFromImage(aligned_mask_img)
    
    # Extraction des composantes connexes (lésions)
    labeled_mask, num_lesions = label(mask_arr > 0)
    
    lesions_data = []
    for k in range(1, num_lesions + 1):
        lesion_mask = (labeled_mask == k)
        suv_max_earl = np.max(earl_arr[lesion_mask])
        
        if suv_max_earl <= 0:
            continue # Ignorer les artefacts à SUV négatif ou nul
            
        # Création du masque VOI50 basé sur le SUVmax de l'image de référence EARL
        voi50_mask = lesion_mask & (earl_arr >= 0.5 * suv_max_earl)
        
        if not np.any(voi50_mask):
            continue
            
        suv_mean_earl = np.mean(earl_arr[voi50_mask])
        
        lesions_data.append({
            'lesion_id': k,
            'voi50_mask': voi50_mask,
            'suv_max_earl': suv_max_earl,
            'suv_mean_earl': suv_mean_earl
        })
        
    return {
        'std_img': std_img,
        'lesions': lesions_data
    }

def process_fwhm_task(task_args):
    """Tâche exécutée par un thread : applique le flou et extrait les SUV sur les VOI50 pré-calculés."""
    subj_id, fwhm = task_args
    data = MEMORY_POOL[subj_id]
    std_img = data['std_img']
    lesions = data['lesions']
    
    results = []
    
    if fwhm == 0.0:
        # FWHM 0 signifie aucun flou, on utilise l'image standard brute
        arr_blurred = sitk.GetArrayFromImage(std_img)
    else:
        # Conversion FWHM (mm) vers Sigma (mm)
        sigma_mm = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        
        # Le filtre SimpleITK gère nativement le spacing (mm -> voxels)
        gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian_filter.SetSigma(sigma_mm)
        gaussian_filter.SetNormalizeAcrossScale(False)
        blurred_img = gaussian_filter.Execute(std_img)
        arr_blurred = sitk.GetArrayFromImage(blurred_img)
        
    # Calcul des métriques pour chaque composante
    for lesion in lesions:
        voi50_mask = lesion['voi50_mask']
        
        suv_mean_filt = np.mean(arr_blurred[voi50_mask])
        suv_max_filt = np.max(arr_blurred[voi50_mask])
        
        # absolute Relative Error (aRE) sur le SUVmean
        safe_earl_mean = lesion['suv_mean_earl'] if lesion['suv_mean_earl'] != 0 else 1e-8
        are_mean = abs(suv_mean_filt - safe_earl_mean) / safe_earl_mean * 100.0
        
        # Biais relatif (Delta SUV %)
        bias_mean = ((suv_mean_filt - safe_earl_mean) / safe_earl_mean) * 100.0
        
        results.append({
            'Subject': subj_id,
            'Lesion_ID': lesion['lesion_id'],
            'FWHM_mm': fwhm,
            'EARL_SUVmax': lesion['suv_max_earl'],
            'EARL_SUVmean': lesion['suv_mean_earl'],
            'Filt_SUVmax': suv_max_filt,
            'Filt_SUVmean': suv_mean_filt,
            'aRE_SUVmean_%': are_mean,
            'Bias_SUVmean_%': bias_mean
        })
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Recherche du filtre Gaussien optimal pour l'harmonisation EARL par centre.")
    parser.add_argument("--data-dir", type=str, required=True, help="Dossier racine contenant les dossiers des patients.")
    parser.add_argument("--std-filename", type=str, required=True, help="Nom du fichier PET standard (ex: std.nii.gz).")
    parser.add_argument("--earl-filename", type=str, required=True, help="Nom du fichier PET EARL cible (ex: earl.nii.gz).")
    parser.add_argument("--mask-filename", type=str, required=True, help="Nom du fichier masque des lésions (ex: lesion.nii.gz).")
    parser.add_argument("--num-subjects", type=int, required=True, help="Nombre de patients à échantillonner.")
    parser.add_argument("--min-fwhm", type=float, default=0.0, help="FWHM minimum en mm (défaut: 0.0).")
    parser.add_argument("--max-fwhm", type=float, default=10.0, help="FWHM maximum en mm (défaut: 10.0).")
    parser.add_argument("--seed", type=int, default=101, help="Seed pour la reproductibilité du tirage aléatoire (défaut: 101).")
    parser.add_argument("--output-csv", type=str, default="optimal_fwhm_results.csv", help="Chemin du CSV de sortie.")
    parser.add_argument("--num-workers", type=int, default=16, help="Nombre de threads pour le traitement parallèle (défaut: 4).")
    args = parser.parse_args()

    random.seed(args.seed)

    logging.info("Scan du répertoire pour trouver les patients valides...")
    valid_subjects = []
    for subj_folder in os.listdir(args.data_dir):
        subj_path = os.path.join(args.data_dir, subj_folder)
        if os.path.isdir(subj_path):
            if (os.path.exists(os.path.join(subj_path, args.std_filename)) and 
                os.path.exists(os.path.join(subj_path, args.earl_filename)) and 
                os.path.exists(os.path.join(subj_path, args.mask_filename))):
                valid_subjects.append(subj_folder)
                
    if len(valid_subjects) == 0:
        logging.error("Aucun patient valide trouvé avec les fichiers demandés.")
        return
        
    if len(valid_subjects) < args.num_subjects:
        logging.warning(f"Seulement {len(valid_subjects)} patients trouvés sur les {args.num_subjects} demandés. Utilisation du maximum disponible.")
        sampled_subjects = valid_subjects
    else:
        sampled_subjects = random.sample(valid_subjects, args.num_subjects)
        
    logging.info(f"{len(sampled_subjects)} patients sélectionnés pour l'étude.")

    logging.info("Pré-chargement en mémoire RAM et alignement des masques...")
    for subj in tqdm(sampled_subjects, desc="Pré-calculs"):
        subj_dir = os.path.join(args.data_dir, subj)
        MEMORY_POOL[subj] = precompute_subject_data(subj_dir, subj, args)

    fwhm_values = np.arange(args.min_fwhm, args.max_fwhm + 0.5, 0.5)
    tasks = [(subj, fwhm) for subj in sampled_subjects for fwhm in fwhm_values]
    
    logging.info(f"Début de l'optimisation multithreadée : {len(tasks)} tâches à exécuter (FWHM de {args.min_fwhm} à {args.max_fwhm} mm).")
    
    all_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_fwhm_task, task): task for task in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Filtrage & Evaluation"):
            all_results.extend(future.result())
            
    df = pd.DataFrame(all_results)
    df.to_csv(args.output_csv, index=False)
    
    summary_df = df.groupby('FWHM_mm')['aRE_SUVmean_%'].mean().reset_index()
    best_row = summary_df.loc[summary_df['aRE_SUVmean_%'].idxmin()]
    
    logging.info(f"=== RESULTATS DE L'OPTIMISATION ===")
    logging.info(f"FWHM optimal trouvé : {best_row['FWHM_mm']} mm")
    logging.info(f"aRE médiane minimale sur le SUVmean (VOI50) : {best_row['aRE_SUVmean_%']:.2f} %")
    logging.info(f"Résultats détaillés sauvegardés dans : {args.output_csv}")

if __name__ == "__main__":
    main()