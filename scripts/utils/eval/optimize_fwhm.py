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
import warnings

warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dictionnaire global servant de mémoire tampon partagée entre les threads
MEMORY_POOL = {}

def align_mask_to_reference(mask_img, ref_img):
    """Aligne spatialement le masque sur l'image de référence."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(mask_img)

def precompute_subject_data(subj_dir, subj_id, args):
    """Charge les images, aligne les masques, et pré-calcule les VOI50 pour chaque composante."""
    std_path = os.path.join(subj_dir, args.std_filename)
    earl_path = os.path.join(subj_dir, args.earl_filename)
    
    std_img = sitk.ReadImage(std_path, sitk.sitkFloat32)
    earl_img = sitk.ReadImage(earl_path, sitk.sitkFloat32)
    earl_arr = sitk.GetArrayFromImage(earl_img)
    
    components_data = []
    missing_or_empty_masks = []
    
    for mask_filename in args.mask_filenames:
        mask_path = os.path.join(subj_dir, mask_filename)
        voi_name = mask_filename.split('.')[0]
        
        # 1. Vérification de l'existence du fichier
        if not os.path.exists(mask_path):
            missing_or_empty_masks.append(voi_name)
            continue
            
        mask_img = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        aligned_mask_img = align_mask_to_reference(mask_img, std_img)
        mask_arr = sitk.GetArrayFromImage(aligned_mask_img)
        
        # 2. Vérification si le masque est complètement vide
        if not np.any(mask_arr > 0):
            missing_or_empty_masks.append(voi_name)
            continue
        
        is_lesion = 'lesion' in voi_name.lower()
        
        if is_lesion:
            labeled_mask, num_components = label(mask_arr > 0)
        else:
            labeled_mask = (mask_arr > 0).astype(int)
            num_components = 1
        
        components_added_for_this_mask = 0
        
        for k in range(1, num_components + 1):
            component_mask = (labeled_mask == k)
            suv_max_earl = np.max(earl_arr[component_mask])
            
            if suv_max_earl <= 0:
                continue 
                
            voi50_mask = component_mask & (earl_arr >= 0.5 * suv_max_earl)
            
            if not np.any(voi50_mask):
                continue
                
            suv_mean_earl = np.mean(earl_arr[voi50_mask])
            
            components_data.append({
                'voi_name': voi_name,
                'component_id': k,
                'voi50_mask': voi50_mask,
                'suv_max_earl': suv_max_earl,
                'suv_mean_earl': suv_mean_earl
            })
            components_added_for_this_mask += 1
            
        # Si aucune composante valide (suv max > 0 et voi50 non vide) n'a été extraite
        if components_added_for_this_mask == 0:
            missing_or_empty_masks.append(voi_name)
            
    return {
        'std_img': std_img,
        'components': components_data,
        'missing_or_empty_masks': missing_or_empty_masks
    }

def process_fwhm_task(task_args):
    """Tâche threadée : applique le flou et compare les SUVmean par composante."""
    subj_id, fwhm = task_args
    data = MEMORY_POOL[subj_id]
    std_img = data['std_img']
    components = data['components']
    
    results = []
    
    if fwhm == 0.0:
        arr_blurred = sitk.GetArrayFromImage(std_img)
    else:
        sigma_mm = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian_filter.SetSigma(sigma_mm)
        gaussian_filter.SetNormalizeAcrossScale(False)
        blurred_img = gaussian_filter.Execute(std_img)
        arr_blurred = sitk.GetArrayFromImage(blurred_img)
        
    for comp in components:
        voi50_mask = comp['voi50_mask']
        
        suv_mean_filt = np.mean(arr_blurred[voi50_mask])
        suv_max_filt = np.max(arr_blurred[voi50_mask])
        
        safe_earl_mean = comp['suv_mean_earl'] if comp['suv_mean_earl'] != 0 else 1e-8
        
        are_mean = abs(suv_mean_filt - safe_earl_mean) / safe_earl_mean * 100.0
        bias_mean = ((suv_mean_filt - safe_earl_mean) / safe_earl_mean) * 100.0
        
        results.append({
            'Subject': subj_id,
            'VOI': comp['voi_name'],
            'Component_ID': comp['component_id'], 
            'FWHM_mm': fwhm,
            'EARL_SUVmax': comp['suv_max_earl'],
            'EARL_SUVmean': comp['suv_mean_earl'],
            'Filt_SUVmax': suv_max_filt,
            'Filt_SUVmean': suv_mean_filt,
            'aRE_SUVmean_%': are_mean,
            'Bias_SUVmean_%': bias_mean
        })
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Recherche du filtre Gaussien optimal multi-VOI.")
    parser.add_argument("--data-dir", type=str, required=True, help="Dossier racine contenant les patients.")
    parser.add_argument("--std-filename", type=str, required=True, help="PET standard (ex: std.nii.gz).")
    parser.add_argument("--earl-filename", type=str, required=True, help="PET EARL cible (ex: earl.nii.gz).")
    parser.add_argument("--mask-filenames", type=str, nargs='+', required=True, 
                        help="Liste des fichiers masques (ex: brain.nii.gz liver.nii.gz lesion.nii.gz).")
    
    parser.add_argument("--num-subjects", type=int, default=None, help="Nombre de patients.")
    parser.add_argument("--include-only", type=str, nargs='+', default=None, help="Liste de patients spécifiques à forcer. Ignore --num-subjects et --seed.")
    
    parser.add_argument("--min-fwhm", type=float, default=0.0, help="FWHM minimum en mm.")
    parser.add_argument("--max-fwhm", type=float, default=10.0, help="FWHM maximum en mm.")
    parser.add_argument("--seed", type=int, default=101, help="Seed reproductibilité.")
    parser.add_argument("--output-csv", type=str, default="optimal_fwhm_multivoi_results.csv", help="CSV sortie brute.")
    parser.add_argument("--num-workers", type=int, default=16, help="Nombre de threads.")
    args = parser.parse_args()

    if args.num_subjects is None and args.include_only is None:
        parser.error("Vous devez spécifier soit --num-subjects, soit --include-only.")
        
    if args.include_only and args.num_subjects is not None:
        logging.warning("⚠️ Les arguments --include-only et --num-subjects ont été fournis ensemble. --num-subjects et --seed seront ignorés.")

    random.seed(args.seed)

    logging.info("Scan du répertoire pour valider l'existence des PET std et earl...")
    valid_subjects = []
    
    # La condition est maintenant allégée : il faut juste que std et earl existent
    for subj_folder in os.listdir(args.data_dir):
        subj_path = os.path.join(args.data_dir, subj_folder)
        if os.path.isdir(subj_path):
            if (os.path.exists(os.path.join(subj_path, args.std_filename)) and 
                os.path.exists(os.path.join(subj_path, args.earl_filename))):
                valid_subjects.append(subj_folder)
                
    if len(valid_subjects) == 0:
        logging.error("Aucun patient valide (std + earl) trouvé.")
        return
        
    if args.include_only:
        sampled_subjects = [s for s in args.include_only if s in valid_subjects]
        if len(sampled_subjects) < len(args.include_only):
            logging.warning(f"Seulement {len(sampled_subjects)}/{len(args.include_only)} patients trouvés valides parmi la liste fournie.")
    else:
        # Priorisation : On sépare les patients "complets" (qui ont tous les fichiers masques) des "incomplets"
        preferred = []
        others = []
        for s in valid_subjects:
            has_all = all(os.path.exists(os.path.join(args.data_dir, s, m)) for m in args.mask_filenames)
            if has_all:
                preferred.append(s)
            else:
                others.append(s)
                
        random.shuffle(preferred)
        random.shuffle(others)
        
        # On remplit d'abord avec les complets, puis on complète avec les autres
        pool = preferred + others
        sampled_subjects = pool[:args.num_subjects]
        
    logging.info(f"{len(sampled_subjects)} patients sélectionnés pour le traitement.")

    logging.info("Pré-chargement en mémoire et alignement...")
    mask_issues_counts = {}
    
    for subj in tqdm(sampled_subjects, desc="Pré-calculs"):
        subj_dir = os.path.join(args.data_dir, subj)
        data = precompute_subject_data(subj_dir, subj, args)
        MEMORY_POOL[subj] = data
        
        for missing_voi in data['missing_or_empty_masks']:
            mask_issues_counts[missing_voi] = mask_issues_counts.get(missing_voi, 0) + 1

    # Affichage du bilan des masques
    if mask_issues_counts:
        logging.info("--- Bilan des masques manquants ou vides (ignorés dans le calcul) ---")
        for voi, count in mask_issues_counts.items():
            logging.info(f"  - '{voi}' manquant/vide pour {count} patient(s) sur {len(sampled_subjects)}")
    else:
        logging.info("--- Tous les patients sélectionnés possèdent tous les masques valides ! ---")

    fwhm_values = np.arange(args.min_fwhm, args.max_fwhm + 0.5, 0.5)
    tasks = [(subj, fwhm) for subj in sampled_subjects for fwhm in fwhm_values]
    
    logging.info(f"Début de l'optimisation multithreadée : {len(tasks)} tâches.")
    
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_fwhm_task, task): task for task in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluation"):
            all_results.extend(future.result())
            
    if not all_results:
        logging.error("Aucun résultat généré. Tous les masques de tous les patients étaient probablement vides.")
        return

    # Sauvegarde des résultats bruts
    df = pd.DataFrame(all_results)
    df.to_csv(args.output_csv, index=False)
    
    # Étape A : Moyenne de toutes les composantes au sein de la MÊME VOI pour UN patient
    # Les masques vides sont ignorés naturellement par pandas car ils n'ont pas de ligne dans df
    patient_voi_df = df.groupby(['Subject', 'FWHM_mm', 'VOI'])['aRE_SUVmean_%'].mean().reset_index()
    
    # Étape B : Moyenne globale à travers tous les patients et toutes les VOIs
    summary_df = patient_voi_df.groupby('FWHM_mm')['aRE_SUVmean_%'].mean().reset_index()
    
    best_row = summary_df.loc[summary_df['aRE_SUVmean_%'].idxmin()]
    
    logging.info(f"=== RESULTATS DE L'OPTIMISATION GLOBALE (MULTI-VOI) ===")
    logging.info(f"FWHM optimal trouvé : {best_row['FWHM_mm']} mm")
    logging.info(f"aRE moyenne (agrégée composantes -> VOI -> globale) : {best_row['aRE_SUVmean_%']:.2f} %")

if __name__ == "__main__":
    main()