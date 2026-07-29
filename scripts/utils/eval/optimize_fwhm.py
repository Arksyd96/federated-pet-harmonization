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
    """Charge les images, aligne les 6 masques, et pré-calcule les VOI50 pour chaque composante."""
    std_path = os.path.join(subj_dir, args.std_filename)
    earl_path = os.path.join(subj_dir, args.earl_filename)
    
    std_img = sitk.ReadImage(std_path, sitk.sitkFloat32)
    earl_img = sitk.ReadImage(earl_path, sitk.sitkFloat32)
    earl_arr = sitk.GetArrayFromImage(earl_img)
    
    components_data = []
    
    for mask_filename in args.mask_filenames:
        mask_path = os.path.join(subj_dir, mask_filename)
        
        if not os.path.exists(mask_path):
            continue
            
        mask_img = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        aligned_mask_img = align_mask_to_reference(mask_img, std_img)
        mask_arr = sitk.GetArrayFromImage(aligned_mask_img)
        
        voi_name = mask_filename.split('.')[0]
        
        # Traitement asymétrique Organes vs Lésions
        is_lesion = 'lesion' in voi_name.lower()
        
        if is_lesion:
            # Séparation en multiples composantes (Lésion 1, Lésion 2...)
            labeled_mask, num_components = label(mask_arr > 0)
        else:
            # Organe normal (Foie, Cerveau...) = 1 seule composante globale
            labeled_mask = (mask_arr > 0).astype(int)
            num_components = 1 if np.any(mask_arr > 0) else 0
        
        # 2. ITÉRATION SUR LES COMPOSANTES DU MASQUE
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
            
    return {
        'std_img': std_img,
        'components': components_data
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
        
    # 3. ITÉRATION SUR TOUTES LES COMPOSANTES (Foie, Cerveau, Lésion 1, Lésion 2...)
    for comp in components:
        voi50_mask = comp['voi50_mask']
        
        suv_mean_filt = np.mean(arr_blurred[voi50_mask])
        suv_max_filt = np.max(arr_blurred[voi50_mask])
        
        safe_earl_mean = comp['suv_mean_earl'] if comp['suv_mean_earl'] != 0 else 1e-8
        
        # aRE sur la composante
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
    
    # Modifs : --num-subjects n'est plus obligatoire, et ajout de --include-only
    parser.add_argument("--num-subjects", type=int, default=None, help="Nombre de patients.")
    parser.add_argument("--include-only", type=str, nargs='+', default=None, help="Liste de patients spécifiques à forcer. Ignore --num-subjects et --seed.")
    
    parser.add_argument("--min-fwhm", type=float, default=0.0, help="FWHM minimum en mm.")
    parser.add_argument("--max-fwhm", type=float, default=10.0, help="FWHM maximum en mm.")
    parser.add_argument("--seed", type=int, default=101, help="Seed reproductibilité.")
    parser.add_argument("--output-csv", type=str, default="optimal_fwhm_multivoi_results.csv", help="CSV sortie brute.")
    parser.add_argument("--num-workers", type=int, default=16, help="Nombre de threads.")
    args = parser.parse_args()

    # Vérification stricte des paramètres exclusifs
    if args.num_subjects is None and args.include_only is None:
        parser.error("Vous devez spécifier soit --num-subjects, soit --include-only.")
        
    if args.include_only and args.num_subjects is not None:
        logging.warning("⚠️ Les arguments --include-only et --num-subjects ont été fournis ensemble. --num-subjects et --seed seront ignorés.")

    random.seed(args.seed)

    logging.info("Scan du répertoire pour trouver les patients avec lésion...")
    lesion_filename = next((m for m in args.mask_filenames if 'lesion' in m.lower()), None)
    if not lesion_filename:
        logging.error("Aucun fichier 'lesion' trouvé dans --mask-filenames.")
        return

    valid_subjects = []
    for subj_folder in os.listdir(args.data_dir):
        subj_path = os.path.join(args.data_dir, subj_folder)
        if os.path.isdir(subj_path):
            if (os.path.exists(os.path.join(subj_path, args.std_filename)) and 
                os.path.exists(os.path.join(subj_path, args.earl_filename))):
                
                lesion_path = os.path.join(subj_path, lesion_filename)
                if os.path.exists(lesion_path):
                    mask_img = sitk.ReadImage(lesion_path, sitk.sitkUInt8)
                    if np.any(sitk.GetArrayFromImage(mask_img) > 0):
                        valid_subjects.append(subj_folder)
                
    if len(valid_subjects) == 0:
        logging.error("Aucun patient valide trouvé.")
        return
        
    if args.include_only:
        sampled_subjects = [s for s in args.include_only if s in valid_subjects]
        if len(sampled_subjects) < len(args.include_only):
            logging.warning(f"Seulement {len(sampled_subjects)}/{len(args.include_only)} patients trouvés valides parmi la liste fournie.")
    else:
        # Logique aléatoire d'origine
        sampled_subjects = random.sample(valid_subjects, min(args.num_subjects, len(valid_subjects)))
        
    logging.info(f"{len(sampled_subjects)} patients sélectionnés pour le traitement.")

    logging.info("Pré-chargement en mémoire et alignement...")
    for subj in tqdm(sampled_subjects, desc="Pré-calculs"):
        subj_dir = os.path.join(args.data_dir, subj)
        MEMORY_POOL[subj] = precompute_subject_data(subj_dir, subj, args)

    fwhm_values = np.arange(args.min_fwhm, args.max_fwhm + 0.5, 0.5)
    tasks = [(subj, fwhm) for subj in sampled_subjects for fwhm in fwhm_values]
    
    logging.info(f"Début de l'optimisation multithreadée : {len(tasks)} tâches.")
    
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_fwhm_task, task): task for task in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluation"):
            all_results.extend(future.result())
            
    # Sauvegarde des résultats bruts (toutes les composantes séparées)
    df = pd.DataFrame(all_results)
    df.to_csv(args.output_csv, index=False)
    
    # Étape A : Moyenne de toutes les composantes au sein de la MÊME VOI pour UN patient
    # -> Le patient aura exactement 6 lignes par FWHM (Foie, Cerveau, Lésion(Moyenne)...)
    patient_voi_df = df.groupby(['Subject', 'FWHM_mm', 'VOI'])['aRE_SUVmean_%'].mean().reset_index()
    
    # -> 1 seule valeur par FWHM mm
    summary_df = patient_voi_df.groupby('FWHM_mm')['aRE_SUVmean_%'].mean().reset_index()
    
    best_row = summary_df.loc[summary_df['aRE_SUVmean_%'].idxmin()]
    
    logging.info(f"=== RESULTATS DE L'OPTIMISATION GLOBALE (MULTI-VOI) ===")
    logging.info(f"FWHM optimal trouvé : {best_row['FWHM_mm']} mm")
    logging.info(f"aRE moyenne (agrégée composantes -> VOI -> globale) : {best_row['aRE_SUVmean_%']:.2f} %")

if __name__ == "__main__":
    main()