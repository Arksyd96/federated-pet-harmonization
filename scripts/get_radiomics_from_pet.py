import os
import argparse
import logging
import pandas as pd
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

import SimpleITK as sitk
from radiomics import featureextractor
from scipy.ndimage import distance_transform_edt


# Configuration du logging par défaut
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def generate_centered_sphere(sitk_mask, radius_mm=15.0, use_barycenter=False, margin_mm=1.0, shift_mm=(0.0, 0.0, 0.0)):
    spacing = sitk_mask.GetSpacing() # (sx, sy, sz)
    origin = sitk_mask.GetOrigin()
    size = sitk_mask.GetSize()
    
    # 2. Calcul de l'origine + Shift
    stats = sitk.LabelShapeStatisticsImageFilter()
    sitk_mask = sitk.Cast(sitk_mask, sitk.sitkUInt8)

    stats.Execute(sitk_mask)
    
    if not stats.HasLabel(1):
        return None
        
    if use_barycenter:
        c_phys = list(stats.GetCentroid(1))
    else:
        mask_arr = sitk.GetArrayFromImage(sitk_mask) # (z, y, x)
        edt_map = distance_transform_edt(mask_arr, sampling=spacing[::-1])

        max_idx_flat = edt_map.argmax()
        cz, cy, cx = np.unravel_index(max_idx_flat, mask_arr.shape)
        
        c_phys = list(sitk_mask.TransformIndexToPhysicalPoint((int(cx), int(cy), int(cz))))

    
    c_phys[0] += shift_mm[0]
    c_phys[1] += shift_mm[1]
    c_phys[2] += shift_mm[2]
    
    c_idx = sitk_mask.TransformPhysicalPointToContinuousIndex(c_phys)
    
    # 4. Création de la grille vectorisée (Numpy style)
    # Attention : SimpleITK est (x, y, z), Numpy est (z, y, x)
    # On crée une grille d'indices qui couvre toute l'image
    nz, ny, nx = size[2], size[1], size[0]
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    
    # 5. Calcul de la distance physique vectorisé
    # Formule : dist² = ((x - cx)*sx)² + ...
    # C'est instantané avec numpy
    dist2 = (
        ((xx - c_idx[0]) * spacing[0])**2 + 
        ((yy - c_idx[1]) * spacing[1])**2 + 
        ((zz - c_idx[2]) * spacing[2])**2
    )
    
    sphere_arr = (dist2 <= radius_mm**2)

    # Vérification d'inclusion avec marge
    safety_radius_sq = (radius_mm + margin_mm)**2
    safety_arr = (dist2 <= safety_radius_sq)

    if not np.any(sphere_arr):
        return None

    mask_arr = sitk.GetArrayFromImage(sitk_mask).astype(bool)

    if not np.all(mask_arr[safety_arr]):
        return None

    # 8. Conversion et retour
    out_sitk = sitk.GetImageFromArray(sphere_arr.astype(np.uint8))
    out_sitk.CopyInformation(sitk_mask)
    
    return out_sitk


def get_extractor(param_file=None):
    if param_file and os.path.isfile(param_file):
        logging.info(f"Chargement de la configuration depuis : {param_file}")
        extractor = featureextractor.RadiomicsFeatureExtractor(param_file)
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllFeatures()
        extractor.disableFeatureByName('shape2D')
        
        extractor.settings['correctMask'] = True 
        extractor.settings['geometryTolerance'] = 1e-5
        extractor.settings['binWidth'] = 0.25

    return extractor


def process_single_subject(args):
    subject_id, root_dir, mask_filename, params_file, use_sphere, sphere_radius = args

    logging.getLogger().setLevel(logging.WARNING) 
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    
    results = []
    subject_path = os.path.join(root_dir, subject_id)
    mask_path = os.path.join(subject_path, mask_filename)

    extractor = get_extractor(params_file)

    if not os.path.exists(mask_path):
        logging.warning(f"[{subject_id}] Masque introuvable ({mask_filename}). Sujet ignoré.")
        return

    
    # --- Gestion du Masque (Sphère vs Original) ---
    current_mask_input = None
    
    if use_sphere:
        original_mask_sitk = sitk.ReadImage(mask_path)
        
        sphere_mask_sitk = generate_centered_sphere(
            original_mask_sitk, 
            radius_mm=sphere_radius, 
            use_barycenter=False, 
            margin_mm=1.0
        )
        
        if sphere_mask_sitk is None:
            logging.warning(f"[{subject_id}] Échec génération masque sphérique. Sujet ignoré.")
            return
            
        current_mask_input = sphere_mask_sitk
    else:
        current_mask_input = mask_path

    # --- Extraction PET ---
    pet_files = [
        f for f in os.listdir(subject_path) 
        if (f.startswith('PET') or f.startswith('EARL') or f.startswith('predicted_'))
        and (f.endswith('.nii') or f.endswith('.nii.gz'))
        and 'MIP' not in f
    ]

    if not pet_files:
        logging.warning(f"[{subject_id}] Aucun fichier PET (PET/EARL*) trouvé.")
        return

    for pet_file in pet_files:
        pet_path = os.path.join(subject_path, pet_file)
        modality = "EARL" if "EARL" in pet_file else "PET"

        # Extraction
        feature_vector = extractor.execute(pet_path, current_mask_input)
        
        row = {k: v for k, v in feature_vector.items()}
        row.update({
            'Subject_ID': subject_id,
            'Modality': modality,
            'Image_Filename': pet_file,
            'Mask_Filename': mask_filename,
            'ROI_type': 'Sphere_{}mm'.format(sphere_radius) if use_sphere else 'Original'
        })
        results.append(row)

    # --- Sauvegarde CSV Individuel (Thread-Safe) ---
    if results:
        df = pd.DataFrame(results)
        # Organisation colonnes
        first_cols = ['Subject_ID', 'Modality', 'Image_Filename', 'Mask_Filename', 'ROI_type']
        remaining_cols = [c for c in df.columns if c not in first_cols]
        df = df[first_cols + remaining_cols]
        
        suffix = '_sphere_radiomics.csv' if use_sphere else '_radiomics.csv'
        output_file = os.path.join(subject_path, mask_filename.split('.')[0] + suffix)
        df.to_csv(output_file, index=False)
    else:
        logging.info(f"[{subject_id}] Aucun résultat d'extraction à sauvegarder.")
        return
    

def process_subjects(root_dir, mask_filename, params_file, use_sphere=False, sphere_radius=15.0):
    if not os.path.exists(root_dir):
        logging.error(f"Le dossier racine n'existe pas : {root_dir}")
        return
    
    # Nettoyage nom de fichier
    if os.path.isdir(mask_filename):
        logging.error(f"Le masque est un dossier : {mask_filename}")
        return
    
    if not mask_filename.endswith(('.nii', '.nii.gz')):
        mask_filename += '.nii.gz'

    # Lister les sujets
    subjects = sorted([s for s in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, s))])
    
    # Config Workers (Laisser 2 coeurs libres pour le système)
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    
    logging.info(f"Traitement de {len(subjects)} sujets avec {num_workers} workers 🚀")
    logging.info(f"Mode Sphère : {use_sphere} | Masque : {mask_filename}")

    # Préparation des tâches
    tasks = [
        (subject, root_dir, mask_filename, params_file, use_sphere, sphere_radius) 
        for subject in subjects
    ]

    # Lancement Parallèle
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_subject, task): task for task in tasks}
        
        # Barre de progression
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Extraction Radiomics"):
            try:
                future.result()     
            except Exception as e:
                # On récupère le nom du sujet qui a planté pour le debug
                failed_subject = futures[future][0] 
                logging.error(f"CRASH sur le sujet {failed_subject} : {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction de Radiomics (PyRadiomics) sur images PET (Standard/EARL) avec masques.")
    parser.add_argument("--root", "-r", type=str, required=True, help="Root folder with subjects.")
    parser.add_argument("--mask", "-m", type=str, required=True, help="Mask filename on the same subject folder (e.g., 'MASK.nii.gz').")
    parser.add_argument("--use-sphere", "-s", action="store_true", help="Use spherical masks centered on the lesion's centroid.")
    parser.add_argument("--sphere-radius", type=float, default=20.0, help="Radius of the spherical mask in mm (default: 15.0).")
    parser.add_argument("--params", "-p", type=str, default=None, help="YAML pyradiomics params file.")
    parser.add_argument("--debug-radiomics", "-db",action="store_true")
    args = parser.parse_args()

    if args.debug_radiomics:    logging.getLogger('radiomics').setLevel(logging.INFO)
    else:                       logging.getLogger('radiomics').setLevel(logging.ERROR)

    process_subjects(args.root, args.mask, args.params, use_sphere=args.use_sphere, sphere_radius=args.sphere_radius)