import os
import glob
import argparse
import logging
import pandas as pd
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

import SimpleITK as sitk
from radiomics import featureextractor
from scipy.ndimage import distance_transform_edt, label, generate_binary_structure
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def generate_centered_sphere(sitk_mask, radius_mm=15.0, use_barycenter=False, margin_mm=1.0, shift_mm=(0.0, 0.0, 0.0)):
    spacing = sitk_mask.GetSpacing() 
    origin = sitk_mask.GetOrigin()
    size = sitk_mask.GetSize()
    
    stats = sitk.LabelShapeStatisticsImageFilter()
    sitk_mask = sitk.Cast(sitk_mask, sitk.sitkUInt8)

    stats.Execute(sitk_mask)
    
    if not stats.HasLabel(1):
        return None, "Masque vide."
        
    if use_barycenter:
        c_phys = list(stats.GetCentroid(1))
    else:
        mask_arr = sitk.GetArrayFromImage(sitk_mask) 
        edt_map = distance_transform_edt(mask_arr, sampling=spacing[::-1])

        max_idx_flat = edt_map.argmax()
        cz, cy, cx = np.unravel_index(max_idx_flat, mask_arr.shape)
        
        c_phys = list(sitk_mask.TransformIndexToPhysicalPoint((int(cx), int(cy), int(cz))))
    
    c_phys[0] += shift_mm[0]
    c_phys[1] += shift_mm[1]
    c_phys[2] += shift_mm[2]
    
    c_idx = sitk_mask.TransformPhysicalPointToContinuousIndex(c_phys)
    
    nz, ny, nx = size[2], size[1], size[0]
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    
    dist2 = (
        ((xx - c_idx[0]) * spacing[0])**2 + 
        ((yy - c_idx[1]) * spacing[1])**2 + 
        ((zz - c_idx[2]) * spacing[2])**2
    )
    
    sphere_arr = (dist2 <= radius_mm**2)
    safety_radius_sq = (radius_mm + margin_mm)**2
    safety_arr = (dist2 <= safety_radius_sq)

    if not np.any(sphere_arr):
        return None, "La sphère générée est vide."

    mask_arr = sitk.GetArrayFromImage(sitk_mask).astype(bool)

    if not np.all(mask_arr[safety_arr]):
        return None, "La sphère avec marge dépasse les limites du masque."

    out_sitk = sitk.GetImageFromArray(sphere_arr.astype(np.uint8))
    out_sitk.CopyInformation(sitk_mask)
    
    return out_sitk, "Sphère générée avec succès."


def get_extractor(param_file=None):
    if param_file and os.path.isfile(param_file):
        extractor = featureextractor.RadiomicsFeatureExtractor(param_file)
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllFeatures()
        extractor.disableFeatureByName('shape2D')
        extractor.settings['correctMask'] = True 
        extractor.settings['geometryTolerance'] = 1e-5
        extractor.settings['binWidth'] = 0.25
    return extractor

def determine_modality(filename):
    name_lower = filename.lower()
    
    if "pet" in name_lower or "standard" in name_lower:
        return "standard"
    elif "vae" in name_lower or "stargan" in name_lower or "unet" in name_lower:
        return "harmonized" 
    elif "gaussian-earl1" in name_lower:
        return "gaussian-earl1"
    elif "gaussian-earl2" in name_lower:
        return "gaussian-earl2"
    elif "pseudo-earl1" in name_lower:
        return "pseudo-earl1"
    elif "pseudo-earl2" in name_lower:
        return "pseudo-earl2"
    elif "earl1" in name_lower:
        return "earl1"
    elif "earl2" in name_lower:
        return "earl2"
    
    return "unknown"

def process_single_folder(args):
    folder_path, mask_filename, params_file, use_sphere, sphere_radius = args
    
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    logging.getLogger('radiomics').setLevel(logging.ERROR)
    
    voi_name = os.path.basename(folder_path)
    subject_id = os.path.basename(os.path.dirname(folder_path))
    
    mask_path = os.path.join(folder_path, mask_filename)
    if not os.path.exists(mask_path):
        return

    extractor = get_extractor(params_file)
    results = []
    
    search_pattern = os.path.join(folder_path, "*.nii.gz")
    nifti_files = [f for f in glob.glob(search_pattern) if os.path.basename(f) != mask_filename]

    if not nifti_files:
        return

    is_lesion = voi_name.lower() in ["lesion", "lesions"]

    # --- TRAITEMENT SPÉCIFIQUE LÉSIONS ---
    if is_lesion:
        original_mask_sitk = sitk.ReadImage(mask_path)
        mask_arr = sitk.GetArrayFromImage(original_mask_sitk) > 0
        
        struct_3d = generate_binary_structure(3, 3)
        labeled_arr, num_features = label(mask_arr, structure=struct_3d)
        
        if num_features == 0:
            return
            
        # 💡 NOUVEAUTÉ : Comptage des voxels pour filtrer le bruit
        unique_labels, counts = np.unique(labeled_arr, return_counts=True)
        size_dict = dict(zip(unique_labels, counts))
        
        MIN_VOXELS = 10 # Seuil : On ignore toute lésion de moins de 10 voxels
        valid_comp_count = 0
        
        # On crée UNE SEULE image multi-labels (en uint16 au cas où num_features > 255)
        labeled_sitk = sitk.GetImageFromArray(labeled_arr.astype(np.uint16))
        labeled_sitk.CopyInformation(original_mask_sitk)
            
        for comp_idx in range(1, num_features + 1):
            # Filtrage du bruit
            if size_dict.get(comp_idx, 0) < MIN_VOXELS:
                continue
                
            valid_comp_count += 1
            
            for pet_file_path in nifti_files:
                filename = os.path.basename(pet_file_path)
                modality = determine_modality(filename)
                
                try:
                    # 💡 OPTIMISATION : On utilise "label=comp_idx" natif à PyRadiomics
                    feature_vector = extractor.execute(pet_file_path, labeled_sitk, label=comp_idx)
                    row = {k: v.item() if hasattr(v, 'item') else v for k, v in feature_vector.items() if k.startswith('original_')}
                    row.update({
                        'Subject_ID': subject_id,
                        'VOI': voi_name,
                        'Component': comp_idx,      
                        'Modality': modality,
                        'Image_Filename': filename,
                        'ROI_type': 'Component'
                    })
                    results.append(row)
                except Exception:
                    pass 
                    
        del labeled_arr, labeled_sitk
        gc.collect()

    # --- TRAITEMENT CLASSIQUE ---
    else:
        current_mask_input = mask_path
        if use_sphere:
            original_mask_sitk = sitk.ReadImage(mask_path)
            sphere_mask_sitk, message = generate_centered_sphere(
                original_mask_sitk, 
                radius_mm=sphere_radius, 
                use_barycenter=False, 
                margin_mm=1.0
            )
            if sphere_mask_sitk is None:
                return 
            current_mask_input = sphere_mask_sitk

        for pet_file_path in nifti_files:
            filename = os.path.basename(pet_file_path)
            modality = determine_modality(filename)
            
            try:
                feature_vector = extractor.execute(pet_file_path, current_mask_input)
                row = {k: v.item() if hasattr(v, 'item') else v for k, v in feature_vector.items() if k.startswith('original_')}
                row.update({
                    'Subject_ID': subject_id,
                    'VOI': voi_name,
                    'Component': None,         
                    'Modality': modality,
                    'Image_Filename': filename,
                    'ROI_type': f'Sphere_{sphere_radius}mm' if use_sphere else 'Original'
                })
                results.append(row)
            except Exception:
                pass
                
        gc.collect()

    # --- SAUVEGARDE & ASSERTION ---
    if results:
        df = pd.DataFrame(results)
        
        first_cols = ['Subject_ID', 'VOI', 'Component', 'Modality', 'Image_Filename', 'ROI_type']
        remaining_cols = [c for c in df.columns if c not in first_cols]
        df = df[first_cols + remaining_cols]
        
        suffix = '_sphere_radiomics.csv' if (use_sphere and not is_lesion) else '_radiomics.csv'
        out_name = mask_filename.split('.')[0] + suffix
        out_file = os.path.join(folder_path, out_name)
        
        # Vérification par assertion stricte
        if is_lesion and valid_comp_count > 0:
            modalities_present = df['Modality'].unique()
            for mod in modalities_present:
                n_rows = len(df[df['Modality'] == mod])
                assert n_rows == valid_comp_count, (
                    f"ASSERTION ERROR | Patient: {subject_id} | VOI: {voi_name} | "
                    f"Modalité: {mod} | Attendu: {valid_comp_count} composantes (valides > 10 voxels), Obtenu: {n_rows} lignes. "
                    "PyRadiomics a ignoré une lésion."
                )

        df.to_csv(out_file, index=False)
    
    
def process_subjects(root_dir, mask_filename="mask_cropped.nii.gz", params_file=None, use_sphere=False, sphere_radius=20.0, include_only=None, vois=None, num_workers=None):
    if not os.path.exists(root_dir):
        logging.error(f"Le dossier racine n'existe pas : {root_dir}")
        return
    
    if not mask_filename.endswith(('.nii', '.nii.gz')):
        mask_filename += '.nii.gz'

    logging.info(f"Recherche des masques '{mask_filename}' dans {root_dir}...")
    
    search_pattern = os.path.join(root_dir, "**", mask_filename)
    all_masks = glob.glob(search_pattern, recursive=True)
    
    target_folders = []
    
    for mask_path in all_masks:
        folder_path = os.path.dirname(mask_path)
        voi_name = os.path.basename(folder_path)
        subject_id = os.path.basename(os.path.dirname(folder_path))
        
        if voi_name == "whole_body":
            continue

        if vois and voi_name not in vois:
            continue

        if include_only and subject_id not in include_only:
            continue
            
        target_folders.append(folder_path)

    if not target_folders:
        logging.warning("Aucun dossier cible valide n'a été trouvé.")
        return

    num_workers = max(1, multiprocessing.cpu_count() - 2) if num_workers is None else num_workers
    logging.info(f"Traitement de {len(target_folders)} dossiers VOI avec {num_workers} workers 🚀")

    tasks = [(folder, mask_filename, params_file, use_sphere, sphere_radius) for folder in target_folders]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_folder, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Extraction Radiomics"):
            # L'exception sera levée ici si l'assertion échoue dans un worker
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction locale de Radiomics par dossier patient/VOI.")
    parser.add_argument("--root", "-r", type=str, required=True, help="Dossier racine (ex: outputs/harmonization/)")
    parser.add_argument("--include-only", "-i", type=str, nargs='*', default=None, help="Liste d'IDs à inclure (défaut: tous).")
    parser.add_argument("--vois", "-v", type=str, nargs='*', default=None, help="Liste des dossiers VOI à traiter.")
    parser.add_argument("--mask", "-m", type=str, default="mask.nii.gz", help="Nom du masque (défaut: mask.nii.gz).")
    parser.add_argument("--use-sphere", "-s", action="store_true", help="Utiliser un masque sphérique centré (ignoré pour 'lesion').")
    parser.add_argument("--sphere-radius", type=float, default=20.0, help="Rayon de la sphère en mm (défaut: 20.0).")
    parser.add_argument("--params", "-p", type=str, default=None, help="YAML pyradiomics params file.")
    parser.add_argument("--num-workers", "-n", type=int, default=None, help="Nombre de workers.")
    parser.add_argument("--debug-radiomics", "-db", action="store_true")
    args = parser.parse_args()

    if args.debug_radiomics:
        logging.getLogger('radiomics').setLevel(logging.INFO)
        logging.getLogger('pykwalify').setLevel(logging.INFO)
    else:
        logging.getLogger('radiomics').setLevel(logging.ERROR)
        logging.getLogger('pykwalify').setLevel(logging.ERROR)
    
    process_subjects(
        args.root, 
        args.mask, 
        args.params, 
        use_sphere=args.use_sphere,
        sphere_radius=args.sphere_radius,
        include_only=args.include_only,
        vois=args.vois,
        num_workers=args.num_workers
    )


# import os
# import glob
# import argparse
# import logging
# import pandas as pd
# import numpy as np

# from concurrent.futures import ProcessPoolExecutor, as_completed
# import multiprocessing
# from tqdm import tqdm

# import SimpleITK as sitk
# from radiomics import featureextractor
# from scipy.ndimage import distance_transform_edt
# import gc

# # Configuration du logging par défaut
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# def generate_centered_sphere(sitk_mask, radius_mm=15.0, use_barycenter=False, margin_mm=1.0, shift_mm=(0.0, 0.0, 0.0)):
#     spacing = sitk_mask.GetSpacing() # (sx, sy, sz)
#     origin = sitk_mask.GetOrigin()
#     size = sitk_mask.GetSize()
    
#     # 2. Calcul de l'origine + Shift
#     stats = sitk.LabelShapeStatisticsImageFilter()
#     sitk_mask = sitk.Cast(sitk_mask, sitk.sitkUInt8)

#     stats.Execute(sitk_mask)
    
#     if not stats.HasLabel(1):
#         return None
        
#     if use_barycenter:
#         c_phys = list(stats.GetCentroid(1))
#     else:
#         mask_arr = sitk.GetArrayFromImage(sitk_mask) # (z, y, x)
#         edt_map = distance_transform_edt(mask_arr, sampling=spacing[::-1])

#         max_idx_flat = edt_map.argmax()
#         cz, cy, cx = np.unravel_index(max_idx_flat, mask_arr.shape)
        
#         c_phys = list(sitk_mask.TransformIndexToPhysicalPoint((int(cx), int(cy), int(cz))))
    
#     c_phys[0] += shift_mm[0]
#     c_phys[1] += shift_mm[1]
#     c_phys[2] += shift_mm[2]
    
#     c_idx = sitk_mask.TransformPhysicalPointToContinuousIndex(c_phys)
    
#     # 4. Création de la grille vectorisée (Numpy style)
#     # Attention : SimpleITK est (x, y, z), Numpy est (z, y, x)
#     # On crée une grille d'indices qui couvre toute l'image
#     nz, ny, nx = size[2], size[1], size[0]
#     zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    
#     # 5. Calcul de la distance physique vectorisé
#     # Formule : dist² = ((x - cx)*sx)² + ...
#     # C'est instantané avec numpy
#     dist2 = (
#         ((xx - c_idx[0]) * spacing[0])**2 + 
#         ((yy - c_idx[1]) * spacing[1])**2 + 
#         ((zz - c_idx[2]) * spacing[2])**2
#     )
    
#     sphere_arr = (dist2 <= radius_mm**2)

#     # Vérification d'inclusion avec marge
#     safety_radius_sq = (radius_mm + margin_mm)**2
#     safety_arr = (dist2 <= safety_radius_sq)

#     if not np.any(sphere_arr):
#         return None, "La sphère générée est vide. Vérifiez les paramètres de rayon et de marge."

#     mask_arr = sitk.GetArrayFromImage(sitk_mask).astype(bool)

#     if not np.all(mask_arr[safety_arr]):
#         return None, "La sphère avec marge dépasse les limites du masque. Augmentez la marge ou réduisez le rayon."

#     # 8. Conversion et retour
#     out_sitk = sitk.GetImageFromArray(sphere_arr.astype(np.uint8))
#     out_sitk.CopyInformation(sitk_mask)
    
#     return out_sitk, "Sphère générée avec succès."


# def get_extractor(param_file=None):
#     if param_file and os.path.isfile(param_file):
#         extractor = featureextractor.RadiomicsFeatureExtractor(param_file)
#     else:
#         extractor = featureextractor.RadiomicsFeatureExtractor()
#         extractor.enableAllFeatures()
#         extractor.disableFeatureByName('shape2D')
#         extractor.settings['correctMask'] = True 
#         extractor.settings['geometryTolerance'] = 1e-5
#         extractor.settings['binWidth'] = 0.25
#     return extractor

# def determine_modality(filename):
#     name_lower = filename.lower()
    
#     if "pet" in name_lower or "standard" in name_lower:
#         return "standard"
#     elif "vae" in name_lower or "stargan" in name_lower or "unet" in name_lower:
#         return "harmonized" 
#     elif "gaussian-earl1" in name_lower:
#         return "gaussian-earl1"
#     elif "gaussian-earl2" in name_lower:
#         return "gaussian-earl2"
#     elif "pseudo-earl1" in name_lower:
#         return "pseudo-earl1"
#     elif "pseudo-earl2" in name_lower:
#         return "pseudo-earl2"
#     elif "earl1" in name_lower:
#         return "earl1"
#     elif "earl2" in name_lower:
#         return "earl2"
    
#     # Fallback pour les cas non prévus
#     return "unknown"

# # =============================================================================

# def process_single_folder(args):
#     folder_path, mask_filename, params_file, use_sphere, sphere_radius = args
    
#     sitk.ProcessObject.SetGlobalWarningDisplay(False)
#     logging.getLogger('radiomics').setLevel(logging.ERROR)
    
#     # Variables de contexte
#     voi_name = os.path.basename(folder_path)
#     subject_id = os.path.basename(os.path.dirname(folder_path))
    
#     mask_path = os.path.join(folder_path, mask_filename)
#     if not os.path.exists(mask_path):
#         return

#     # --- Gestion de la sphère ---
#     current_mask_input = mask_path
#     if use_sphere:
#         original_mask_sitk = sitk.ReadImage(mask_path)
#         sphere_mask_sitk, message = generate_centered_sphere(
#             original_mask_sitk, 
#             radius_mm=sphere_radius, 
#             use_barycenter=False, 
#             margin_mm=1.0
#         )
#         if sphere_mask_sitk is None:
#             # Échec de la sphère (ex: déborde du foie), on passe ce dossier
#             return
#         current_mask_input = sphere_mask_sitk

#     extractor = get_extractor(params_file)
#     results = []
    
#     # Parcours des images avec glob
#     search_pattern = os.path.join(folder_path, "*.nii.gz")
#     nifti_files = glob.glob(search_pattern)

#     for pet_file_path in nifti_files:
#         filename = os.path.basename(pet_file_path)
        
#         # On ignore le masque
#         if filename == mask_filename:
#             continue
            
#         modality = determine_modality(filename)
        
#         try:
#             feature_vector = extractor.execute(pet_file_path, current_mask_input)
            
#             row = {k: v.item() if hasattr(v, 'item') else v for k, v in feature_vector.items() if k.startswith('original_')}
#             row.update({
#                 'Subject_ID': subject_id,
#                 'VOI': voi_name,
#                 'Modality': modality,
#                 'Image_Filename': filename,
#                 'ROI_type': f'Sphere_{sphere_radius}mm' if use_sphere else 'Original'
#             })
#             results.append(row)
            
#         except Exception:
#             pass # Silencieux pour ne pas polluer l'exécution parallèle
            
#         del feature_vector
#         gc.collect()

#     # --- Sauvegarde locale ---
#     if results:
#         df = pd.DataFrame(results)
#         first_cols = ['Subject_ID', 'VOI', 'Modality', 'Image_Filename', 'ROI_type']
#         remaining_cols = [c for c in df.columns if c not in first_cols]
#         df = df[ first_cols + remaining_cols ]
        
#         suffix = '_sphere_radiomics.csv' if use_sphere else '_radiomics.csv'
#         out_name = mask_filename.split('.')[ 0 ] + suffix
#         out_file = os.path.join(folder_path, out_name)
        
#         df.to_csv(out_file, index=False)
    

# def process_subjects(
#     root_dir, mask_filename="mask_cropped.nii.gz", params_file=None, use_sphere=False, sphere_radius=20.0, 
#     include_only=None, vois=None, num_workers=None):
#     if not os.path.exists(root_dir):
#         logging.error(f"Le dossier racine n'existe pas : {root_dir}")
#         return
    
#     if not mask_filename.endswith(('.nii', '.nii.gz')):
#         mask_filename += '.nii.gz'

#     logging.info(f"Recherche des masques '{mask_filename}' dans {root_dir}...")
#     if vois:
#         logging.info(f"Filtre activé pour les VOIs : {', '.join(vois)}")
    
#     # glob récursif pour trouver tous les masques
#     search_pattern = os.path.join(root_dir, "**", mask_filename)
#     all_masks = glob.glob(search_pattern, recursive=True)
    
#     target_folders = []
    
#     for mask_path in all_masks:
#         folder_path = os.path.dirname(mask_path)
#         voi_name = os.path.basename(folder_path)
#         subject_id = os.path.basename(os.path.dirname(folder_path))
        
#         # 1. Ignorer les whole_body systématiquement
#         if voi_name == "whole_body":
#             continue

#         if vois and voi_name not in vois:
#             continue

#         if include_only and subject_id not in include_only:
#             continue
            
#         target_folders.append(folder_path)

#     if not target_folders:
#         logging.warning("Aucun dossier cible valide n'a été trouvé.")
#         return

#     num_workers = max(1, multiprocessing.cpu_count() - 2) if num_workers is None else num_workers
#     logging.info(f"Traitement de {len(target_folders)} dossiers VOI avec {num_workers} workers 🚀")

#     tasks = [(folder, mask_filename, params_file, use_sphere, sphere_radius) for folder in target_folders]

#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         futures = {executor.submit(process_single_folder, task): task for task in tasks}
        
#         for future in tqdm(as_completed(futures), total=len(tasks), desc="Extraction Radiomics"):
#             future.result()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Extraction locale de Radiomics par dossier patient/VOI.")
#     parser.add_argument("--root", "-r", type=str, required=True, help="Dossier racine (ex: outputs/harmonization/)")
#     parser.add_argument("--include-only", "-i", type=str, nargs='*', default=None, help="Liste d'IDs à inclure (défaut: tous).")
#     parser.add_argument("--vois", "-v", type=str, nargs='*', default=None, help="Liste des dossiers VOI à traiter (ex: liver lung).")
#     parser.add_argument("--mask", "-m", type=str, default="mask.nii.gz", help="Nom du masque (défaut: mask.nii.gz).")
#     parser.add_argument("--use-sphere", "-s", action="store_true", help="Utiliser un masque sphérique centré.")
#     parser.add_argument("--sphere-radius", type=float, default=20.0, help="Rayon de la sphère en mm (défaut: 20.0).")
#     parser.add_argument("--params", "-p", type=str, default=None, help="YAML pyradiomics params file.")
#     parser.add_argument("--num-workers", "-n", type=int, default=None, help="Nombre de workers.")
#     parser.add_argument("--debug-radiomics", "-db", action="store_true")
#     args = parser.parse_args()

#     if args.debug_radiomics:
#         logging.getLogger('radiomics').setLevel(logging.INFO)
#         logging.getLogger('pykwalify').setLevel(logging.INFO)
#     else:
#         logging.getLogger('radiomics').setLevel(logging.ERROR)
#         logging.getLogger('pykwalify').setLevel(logging.ERROR)
    
#     process_subjects(
#         args.root, 
#         args.mask, 
#         args.params, 
#         use_sphere=args.use_sphere,
#         sphere_radius=args.sphere_radius,
#         include_only=args.include_only,
#         vois=args.vois,
#         num_workers=args.num_workers
#     )