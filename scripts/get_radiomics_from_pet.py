import os
import argparse
import logging
import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from scipy.ndimage import distance_transform_edt

# Configuration du logging par défaut
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extraction de Radiomics (PyRadiomics) sur images PET (Standard/EARL) avec masques.")
    parser.add_argument("--root", "-r", type=str, required=True, help="Root folder with subjects.")
    parser.add_argument("--mask", "-m", type=str, required=True, help="Mask filename on the same subject folder (e.g., 'MASK.nii.gz').")
    parser.add_argument("--use-sphere", "-s", action="store_true", help="Use spherical masks centered on the lesion's centroid.")
    parser.add_argument("--params", "-p", type=str, default=None, help="YAML pyradiomics params file.")
    parser.add_argument("--verbose", "-v",action="store_true")
    return parser.parse_args()

def get_largest_inscribed_sphere_mask(sitk_mask, shrinkage_mm=0.0):
    mask_arr = sitk.GetArrayFromImage(sitk_mask)
    spacing = sitk_mask.GetSpacing() # (sx, sy, sz)
    
    # On inverse le spacing pour matcher l'ordre de numpy (sz, sy, sx)
    spacing_numpy = spacing[::-1]

    if mask_arr.sum() == 0:
        return sitk_mask

    edt_map = distance_transform_edt(mask_arr, sampling=spacing_numpy)
    max_radius = edt_map.max()
    final_radius = max(0, max_radius - shrinkage_mm)
    
    if final_radius == 0:
        sphere_arr = np.zeros_like(mask_arr, dtype=np.uint8)
    else:
        center_idx = np.unravel_index(edt_map.argmax(), edt_map.shape)
        cz, cy, cx = center_idx
        zz, yy, xx = np.ogrid[:mask_arr.shape[0], :mask_arr.shape[1], :mask_arr.shape[2]]
        
        dist2 = (
            ((zz - cz) * spacing_numpy[0])**2 + 
            ((yy - cy) * spacing_numpy[1])**2 + 
            ((xx - cx) * spacing_numpy[2])**2
        )
        
        sphere_arr = (dist2 <= final_radius**2).astype(np.uint8)

    # 5. Reconversion en image SimpleITK
    sphere_mask_sitk = sitk.GetImageFromArray(sphere_arr)
    sphere_mask_sitk.CopyInformation(sitk_mask) # Copie origine, spacing, direction

    return sphere_mask_sitk

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

def process_subjects(root_dir, mask_filename, extractor, use_sphere=False):
    if not os.path.exists(root_dir):
        logging.error(f"Le dossier racine n'existe pas : {root_dir}")
        return
    
    if os.path.isdir(mask_filename):
        logging.error(f"Le nom du masque fourni est un dossier, pas un fichier : {mask_filename}")
        return 
    else:
        if not mask_filename.endswith(('.nii', '.nii.gz')):
            mask_filename += '.nii.gz'
        logging.info(f"Utilisation du masque : {mask_filename}")    

    # Lister les dossiers sujets
    subjects = [s for s in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, s))]
    subjects.sort()
    
    logging.info(f"Début du traitement sur {len(subjects)} sujets. Mode Sphère : {use_sphere}")

    for idx, subject_id in enumerate(subjects): # DEBUG
        logging.info(f"Traitement du sujet : {subject_id}; {idx + 1} / {len(subjects)}")

        results = list()
        subject_path = os.path.join(root_dir, subject_id)
        mask_path = os.path.join(subject_path, mask_filename)
        
        # 1. Vérification du masque
        if not os.path.exists(mask_path):
            logging.warning(f"[{subject_id}] Masque introuvable ({mask_filename}). Sujet ignoré.")
            continue

        current_mask_input = None
        if use_sphere:
            original_mask_sitk = sitk.ReadImage(mask_path)
            sphere_mask_sitk = get_largest_inscribed_sphere_mask(original_mask_sitk)
            current_mask_input = sphere_mask_sitk
            logging.info(f"[{subject_id}] Masque sphérique généré.")
        else:
            current_mask_input = mask_path
            
        # 2. Recherche des fichiers PET (PT*)
        files = os.listdir(subject_path)
        pet_files = [
            f for f in files 
            if f.startswith('PT') and (f.endswith('.nii') or f.endswith('.nii.gz'))
            and 'MIP' not in f  # Exclure les MIP
        ]

        if not pet_files:
            logging.warning(f"[{subject_id}] Aucun fichier PET (PT*) trouvé.")
            continue

        # 3. Extraction pour chaque image PET trouvée
        for pet_file in pet_files:
            pet_path = os.path.join(subject_path, pet_file)
            
            # Si "EARL" est dans le nom ET que ça commence par PT (déjà filtré), c'est un EARL
            modality = "PET_EARL" if "EARL" in pet_file else "PET"
            
            logging.info(f"[{subject_id}] Extraction... Type: {modality} | Fichier: {pet_file}")

            try:
                # Extraction
                feature_vector = extractor.execute(pet_path, current_mask_input)
                row = {k: v for k, v in feature_vector.items()} 
                
                # Ajout des colonnes d'identification
                row['Subject_ID'] = subject_id
                row['Modality'] = modality
                row['Image_Filename'] = pet_file
                row['Mask_Filename'] = mask_filename
                row['ROI_type'] = 'Sphere' if use_sphere else 'Original'
                
                results.append(row)

            except Exception as e:
                logging.error(f"[{subject_id}] Échec extraction sur {pet_file} : {e}")

        # saveugarde du csv dans le répértoire racine après chaque sujet
        if results.__len__() > 0:
            df = pd.DataFrame(results)
            
            first_cols = ['Subject_ID', 'Modality', 'Image_Filename', 'Mask_Filename']
            remaining_cols = [c for c in df.columns if c not in first_cols]
            df = df[first_cols + remaining_cols]
            
            output_file = os.path.join(subject_path, mask_filename.split('.')[0] + '_radiomics.csv')
            df.to_csv(output_file, index=False)

        else:
            logging.info(f"[{subject_id}] Aucun résultat d'extraction à sauvegarder.")


if __name__ == "__main__":
    args = parse_arguments()

    if args.verbose:    logging.getLogger('radiomics').setLevel(logging.INFO)
    else:               logging.getLogger('radiomics').setLevel(logging.WARNING)

    extractor = get_extractor(args.params)
    process_subjects(args.root, args.mask, extractor, use_sphere=args.use_sphere)