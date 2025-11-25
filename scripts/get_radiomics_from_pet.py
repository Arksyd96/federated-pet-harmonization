import os
import argparse
import logging
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# Configuration du logging par défaut
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extraction de Radiomics (PyRadiomics) sur images PET (Standard/EARL) avec masques."
    )
    
    parser.add_argument(
        "--root", "-r", 
        type=str, 
        required=True, 
        help="Chemin vers le dossier racine (Root folder) contenant les dossiers sujets."
    )
    
    parser.add_argument(
        "--mask", "-m", 
        type=str, 
        required=True, 
        help="Nom exact du fichier de masque à chercher dans chaque dossier sujet (ex: mask_liver.nii.gz)."
    )
    
    parser.add_argument(
        "--params", "-p", 
        type=str, 
        default=None, 
        help="(Optionnel) Chemin vers un fichier YAML de configuration pyradiomics. Recommandé pour la reproductibilité."
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Active le mode verbeux (DEBUG) pour pyradiomics."
    )

    return parser.parse_args()

def get_extractor(param_file=None):
    if param_file and os.path.isfile(param_file):
        logger.info(f"Chargement de la configuration depuis : {param_file}")
        extractor = featureextractor.RadiomicsFeatureExtractor(param_file)
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllFeatures()
        extractor.disableFeatureByName('shape2D')
        
        extractor.settings['correctMask'] = True 
        extractor.settings['geometryTolerance'] = 1e-5
        extractor.settings['binWidth'] = 0.25

    return extractor

def process_subjects(root_dir, mask_filename, extractor):
    if not os.path.exists(root_dir):
        logger.error(f"Le dossier racine n'existe pas : {root_dir}")
        return
    
    if os.path.isdir(mask_filename):
        logger.error(f"Le nom du masque fourni est un dossier, pas un fichier : {mask_filename}")
    else:
        if not mask_filename.endswith(('.nii', '.nii.gz')):
            mask_filename += '.nii.gz'
        logger.info(f"Utilisation du masque : {mask_filename}")    

    # Lister les dossiers sujets
    subjects = [s for s in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, s))]
    subjects.sort()
    
    logger.info(f"Début du traitement sur {len(subjects)} sujets trouvés dans {root_dir}")

    for idx, subject_id in enumerate(subjects): # DEBUG
        logger.info(f"Traitement du sujet : {subject_id}; {idx} / {len(subjects)}")

        results = list()

        subject_path = os.path.join(root_dir, subject_id)
        mask_path = os.path.join(subject_path, mask_filename)
        
        # 1. Vérification du masque
        if not os.path.exists(mask_path):
            logger.warning(f"[{subject_id}] Masque introuvable ({mask_filename}). Sujet ignoré.")
            continue
            
        # 2. Recherche des fichiers PET (PT*)
        files = os.listdir(subject_path)
        pet_files = [
            f for f in files 
            if f.startswith('PT') and (f.endswith('.nii') or f.endswith('.nii.gz'))
            and 'MIP' not in f  # Exclure les MIP
        ]

        if not pet_files:
            logger.warning(f"[{subject_id}] Aucun fichier PET (PT*) trouvé.")
            continue

        # 3. Extraction pour chaque image PET trouvée
        for pet_file in pet_files:
            pet_path = os.path.join(subject_path, pet_file)
            
            # Si "EARL" est dans le nom ET que ça commence par PT (déjà filtré), c'est un EARL
            modality = "PET_EARL" if "EARL" in pet_file else "PET"
            
            logger.info(f"[{subject_id}] Extraction... Type: {modality} | Fichier: {pet_file}")

            try:
                # Extraction
                feature_vector = extractor.execute(pet_path, mask_path)
                row = {k: v for k, v in feature_vector.items()} 
                
                # Ajout des colonnes d'identification
                row['Subject_ID'] = subject_id
                row['Modality'] = modality
                row['Image_Filename'] = pet_file
                row['Mask_Filename'] = mask_filename
                
                results.append(row)

            except Exception as e:
                logger.error(f"[{subject_id}] Échec extraction sur {pet_file} : {e}")

        # saveugarde du csv dans le répértoire racine après chaque sujet
        if results.__len__() > 0:
            df = pd.DataFrame(results)
            
            first_cols = ['Subject_ID', 'Modality', 'Image_Filename', 'Mask_Filename']
            remaining_cols = [c for c in df.columns if c not in first_cols]
            df = df[first_cols + remaining_cols]
            
            output_file = os.path.join(subject_path, mask_filename.split('.')[0] + '_radiomics.csv')
            df.to_csv(output_file, index=False)

        else:
            logger.info(f"[{subject_id}] Aucun résultat d'extraction à sauvegarder.")


if __name__ == "__main__":
    args = parse_arguments()

    if args.verbose:    logging.getLogger('radiomics').setLevel(logging.DEBUG)
    else:               logging.getLogger('radiomics').setLevel(logging.WARNING)

    extractor = get_extractor(args.params)
    process_subjects(args.root, args.mask, extractor)