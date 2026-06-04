import os
import glob
import argparse
import logging
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Configuration du logging
sitk.ProcessObject.SetGlobalWarningDisplay(False)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def ensure_nii_gz(filenames):
    return [f if f.endswith(('.nii', '.nii.gz')) else f + '.nii.gz' for f in filenames]

def parse_file_name(filename):
    if filename.lower().startswith('pet'):
        return 'pet'
    elif 'pseudo-earl1' in filename.lower():
        return 'pseudo-earl1'
    elif 'pseudo-earl2' in filename.lower():
        return 'pseudo-earl2'
    elif 'gaussian-earl1' in filename.lower():
        return 'gaussian-earl1'
    elif 'gaussian-earl2' in filename.lower():
        return 'gaussian-earl2'
    elif 'earl1' in filename.lower():
        return 'earl1'
    elif 'earl2' in filename.lower():
        return 'earl2'
    elif 'earl' in filename.lower():
        return 'earl1'  # Par défaut, si "earl" est présent sans précision, on considère earl1
    else:
        return filename.split('.')[0]

def process_single_subject(args):
    subj_in_dir, subj_out_dir, mask_names, file_names = args
    all_files = os.listdir(subj_in_dir)
    
    found_masks = [m for m in mask_names if m in all_files]
    found_images = [f for f in file_names if f in all_files]
    
    if not found_masks or not found_images:
        return subj_in_dir, "Masques ou images cibles manquants."

    try:
        loaded_images = {}
        for img_name in found_images:
            loaded_images[ img_name ] = sitk.ReadImage(os.path.join(subj_in_dir, img_name))

        # --- EXTRACTION STRICTE DES VOIS ---
        for mask_name in found_masks:
            mask_path = os.path.join(subj_in_dir, mask_name)
            mask_sitk = sitk.ReadImage(mask_path)
            
            label_stats = sitk.LabelShapeStatisticsImageFilter()
            label_stats.Execute(sitk.Cast(mask_sitk, sitk.sitkUInt8))
            
            if not label_stats.HasLabel(1):
                continue
                
            # bbox = (startX, startY, startZ, sizeX, sizeY, sizeZ)
            bbox = label_stats.GetBoundingBox(1)
            
            # Création du dossier de la région
            voi_name = mask_name.split('.')[ 0 ]
            voi_out_dir = os.path.join(subj_out_dir, voi_name)
            os.makedirs(voi_out_dir, exist_ok=True)
            
            roi_filter = sitk.RegionOfInterestImageFilter()
            roi_filter.SetIndex(bbox[0:3])
            roi_filter.SetSize(bbox[3:6])

            mask_cropped = roi_filter.Execute(mask_sitk)
            sitk.WriteImage(mask_cropped, os.path.join(voi_out_dir, "mask.nii.gz"))
            
            for img_name, img_sitk in loaded_images.items():
                img_cropped = roi_filter.Execute(img_sitk)
                out_name = parse_file_name(img_name)
                sitk.WriteImage(img_cropped, os.path.join(voi_out_dir, f'{out_name}.nii.gz'))
                
        return subj_in_dir, "Succès"

    except Exception as e:
        return subj_in_dir, f"Erreur : {str(e)}"


def extract_vois(input_dir, output_dir, mask_names, file_names, num_workers=None):
    if not os.path.exists(input_dir):
        logging.error(f"Le dossier racine d'entrée n'existe pas : {input_dir}")
        return

    mask_names = ensure_nii_gz(mask_names)
    file_names = ensure_nii_gz(file_names)

    search_pattern = os.path.join(input_dir, "**", "*.nii.gz")
    all_niftis = glob.glob(search_pattern, recursive=True)
    subject_dirs = list(set([os.path.dirname(f) for f in all_niftis]))
    
    if not subject_dirs:
        logging.warning("Aucun dossier patient trouvé.")
        return

    tasks = []
    for subj_in_dir in subject_dirs:
        relative_path = os.path.relpath(subj_in_dir, input_dir)
        subj_out_dir = os.path.join(output_dir, relative_path)
        tasks.append((subj_in_dir, subj_out_dir, mask_names, file_names))

    print(f"🚀 Début de l'extraction sur {len(tasks)} dossiers potentiels...")
    
    num_workers = os.cpu_count() // 2 if num_workers is None else num_workers
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_subject, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Extraction VOIs"):
            subj_path, status = future.result()
            if status != "Succès" and "manquants" not in status:
                logging.warning(f"Problème avec {os.path.basename(subj_path)} : {status}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction stricte des VOIs depuis les volumes TEP complets.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Dossier racine contenant les domaines/patients.")
    parser.add_argument("--output", "-o", type=str, required=True, help="Dossier racine de sortie (BIDS like).")
    parser.add_argument("--filenames", "-f", type=str, nargs='+', required=True, 
                        help="Liste des fichiers images à cropper (ex: pet earl pseudo-earl).")
    parser.add_argument("--masks", "-m", type=str, nargs='+', required=True, 
                        help="Liste des noms de fichiers masques (ex: liver spleen lung).")
    parser.add_argument("--num-workers", "-n", type=int, default=None, help="Nombre de workers.")
    
    args = parser.parse_args()
    
    extract_vois(
        input_dir=args.input,
        output_dir=args.output,
        mask_names=args.masks,
        file_names=args.filenames,
        num_workers=args.num_workers
    )