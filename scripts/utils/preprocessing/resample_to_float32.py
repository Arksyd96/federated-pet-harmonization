import os
import argparse
import logging
import SimpleITK as sitk
import numpy as np
import concurrent.futures
from tqdm import tqdm
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def process_nifti_to_float32(file_path: Path):
    img = sitk.ReadImage(str(file_path))

    arr = sitk.GetArrayFromImage(img)
    arr_float32 = arr.astype(np.float32)

    arr_float32[arr_float32 < 1e-4] = 0.0

    new_img = sitk.GetImageFromArray(arr_float32)

    new_img.CopyInformation(img)
    
    for key in img.GetMetaDataKeys():
        new_img.SetMetaData(key, img.GetMetaData(key))

    sitk.WriteImage(new_img, str(file_path))
    
    return (file_path.name, "Converti en Float32")


# =====================================================================
# BLOC D'EXÉCUTION PRINCIPAL (Niveau 0)
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convertit les NIfTI ciblés d'un dossier en Float32 (en parallèle).")
    parser.add_argument("directory", type=str, help="Dossier racine contenant les fichiers .nii ou .nii.gz")
    parser.add_argument("--workers", type=int, default=os.cpu_count() // 2, 
                        help="Nombre de processus parallèles (défaut: moitié des CPU)")
    args = parser.parse_args()

    root_dir = Path(args.directory)
    if not root_dir.exists() or not root_dir.is_dir():
        logging.error(f"Le dossier {root_dir} n'existe pas ou n'est pas un répertoire.")
        exit( 1 ) 

    logging.info(f"Recherche des fichiers NIfTI dans {root_dir}...")

    # On liste tous les fichiers NIfTI en brut
    all_niftis = list(root_dir.rglob("*.nii")) + list(root_dir.rglob("*.nii.gz"))
    
    allowed_prefixes = ('PET', 'PT', 'Gaussian', 'harmonized', 'predicted', 'EARL')
    
    nifti_files = [
        f for f in all_niftis 
        if f.name.startswith(allowed_prefixes)
    ]
    
    if not nifti_files:
        logging.warning(f"Aucun fichier NIfTI correspondant aux préfixes autorisés n'a été trouvé.")
        exit( 0 ) 

    logging.info(f"{len(nifti_files)} fichiers valides trouvés (CT et autres ignorés). Lancement ({args.workers} workers)...")

    results = {'Converti': 0, 'Erreur': 0}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_nifti_to_float32, path): path for path in nifti_files}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(nifti_files), desc="Traitement"):
            filename, status = future.result()
            
            if "Converti" in status:
                results['Converti'] += 1
            else:
                results['Erreur'] += 1
                logging.error(f"[{filename}] {status}")

    # Bilan
    print("\n" + "="*40)
    print("BILAN DE LA CONVERSION FLOAT32")
    print("="*40)
    print(f"Total traités : {len(nifti_files)}")
    print(f"✅ Convertis   : {results['Converti']}")
    print(f"❌ Erreurs     : {results['Erreur']}")
    print("="*40)