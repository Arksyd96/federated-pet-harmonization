import os
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import concurrent.futures
import multiprocessing

def process_single_subject(args):
    pet_path, threshold = args

    folder_path = os.path.dirname(pet_path)
    output_path = os.path.join(folder_path, "body.nii.gz")

    if os.path.exists(output_path):
        return False

    img = sitk.ReadImage(pet_path)
    mask = img > threshold
    mask: sitk.Image = sitk.Cast(mask, sitk.sitkUInt8)
    mask.CopyInformation(img)
    sitk.WriteImage(mask, output_path)
    return True

def main():
    parser = argparse.ArgumentParser(description="Génération de masques corporels (body.nii.gz) depuis PET.")
    parser.add_argument("--data-dir", '-d', type=str, required=True, help="Racine du dataset")
    parser.add_argument("--threshold", '-t', type=float, default=0.2, help="Seuil pour la génération du masque")
    # On limite un peu les workers par sécurité mémoire (ex: CPU - 2)
    default_workers = max(1, multiprocessing.cpu_count() - 2)
    parser.add_argument("--workers", type=int, default=default_workers, help="Nombre de processus parallèles")
    
    args = parser.parse_args()

    # 1. Recherche des fichiers PET
    print(f"Exploration de {args.data_dir}...")
    pet_files = []
    
    for root, dirs, files in os.walk(args.data_dir):
        for filename in files:
            if filename.startswith("PET") and filename.endswith(".nii.gz"):
                pet_files.append(os.path.join(root, filename))

    if not pet_files:
        print("Aucun fichier PET trouvé.")
        return

    print(f"Traitement de {len(pet_files)} volumes avec {args.workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Soumission des tâches
        future_to_file = {executor.submit(process_single_subject, (f, args.threshold)): f for f in pet_files}
        
        # Barre de progression
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(pet_files), desc="Génération Masques"):
            result = future.result()
            
            # Gestion basique des erreurs (affichage uniquement si erreur)
            if isinstance(result, str) and result.startswith("ERROR"):
                print(f"\n{result}")

    print("\nTerminé !")

if __name__ == "__main__":
    main()