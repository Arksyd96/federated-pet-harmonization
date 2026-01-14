import argparse
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import concurrent.futures
import multiprocessing

def process_patient_volume(args):
    file_path, log_transform = args
    try:
        img = nib.load(file_path)
        data = img.get_fdata()

        if log_transform:
            data = np.log1p(data)
        
        valid_max = np.nanmax(data)
        
        if np.isfinite(valid_max):
            return valid_max
            
    except Exception as e:
        return None
    
    return None

def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Analyse des maxima SUV dans des volumes PET NIfTI (Parallélisé).")
    parser.add_argument("--data-dir", type=str, required=True, help="Répertoire racine contenant les données PET NIfTI.")
    parser.add_argument("--log-transform", action='store_true', help="Appliquer une transformation logarithmique aux données avant analyse.")
    default_workers = max(1, multiprocessing.cpu_count() - 2)
    parser.add_argument("--workers", type=int, default=default_workers, help=f"Nombre de processus (défaut: {default_workers}).")
    args = parser.parse_args()

    # --- CONFIGURATION ---
    ROOT_DIR = args.data_dir

    print(f"Exploration de {ROOT_DIR} ...")

    # 1. Récupération de la liste des fichiers (étape rapide, monothread)
    pet_files = []
    for root, dirs, files in os.walk(ROOT_DIR):
        for filename in files:
            if filename.startswith("PET_") and filename.endswith(".nii.gz"):
                pet_files.append(os.path.join(root, filename))

    total_files = len(pet_files)
    print(f"Analyse des maxima pour {total_files} patients avec {args.workers} workers...")

    pet_max_values = []

    # 2. Exécution Parallèle (étape lente, multithread)
    if total_files > 0:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            # On soumet toutes les tâches
            # future_to_file permettrait de savoir quel fichier a échoué si besoin
            futures = {executor.submit(process_patient_volume, (f, args.log_transform)): f for f in pet_files}
            
            # as_completed permet de mettre à jour la barre dès qu'un fichier est fini
            for future in tqdm(concurrent.futures.as_completed(futures), total=total_files, desc="Calcul Maxima"):
                result = future.result()
                if result is not None:
                    pet_max_values.append(result)
                else:
                    # Optionnel: Récupérer le nom du fichier qui a échoué
                    failed_file = futures[future]
                    # print(f"Echec sur : {failed_file}")

    # 3. Analyse Statistique Détaillée
    if not pet_max_values:
        print("Aucune donnée valide trouvée.")
        return

    pet_max_values = np.array(pet_max_values)
    total_patients = len(pet_max_values)

    # Calcul des percentiles
    percentiles_to_check = [90, 95, 98, 99, 99.5, 99.9]
    stats = {}

    for p in percentiles_to_check:
        thresh = np.percentile(pet_max_values, p)
        # Nombre de patients qui dépassent ce seuil (qui seront clippés)
        n_clipped = np.sum(pet_max_values > thresh)
        n_kept = total_patients - n_clipped
        stats[p] = (thresh, n_kept, n_clipped)

    absolute_max = np.max(pet_max_values)

    print("\n" + "="*60)
    print(f"DISTRIBUTION DES MAXIMA SUV (Sur {total_patients} patients valides)")
    print("="*60)
    print(f"{'Percentile':<10} | {'Seuil (SUV)':<12} | {'Patients OK':<12} | {'Patients Clippés':<15}")
    print("-" * 60)

    for p in percentiles_to_check:
        thresh, kept, clipped = stats[p]
        print(f"P{p:<9} | {thresh:<12.2f} | {kept:<12} | {clipped:<15} ({100*clipped/total_patients:.1f}%)")

    print("-" * 60)
    print(f"Max Absolu : {absolute_max:.2f} SUV (Patient le plus chaud ou artefact)")
    print("="*60)

    # Interprétation automatique
    recommended_p = 98 # Par défaut
    for p in percentiles_to_check:
        thresh, kept, clipped = stats[p]
        # Règle empirique : on tolère de clipper ~2-5% des patients si ça permet de gagner beaucoup en dynamique
        if clipped <= (0.05 * total_patients): 
            recommended_p = p
            break

    rec_thresh, _, rec_clipped = stats[recommended_p]
    print(f"\n💡 RECOMMANDATION : Utilise le P{recommended_p} (Seuil = {rec_thresh:.2f})")
    print(f"   Cela permet de couvrir {total_patients - rec_clipped} patients sans aucune perte,")
    print(f"   et de ne saturer légèrement que les {rec_clipped} patients les plus extrêmes.")

if __name__ == '__main__':
    # Protection nécessaire pour le multiprocessing sous Windows/macOS
    main()