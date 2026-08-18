import argparse
import os
from pathlib import Path
import torchio as tio
from tqdm import tqdm
import concurrent.futures

def process_file(filepath: Path):
    """
    Fonction isolée exécutée par chaque worker.
    Retourne un tuple : (statut, nom_fichier, message_erreur)
    """
    try:
        # On instancie la transfo DANS le worker pour éviter tout problème de pickle en RAM
        transform = tio.ToCanonical()
        img = tio.ScalarImage(filepath)
        
        if img.orientation == ('R', 'A', 'S'):
            return ('ALREADY_CANONICAL', filepath.name, None)
        else:
            img_canonical = transform(img)
            img_canonical.save(filepath)
            return ('CONVERTED', filepath.name, None)
    except Exception as e:
        return ('ERROR', filepath.name, str(e))

def main():
    # --- 1. Gestion des arguments ---
    parser = argparse.ArgumentParser(
        description="Standardise l'orientation (ToCanonical) en MULTIPROCESSING."
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        required=True, 
        help="Chemin vers le dossier racine des données (ex: ./data/PET-EARL/)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Nombre de processus parallèles (par défaut: tous les cœurs disponibles)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"❌ Erreur : Le dossier {data_dir} n'existe pas.")
        return

    # --- 2. Recherche et filtrage ---
    print(f"🔍 Recherche des fichiers (PET et EARL) dans : {data_dir}")
    all_nifti_files = list(data_dir.rglob("*.nii.gz")) + list(data_dir.rglob("*.nii"))
    target_files = [
        f for f in all_nifti_files 
        if f.name.lower().startswith(('pet', 'earl'))
    ]
    
    if not target_files:
        print(f"⚠️ Aucun fichier PET ou EARL trouvé.")
        return

    print(f"🎯 {len(target_files)} fichiers cibles trouvés.")
    print(f"🚀 Lancement du pool multiprocessing sur {args.workers} cœurs...\n")

    already_canonical = 0
    converted = 0
    errors = 0

    # --- 3. Exécution Multiprocessing avec suivi de progression ---
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Soumission de toutes les tâches à l'executor
        futures = {executor.submit(process_file, path): path for path in target_files}
        
        # as_completed permet de mettre à jour la barre de progression dès qu'un fichier a terminé
        with tqdm(total=len(target_files), desc="Standardisation RAS", unit=" fichier") as pbar:
            for future in concurrent.futures.as_completed(futures):
                status, filename, err_msg = future.result()
                
                if status == 'ALREADY_CANONICAL':
                    already_canonical += 1
                elif status == 'CONVERTED':
                    converted += 1
                elif status == 'ERROR':
                    errors += 1
                    tqdm.write(f"❌ Erreur sur {filename} : {err_msg}")
                    
                pbar.update(1)

    # --- 4. Bilan final ---
    print("\n" + "="*55)
    print("✅ TRAITEMENT MULTIPROCESSING TERMINÉ")
    print("="*55)
    print(f"➤ Fichiers déjà au format RAS (ignorés) : {already_canonical}")
    print(f"➤ Fichiers convertis et sauvegardés     : {converted}")
    if errors > 0:
        print(f"➤ Erreurs de lecture/écriture           : {errors}")
    print("="*55)

if __name__ == "__main__":
    main()