import argparse
import os
from pathlib import Path
import torchio as tio
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description="Standardise l'orientation des images PET et EARL (ToCanonical) via TorchIO."
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        required=True, 
        help="Chemin vers le dossier racine des données (ex: ./data/PET-EARL/)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"❌ Erreur : Le dossier {data_dir} n'existe pas.")
        return

    print(f"🔍 Recherche des fichiers (PET et EARL) dans : {data_dir}")
    
    all_nifti_files = list(data_dir.rglob("*.nii.gz")) + list(data_dir.rglob("*.nii"))
    target_files = [f for f in all_nifti_files if f.name.lower().startswith(('pet', 'earl'))]
    
    if not target_files:
        print(f"⚠️ Aucun fichier PET ou EARL trouvé.")
        return

    print(f"🎯 {len(target_files)} fichiers cibles trouvés. Lancement de l'audit et de la conversion...\n")

    transform = tio.ToCanonical()
    already_canonical = 0
    converted = 0
    errors = 0

    with tqdm(total=len(target_files), desc="Standardisation RAS", unit=" fichier") as pbar:
        for filepath in target_files:
            try:
                img = tio.ScalarImage(filepath)
                
                if img.orientation == ('R', 'A', 'S'):
                    already_canonical += 1
                else:
                    img_canonical = transform(img)
                    img_canonical.save(filepath)
                    converted += 1
                    
            except Exception as e:
                tqdm.write(f"❌ Erreur sur {filepath.name} : {e}")
                errors += 1
                
            pbar.update(1)

    print("\n" + "="*55)
    print("✅ TRAITEMENT TERMINÉ")
    print("="*55)
    print(f"➤ Fichiers déjà au bon format (ignorés) : {already_canonical}")
    print(f"➤ Fichiers convertis et sauvegardés     : {converted}")
    if errors > 0:
        print(f"➤ Erreurs de lecture/écriture           : {errors}")
    print("="*55)
    print("💡 Tu peux maintenant retirer 'tio.ToCanonical()' de ton DataLoader !")

if __name__ == "__main__":
    main()