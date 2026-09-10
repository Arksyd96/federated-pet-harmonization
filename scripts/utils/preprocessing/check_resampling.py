import argparse
import os
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm
import concurrent.futures

# Désactive le multithreading interne de SimpleITK pour éviter les conflits avec ProcessPoolExecutor
sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

def get_metadata(filepath: Path):
    """Lit uniquement l'en-tête du fichier pour extraire la géométrie."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(filepath))
    reader.ReadImageInformation()
    return {
        "size": reader.GetSize(),
        "spacing": reader.GetSpacing(),
        "origin": reader.GetOrigin(),
        "direction": reader.GetDirection()
    }

def compare_metadata(ref_meta, target_meta):
    """Compare les métadonnées avec une exigence d'égalité stricte (exact match)."""
    if ref_meta["size"] != target_meta["size"]:
        return False, f"Size: {ref_meta['size']} vs {target_meta['size']}"
    
    if ref_meta["spacing"] != target_meta["spacing"]:
        return False, f"Spacing: {ref_meta['spacing']} vs {target_meta['spacing']}"
    
    if ref_meta["origin"] != target_meta["origin"]:
        return False, f"Origin: {ref_meta['origin']} vs {target_meta['origin']}"
    
    if ref_meta["direction"] != target_meta["direction"]:
        return False, f"Direction: {ref_meta['direction']} vs {target_meta['direction']}"
        
    return True, ""

def process_patient(p_dir: Path):
    """Fonction exécutée par chaque worker pour un patient donné."""
    expected_files = ["pet.nii.gz", "earl.nii.gz", "body.nii.gz"]
    files = {name: p_dir / name for name in expected_files}
    
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        return ('MISSING', p_dir.name, f"Fichiers manquants : {', '.join(missing)}")

    try:
        # On prend PET comme référence absolue
        meta_pet = get_metadata(files["pet.nii.gz"])
        
        is_valid = True
        errors = []
        
        # Comparaison stricte avec EARL et BODY
        for target_name in ["earl.nii.gz", "body.nii.gz"]:
            meta_target = get_metadata(files[target_name])
            match, msg = compare_metadata(meta_pet, meta_target)
            
            if not match:
                is_valid = False
                errors.append(f"[{target_name}] {msg}")

        if is_valid:
            return ('VALID', p_dir.name, "")
        else:
            return ('INVALID', p_dir.name, " | ".join(errors))
            
    except Exception as e:
        return ('ERROR', p_dir.name, f"Erreur de lecture : {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Vérifie la cohérence géométrique stricte en multiprocessing.")
    parser.add_argument("--data-dir", type=str, required=True, help="Dossier racine des patients.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Nombre de processus parallèles.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Le dossier {data_dir} n'existe pas.")
        return

    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"🔍 Analyse géométrique stricte de {len(patient_dirs)} patients sur {args.workers} cœurs...\n")

    valid_count = 0
    error_count = 0

    # Lancement du multiprocessing
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_patient, p_dir): p_dir for p_dir in patient_dirs}
        
        with tqdm(total=len(patient_dirs), desc="Vérification", unit=" patient") as pbar:
            for future in concurrent.futures.as_completed(futures):
                status, p_name, msg = future.result()
                
                if status == 'VALID':
                    valid_count += 1
                elif status == 'INVALID':
                    error_count += 1
                    tqdm.write(f"❌ {p_name:<15} | Incohérence : {msg}")
                elif status == 'MISSING' or status == 'ERROR':
                    error_count += 1
                    tqdm.write(f"⚠️ {p_name:<15} | {msg}")
                    
                pbar.update(1)

    print("\n" + "="*55)
    print("✅ VÉRIFICATION STRICTE TERMINÉE")
    print("="*55)
    print(f"➤ Patients valides (géométrie 100% identique) : {valid_count}")
    print(f"➤ Patients avec incohérences ou manquants   : {error_count}")
    print("="*55)

if __name__ == "__main__":
    main()