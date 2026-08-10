import os
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Désactiver les warnings SimpleITK dans les processus enfants
sitk.ProcessObject.SetGlobalWarningDisplay(False)

def process_single_subject(args):
    """Fonction exécutée par chaque worker pour un patient donné."""
    subj, input_dir, output_dir, output_filename, lobes_to_merge = args
    
    subj_input_path = os.path.join(input_dir, subj)
    subj_output_path = os.path.join(output_dir, subj)
    
    ref_img = None
    merged_mask = None
    missing_lobes = []
    
    # 1. Lecture et fusion
    for lobe in lobes_to_merge:
        lobe_path = os.path.join(subj_input_path, lobe)
        
        if os.path.exists(lobe_path):
            img = sitk.ReadImage(lobe_path)
            arr = sitk.GetArrayFromImage(img)
            
            # Binarisation et union booléenne
            binary_arr = (arr > 0)
            
            if merged_mask is None:
                ref_img = img
                merged_mask = binary_arr
            else:
                merged_mask = merged_mask | binary_arr
        else:
            missing_lobes.append(lobe)

    # 2. Sauvegarde
    if merged_mask is not None:
        os.makedirs(subj_output_path, exist_ok=True)
        
        final_array = merged_mask.astype(np.uint8)
        
        out_img = sitk.GetImageFromArray(final_array)
        out_img.CopyInformation(ref_img)
        
        out_path = os.path.join(subj_output_path, output_filename)
        sitk.WriteImage(out_img, out_path)
        
        if missing_lobes:
            return (subj, "partial", f"⚠️ {subj} : Fusion partielle (manque {', '.join(missing_lobes)})")
        return (subj, "success", None)
    else:
        return (subj, "missing", f"❌ {subj} : Aucun masque source trouvé. Ignoré.")


def merge_masks(input_dir, output_dir, output_filename, num_workers):
    lobes_to_merge = [
        'lung_upper_lobe_right.nii.gz',
        'lung_middle_lobe_right.nii.gz',
        'lung_lower_lobe_right.nii.gz'
    ]

    if not os.path.exists(input_dir):
        print(f"❌ Erreur : Le répertoire d'entrée '{input_dir}' n'existe pas.")
        return

    subjects = [s for s in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, s))]
    
    if not subjects:
        print(f"⚠️ Aucun dossier patient trouvé dans '{input_dir}'.")
        return

    print(f"🚀 Lancement de la fusion pour {len(subjects)} patients avec {num_workers} workers...")
    
    # Préparation des arguments pour le multiprocessing
    tasks = [
        (subj, input_dir, output_dir, output_filename, lobes_to_merge) 
        for subj in subjects
    ]
    
    success_count = 0
    missing_count = 0

    # Lancement du pool de processus
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_subject, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Traitement"):
            subj, status, msg = future.result()
            
            if status == "success" or status == "partial":
                success_count += 1
                if msg: # Affiche les alertes partielles sans casser la barre de progression
                    tqdm.write(msg)
            elif status == "missing":
                missing_count += 1
                tqdm.write(msg)

    print("\n✅ BILAN DE LA FUSION")
    print("-" * 30)
    print(f"Dossier source : {input_dir}")
    print(f"Dossier cible  : {output_dir}")
    print(f"Patients traités avec succès : {success_count}")
    print(f"Patients ignorés (fichiers manquants) : {missing_count}")


def main():
    parser = argparse.ArgumentParser(description="Fusion de plusieurs masques NIfTI (union) pour une liste de patients (Multiprocessing).")
    
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Chemin vers le répertoire d'entrée contenant les dossiers des patients."
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Chemin vers le répertoire de sortie. Si non renseigné, sauvegarde dans le répertoire d'entrée."
    )
    
    parser.add_argument(
        "--filename", 
        type=str, 
        default="lung.nii.gz", 
        help="Nom du fichier généré (défaut: lung.nii.gz)."
    )
    
    parser.add_argument(
        "--workers", 
        type=int, 
        default=os.cpu_count() // 2, 
        help="Nombre de processus à utiliser en parallèle (défaut: tous les cœurs disponibles)."
    )

    args = parser.parse_args()

    output_dir = args.output if args.output else args.input
    merge_masks(args.input, output_dir, args.filename, args.workers)


if __name__ == "__main__":
    main()