import os
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

def merge_masks(input_dir, output_dir, output_filename):
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

    success_count = 0
    missing_count = 0

    print(f"🚀 Lancement de la fusion pour {len(subjects)} patients...")
    
    for subj in tqdm(subjects, desc="Traitement"):
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
                tqdm.write(f"⚠️ {subj} : Fusion partielle (manque {', '.join(missing_lobes)})")
                
            success_count += 1
        else:
            tqdm.write(f"❌ {subj} : Aucun masque source trouvé. Ignoré.")
            missing_count += 1

    print("\n✅ BILAN DE LA FUSION")
    print("-" * 30)
    print(f"Dossier source : {input_dir}")
    print(f"Dossier cible  : {output_dir}")
    print(f"Patients traités avec succès : {success_count}")
    print(f"Patients ignorés (fichiers manquants) : {missing_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusion de plusieurs masques NIfTI (union) pour une liste de patients.")
    parser.add_argument("--input", type=str, required=True, help="Chemin vers le répertoire d'entrée contenant les dossiers des patients.")
    parser.add_argument("--output", type=str, default=None, help="Chemin vers le répertoire de sortie. Si non renseigné, sauvegarde dans le répertoire d'entrée.")
    parser.add_argument("--filename", type=str, default="lung.nii.gz", help="Nom du fichier généré (défaut: lung.nii.gz).")
    args = parser.parse_args()

    output_dir = args.output if args.output else args.input
    merge_masks(args.input, output_dir, args.filename)
