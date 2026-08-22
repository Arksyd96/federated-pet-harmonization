import os
import argparse
import SimpleITK as sitk
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def resample_mask_to_pet_grid(data):
    """
    Worker function: Cherche la référence et les masques via wildcards, 
    puis rééchantillonne tous les masques correspondants sur la grille de la référence.
    """
    input_subject_path, output_subject_path, ref_wildcard, mask_wildcards = data
    subject_id = os.path.basename(input_subject_path)
    
    try:
        # 1. Identifier l'image de référence STRICTEMENT UNIQUE
        ref_pattern = os.path.join(input_subject_path, ref_wildcard)
        ref_candidates = glob.glob(ref_pattern)
        
        if len(ref_candidates) == 0:
            return subject_id, False, f"Aucune référence trouvée pour '{ref_wildcard}'"
        if len(ref_candidates) > 1:
            noms = [os.path.basename(f) for f in ref_candidates]
            return subject_id, False, f"Plusieurs références trouvées pour '{ref_wildcard}' : {noms}. Il en faut EXACTEMENT UNE."
        
        ref_path = ref_candidates[0]
        pet_ref = sitk.ReadImage(ref_path)

        # 2. Configurer le Resampler (Strictement pour Masques : NearestNeighbor + UInt8)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(pet_ref) # Copie Size, Spacing, Origin, Direction
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        resampler.SetOutputPixelType(sitk.sitkUInt8) 

        os.makedirs(output_subject_path, exist_ok=True)

        # 3. Récupérer tous les masques via les wildcards fournis
        mask_paths = set() # Utilisation d'un set pour éviter les doublons si les wildcards se chevauchent
        for m_wildcard in mask_wildcards:
            m_pattern = os.path.join(input_subject_path, m_wildcard)
            for match in glob.glob(m_pattern):
                # Sécurité : On s'assure de ne pas traiter l'image de référence comme un masque
                if os.path.abspath(match) != os.path.abspath(ref_path):
                    mask_paths.add(match)
                    
        if not mask_paths:
            return subject_id, False, f"Aucun masque correspondant aux patterns fournis : {mask_wildcards}"
            
        # 4. Exécuter le rééchantillonnage
        resampled_count = 0
        for mask_path in mask_paths:
            mask_filename = os.path.basename(mask_path)
            mask_img = sitk.ReadImage(mask_path)
            
            resampled_mask = resampler.Execute(mask_img)
            
            out_path = os.path.join(output_subject_path, mask_filename)
            sitk.WriteImage(resampled_mask, out_path, useCompression=True)
            resampled_count += 1
            
        info_msg = f"{resampled_count} masques alignés sur {os.path.basename(ref_path)}"
        return subject_id, True, info_msg

    except Exception as e:
        return subject_id, False, str(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rééchantillonne des masques sur une image de référence définie par wildcard.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Dossier racine d'entrée")
    parser.add_argument("--output", "-o", type=str, required=True, help="Dossier racine de sortie")
    parser.add_argument("--ref", "-r", type=str, required=True, 
                        help="Wildcard pour l'image de référence (ex: 'PET*.nii.gz')")
    parser.add_argument("--masks", "-m", type=str, nargs='+', required=True, 
                        help="Liste de wildcards pour les masques (ex: 'liver*.nii.gz' '*mask*.nii')")
    
    args = parser.parse_args()

    subjects = sorted([d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))])
    
    tasks = [
        (os.path.join(args.input, s), os.path.join(args.output, s), args.ref, args.masks) 
        for s in subjects
    ]

    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"🚀 Début de l'alignement sur {num_workers} cœurs pour {len(subjects)} patients.")
    print(f"🎯 Pattern Référence : {args.ref}")
    print(f"🎯 Patterns Masques  : {', '.join(args.masks)}\n")

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_subject = {executor.submit(resample_mask_to_pet_grid, task): task for task in tasks}
        
        for future in tqdm(as_completed(future_to_subject), total=len(subjects), desc="Progression"):
            subj_id, success, info = future.result()
            
            if not success:
                tqdm.write(f"⚠️ Échec Sujet {subj_id} : {info}")
            else:
                pass
                # tqdm.write(f"✅ Sujet {subj_id} : {info}")
                
            results.append(success)

    print(f"\n✅ Terminé. {sum(results)}/{len(subjects)} patients traités avec succès.")