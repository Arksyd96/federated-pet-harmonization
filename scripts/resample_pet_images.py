import os
import argparse
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def resample_one_patient(data):
    """
    Fonction wrapper qui traite UN patient (pour le multiprocessing).
    """
    input_root, output_root, subject, target_spacing = data
    
    subject_input_path = os.path.join(input_root, subject)
    subject_output_path = os.path.join(output_root, subject)
    
    # Création du dossier destination
    os.makedirs(subject_output_path, exist_ok=True)
    
    files = [f for f in os.listdir(subject_input_path) 
             if f.endswith(('.nii', '.nii.gz')) 
             and 'MIP' not in f
             and not f.startswith('PT')]
    
    processed_count = 0
    
    for filename in files:
        input_file = os.path.join(subject_input_path, filename)
        output_file = os.path.join(subject_output_path, filename)

        try:
            # 1. Lecture
            image = sitk.ReadImage(input_file)

            # 2. Calcul nouvelle taille
            original_size = image.GetSize()
            original_spacing = image.GetSpacing()
            
            new_size = [
                int(round(osz * osp / nsp))
                for osz, osp, nsp in zip(original_size, original_spacing, target_spacing)
            ]

            # 3. Resampling
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(target_spacing)
            resampler.SetSize(new_size)
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetInterpolator(sitk.sitkBSpline) # B-Spline pour PET
            resampler.SetDefaultPixelValue(0)
            
            resampled_image = resampler.Execute(image)

            # 4. Sauvegarde
            # ASTUCE VITESSE : Si vous avez de la place disque, 
            # remplacez filename par filename.replace('.gz', '') pour sauver en .nii non compressé.
            # C'est 2x plus rapide à l'écriture.
            sitk.WriteImage(resampled_image, output_file)
            processed_count += 1
            
        except Exception as e:
            print(f"Erreur sur {subject}/{filename}: {e}")
            
    return processed_count

def process_dataset_multicore(input_root, output_root, target_spacing):
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    subjects = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    subjects.sort()

    # Détection du nombre de coeurs (laisser 1 ou 2 coeurs libres pour le système)
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    
    print(f"🚀 Démarrage du resampling sur {num_workers} cœurs CPU.")
    print(f"Dataset : {len(subjects)} sujets -> {target_spacing} mm")

    # Préparation des arguments pour chaque process
    tasks = []
    for subject in subjects:
        tasks.append((input_root, output_root, subject, target_spacing))

    # Exécution Parallèle
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # On utilise tqdm pour suivre l'avancement global
        futures = {executor.submit(resample_one_patient, task): task for task in tasks}
        
        for _ in tqdm(as_completed(futures), total=len(subjects), desc="Progression"):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--spacing", "-s", type=float, nargs=3, default=[2.0, 2.0, 2.0])
    args = parser.parse_args()

    process_dataset_multicore(args.input, args.output, args.spacing)