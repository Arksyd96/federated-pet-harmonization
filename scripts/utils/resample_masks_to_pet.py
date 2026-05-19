import os
import argparse
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def resample_mask_to_pet_grid(data):
    """
    Worker function: Resamples all masks of a subject to its PET image space.
    """
    input_subject_path, output_subject_path = data
    subject_id = os.path.basename(input_subject_path)
    
    try:
        os.makedirs(output_subject_path, exist_ok=True)
        all_files = os.listdir(input_subject_path)
        
        # 1. Identify the reference PET image
        pet_files = [f for f in all_files if f.startswith('PET') and f.endswith(('.nii', '.nii.gz')) and 'MIP' not in f]
        if not pet_files:
            return subject_id, False, "No PET image found"
        
        pet_ref = sitk.ReadImage(os.path.join(input_subject_path, pet_files[0]))
        
        # 2. Identify masks (excluding PET, EARL, CT, NAC)
        mask_files = [
            f for f in all_files 
            if f.endswith(('.nii', '.nii.gz')) 
            and not (f.startswith('PET') or f.startswith('EARL') or f.startswith('CT') or f.startswith('NAC') or f.startswith('predicted'))
        ]
        
        if not mask_files:
            return subject_id, False, "No masks found"

        # 3. Setup Resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(pet_ref) # Sets Size, Spacing, Origin, Direction
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        resampler.SetOutputPixelType(sitk.sitkUInt8) # Masks should be UInt8

        for mask_name in mask_files:
            mask_path = os.path.join(input_subject_path, mask_name)
            mask_img = sitk.ReadImage(mask_path)
            
            # Execute resampling
            resampled_mask = resampler.Execute(mask_img)
            
            # Save to output directory
            sitk.WriteImage(resampled_mask, os.path.join(output_subject_path, mask_name), useCompression=True)
            
        return subject_id, True, f"{len(mask_files)} masks resampled"

    except Exception as e:
        return subject_id, False, str(e)

def main():
    parser = argparse.ArgumentParser(description="Resample masks to PET image space for all subjects.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Root input directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Root output directory")
    args = parser.parse_args()

    subjects = sorted([d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))])
    tasks = [
        (os.path.join(args.input, s), os.path.join(args.output, s)) 
        for s in subjects
    ]

    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"🚀 Starting resampling on {num_workers} cores for {len(subjects)} subjects.")

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_subject = {executor.submit(resample_mask_to_pet_grid, task): task for task in tasks}
        
        for future in tqdm(as_completed(future_to_subject), total=len(subjects), desc="Resampling Progress"):
            subj_id, success, info = future.result()
            if not success:
                print(f"⚠️ Subject {subj_id} failed: {info}")
            results.append(success)

    print(f"\n✅ Finished. Processed {sum(results)}/{len(subjects)} subjects successfully.")

if __name__ == "__main__":
    main()