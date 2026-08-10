import os
import argparse
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def resample_mask_to_pet_grid(data):
    """
    Worker function: Resamples specific masks of a subject to the provided reference image space.
    """
    input_subject_path, output_subject_path, ref_filename, mask_filenames = data
    subject_id = os.path.basename(input_subject_path)
    
    try:
        os.makedirs(output_subject_path, exist_ok=True)
        
        # 1. Identify the explicit reference image
        ref_path = os.path.join(input_subject_path, ref_filename)
        if not os.path.exists(ref_path):
            return subject_id, False, f"Reference image '{ref_filename}' not found"
        
        pet_ref = sitk.ReadImage(ref_path)

        # 2. Setup Resampler (Strictly for Masks: NearestNeighbor + UInt8)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(pet_ref) # Sets Size, Spacing, Origin, Direction
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        resampler.SetOutputPixelType(sitk.sitkUInt8) 

        # 3. Resample specifically requested masks
        resampled_count = 0
        missing_masks = []
        
        for mask_name in mask_filenames:
            mask_path = os.path.join(input_subject_path, mask_name)
            
            if os.path.exists(mask_path):
                mask_img = sitk.ReadImage(mask_path)
                
                # Execute resampling
                resampled_mask = resampler.Execute(mask_img)
                
                # Save to output directory
                out_path = os.path.join(output_subject_path, mask_name)
                sitk.WriteImage(resampled_mask, out_path, useCompression=True)
                resampled_count += 1
            else:
                missing_masks.append(mask_name)
                
        if resampled_count == 0:
            return subject_id, False, "None of the specified masks were found"
            
        info_msg = f"{resampled_count} masks resampled"
        if missing_masks:
            info_msg += f" (Missing: {', '.join(missing_masks)})"
            
        return subject_id, True, info_msg

    except Exception as e:
        return subject_id, False, str(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample specific masks to a reference image space.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Root input directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Root output directory")
    parser.add_argument("--ref", "-r", type=str, required=True, 
                        help="Exact filename of the reference image (e.g., PET_TEP_TAP_AC.nii.gz)")
    parser.add_argument("--masks", "-m", type=str, nargs='+', required=True, 
                        help="List of mask filenames to resample (e.g., liver.nii.gz brain.nii.gz lesion.nii.gz)")
    
    args = parser.parse_args()

    subjects = sorted([d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))])
    
    # Intégration des nouveaux arguments dans le tuple de tâches pour les workers
    tasks = [
        (os.path.join(args.input, s), os.path.join(args.output, s), args.ref, args.masks) 
        for s in subjects
    ]

    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"🚀 Starting resampling on {num_workers} cores for {len(subjects)} subjects.")
    print(f"🎯 Reference image: {args.ref}")
    print(f"🎯 Masks to process: {', '.join(args.masks)}\n")

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_subject = {executor.submit(resample_mask_to_pet_grid, task): task for task in tasks}
        
        for future in tqdm(as_completed(future_to_subject), total=len(subjects), desc="Resampling Progress"):
            subj_id, success, info = future.result()
            
            if not success:
                tqdm.write(f"⚠️ Subject {subj_id} failed: {info}")
            elif "Missing" in info:
                tqdm.write(f"ℹ️ Subject {subj_id} partial: {info}")
                
            results.append(success)

    print(f"\n✅ Finished. Processed {sum(results)}/{len(subjects)} subjects successfully.")