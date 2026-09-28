import os
import argparse
import concurrent.futures
import logging
import fnmatch
import gc
import torch
import SimpleITK as sitk
import torchio as tio
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

def align_image_sitk(input_path, ref_path, interpolator_name):
    """
    Aligns an image to a reference grid strictly using SimpleITK ResampleImageFilter.
    Dynamically casts OutputPixelType based on the interpolator to preserve masks as UInt8.
    """
    input_img = sitk.ReadImage(input_path)
    ref_img = sitk.ReadImage(ref_path)
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetDefaultPixelValue(0)
    
    if interpolator_name == 'nearest':
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetOutputPixelType(sitk.sitkUInt8)
    elif interpolator_name == 'bspline':
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetOutputPixelType(sitk.sitkFloat32)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        
    aligned_img = resampler.Execute(input_img)
    
    if interpolator_name in ['bspline', 'linear']:
        array = sitk.GetArrayFromImage(aligned_img)
        array[array < 0.0] = 0.0
        clamped_img = sitk.GetImageFromArray(array)
        clamped_img.CopyInformation(aligned_img)
        aligned_img = clamped_img
        
    return aligned_img

def process_patient_resample(p_dir, out_dir, args):
    """
    Worker function executing the dynamic resampling pipeline.
    Branches into SimpleITK for reference-based alignment or TorchIO for pure spacing operations.
    """
    p_name = os.path.basename(p_dir)
    
    files_to_process = []
    for f in os.listdir(p_dir):
        if not (f.endswith('.nii.gz') or f.endswith('.nii')):
            continue
        for pattern in args.images:
            if fnmatch.fnmatch(f.lower(), pattern.lower()):
                files_to_process.append(f)
                break
                
    if not files_to_process:
        return p_name, False

    out_patient_dir = os.path.join(out_dir, p_name)
    os.makedirs(out_patient_dir, exist_ok=True)

    if args.reference:
        ref_files = [f for f in os.listdir(p_dir) if fnmatch.fnmatch(f.lower(), args.reference.lower())]
        if not ref_files:
            return p_name, False
            
        ref_path = os.path.join(p_dir, ref_files[0])
        
        for file_name in files_to_process:
            input_path = os.path.join(p_dir, file_name)
            aligned_sitk_img = align_image_sitk(input_path, ref_path, args.interpolator)
            
            save_path = os.path.join(out_patient_dir, file_name)
            sitk.WriteImage(aligned_sitk_img, save_path)
            
    else:
        images_dict = {}
        for file_name in files_to_process:
            path = os.path.join(p_dir, file_name)
            images_dict[file_name] = tio.ScalarImage(path)
            
        subject = tio.Subject(**images_dict)
        
        resample_transform = tio.Resample(
            target=tuple(args.spacing),
            image_interpolation=args.interpolator
        )
        subject = resample_transform(subject)
        
        if args.interpolator in ['bspline', 'linear']:
            for image_name in subject.get_images_names():
                mod_tensor = subject[image_name].data
                mod_tensor = torch.clamp(mod_tensor, min=0.0)
                subject[image_name].set_data(mod_tensor)
                
        for file_name in subject.get_images_names():
            save_path = os.path.join(out_patient_dir, file_name)
            subject[file_name].save(save_path)

    gc.collect()
    return p_name, True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Medical Image Resampling & Alignment.")
    parser.add_argument("--input-dir", type=str, required=True, help="Input dataset root directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory. Defaults to overwrite.")
    parser.add_argument("--images", nargs='+', required=True, help="Image wildcard patterns to process.")
    parser.add_argument("--interpolator", type=str, choices=['bspline', 'nearest', 'linear'], required=True, help="Interpolation method.")
    
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--spacing", type=float, nargs=3, help="Target spacing in mm.")
    target_group.add_argument("--reference", type=str, help="Reference image wildcard for perfect grid alignment.")
    
    default_workers = max(1, (os.cpu_count() or 2) // 2)
    parser.add_argument("--workers", type=int, default=default_workers, help="Number of thread workers.")
    
    args = parser.parse_args()
    
    out_dir = args.output_dir if args.output_dir else args.input_dir
    patient_dirs = [os.path.join(args.input_dir, d) for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    
    if out_dir != args.input_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    logger.info(f"Resampling {len(patient_dirs)} patients with {args.workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_patient_resample, p_dir, out_dir, args): p_dir for p_dir in patient_dirs}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(patient_dirs), desc="Resampling"):
            p_name, success = future.result()
            if not success:
                tqdm.write(f"WARNING: Failed or missing files for {p_name}")

