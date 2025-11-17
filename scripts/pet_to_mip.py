#!/usr/bin/env python3
"""
Convert PET and EARL NIfTI files to 2D MIP images and save next to originals.

Dataset structure expected:
    root/
      subject1/
        PT...(.nii or .nii.gz)         # PET (no 'EARL' in filename)
        PT...EARL...(.nii or .nii.gz)  # EARL PET (contains 'EARL')
        CT...(.nii.gz)

Rules used to identify files:
- file name starting with 'PT' (case-insensitive) and NOT containing 'EARL' -> PET
- file name starting with 'PT' and containing 'EARL' (case-insensitive) -> EARL PET

For each PET/ EARL found the script computes a maximum-intensity projection (MIP)
along the axial direction (Z) and saves a 2D NIfTI next to the original file with
suffix `_MIP.nii.gz` (unless --overwrite is used).

Usage:
    python pet_to_mip.py --input /path/to/dataset_root [--overwrite]

Dependencies:
    pip install SimpleITK numpy tqdm

Notes:
- The script uses only the `os` library for filesystem operations (no pathlib).
- Handles 3D and simple 4D (time,x,y,z) volumes by collapsing time then doing Z-MIP.

"""

import os
import sys
import argparse
import logging

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def compute_mip_from_sitk(img):
    """Compute a 2D MIP (max intensity projection along Z) from a SimpleITK image.

    Behaviour:
      - If image is 3D: assume shape (x,y,z) in SITK ordering -> numpy array shape (z,y,x)
        -> MIP = max over axis=0 of the array -> result (y,x)
      - If image is 4D: assume (t,z,y,x) in numpy after GetArrayFromImage -> collapse t by max,
        then max over z.
      - If image is already 2D -> returns a copy.

    Returns (mip_sitk_image, info_dict)
    """
    arr = sitk.GetArrayFromImage(img)  # shape: (z,y,x) or (t,z,y,x) or (y,x)
    info = {'orig_shape': arr.shape}

    if arr.ndim == 4:
        # assume (t, z, y, x) or (time, slices, rows, cols)
        # collapse time first (max over axis 0), then collapse z (now axis 0)
        logging.info('4D image detected - collapsing time dimension by max before Z-MIP')
        arr_coll = np.max(arr, axis=0)
        mip2d = np.max(arr_coll, axis=0)
    elif arr.ndim == 3:
        # typical case (z,y,x)
        mip2d = np.max(arr, axis=0)
    elif arr.ndim == 2:
        mip2d = arr.copy()
    else:
        raise RuntimeError(f'Unsupported image dimensionality: {arr.ndim}')

    info['mip_shape'] = mip2d.shape

    # Recreate SimpleITK image from mip2d (2D). Note: GetImageFromArray expects (rows, cols)
    mip_img = sitk.GetImageFromArray(mip2d)

    # Propagate spacing/origin/direction where sensible (drop Z components)
    try:
        spacing = img.GetSpacing()  # tuple (sx, sy, sz) for 3D
        if len(spacing) >= 2:
            new_spacing = (spacing[0], spacing[1])
            mip_img.SetSpacing(new_spacing)
    except Exception:
        pass

    try:
        origin = img.GetOrigin()  # tuple (ox, oy, oz)
        if len(origin) >= 2:
            new_origin = (origin[0], origin[1])
            mip_img.SetOrigin(new_origin)
    except Exception:
        pass

    try:
        direction = img.GetDirection()  # flat tuple length 9 for 3D
        if len(direction) >= 4:
            # 2D direction is 4 elements: [d00, d01, d10, d11]
            new_dir = (direction[0], direction[1], direction[3], direction[4])
            mip_img.SetDirection(new_dir)
    except Exception:
        pass

    return mip_img, info


def find_subject_folders(root):
    """Return list of immediate subfolders of root (subjects). Uses os.listdir and os.path.isdir."""
    subs = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full):
            subs.append(full)
    return subs


def find_nifti_files_in_folder(folder):
    files = []
    for name in sorted(os.listdir(folder)):
        low = name.lower()
        if low.endswith('.nii') or low.endswith('.nii.gz'):
            files.append(os.path.join(folder, name))
    return files


def process_dataset(root, overwrite=False):
    subjects = find_subject_folders(root)
    if not subjects:
        logging.error('No subject subfolders found under root: %s', root)
        return

    for subj in subjects:
        logging.info('Processing subject folder: %s', subj)
        nifti_files = find_nifti_files_in_folder(subj)
        if not nifti_files:
            logging.warning('No NIfTI files found in %s - skipping', subj)
            continue

        for infile in nifti_files:
            if not os.path.basename(infile).lower().startswith('pt'):
                continue

            base_no_ext = os.path.basename(infile).split('.nii.gz')[0]
            out_name = base_no_ext + '_MIP.nii.gz'
            out_path = os.path.join(os.path.dirname(infile), out_name)
            

            if os.path.exists(out_path) and not overwrite:
                logging.info('Output exists: %s - skipping (use --overwrite to force)', out_path)
                continue

            try:
                img = sitk.ReadImage(infile)
                mip_img, info = compute_mip_from_sitk(img)
                logging.info('Saving MIP to: %s (orig shape=%s -> mip shape=%s)', out_path, info.get('orig_shape'), info.get('mip_shape'))
                sitk.WriteImage(mip_img, out_path)
            except Exception as e:
                logging.exception('Failed processing %s: %s', infile, e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert PET and EARL NIfTI to 2D MIP images')
    parser.add_argument('--input', '-i', required=True, help='Root dataset folder (contains subject subfolders)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing MIP outputs')
    args = parser.parse_args()

    root = args.input
    if not os.path.isdir(root):
        logging.error('Input root not found or not a directory: %s', root)
        sys.exit(1)

    process_dataset(root, overwrite=args.overwrite)

