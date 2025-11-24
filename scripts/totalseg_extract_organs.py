#!/usr/bin/env python3
import os
import argparse
import logging
import shutil

from totalsegmentator.python_api import totalsegmentator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_subject_folders(root):
    subs = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full):
            subs.append(full)
    return subs


def find_ct_file_in_folder(folder):
    for name in sorted(os.listdir(folder)):
        low = name.lower()
        if low.endswith('.nii') or low.endswith('.nii.gz'):
            if 'ct' in low.split('_')[0] or low.startswith('ct') or 'ct' in low:
                return os.path.join(folder, name)
    return None


def collect_segmentation_outputs(ts_out_dir, subj_dir, overwrite=False, move_files=True):
    moved = []

    for name in sorted(os.listdir(ts_out_dir)):
        low = name.lower()
        if low.endswith('.nii') or low.endswith('.nii.gz'):
            # skip the combined segmentation file if present (optional)
            if low in ('segmentation.nii', 'segmentation.nii.gz', 'segmentation_all.nii', 'segmentation_all.nii.gz'):
                continue

            src = os.path.join(ts_out_dir, name)
            dst = os.path.join(subj_dir, name)
            if os.path.exists(dst) and not overwrite:
                logging.info('Destination exists, skipping: %s', dst)
                continue
            try:
                if move_files:
                    shutil.move(src, dst)
                else:
                    shutil.copy2(src, dst)
                moved.append(dst)
                # logging.info('Saved organ mask: %s', dst)
            except Exception as e:
                logging.exception('Failed to move/copy %s -> %s : %s', src, dst, e)
    return moved


def run_totalseg_on_subject(ct_file, out_folder, device='cpu', task='total', roi_subset=None, fast=False):
    logging.info('Running TotalSegmentator on %s -> %s (device=%s, task=%s, roi_subset=%s)', ct_file, out_folder, device, task, roi_subset)
    try:
        totalsegmentator(ct_file, out_folder, task=task, device=device, roi_subset=roi_subset, fast=fast)
        return True
    except Exception as e:
        print('TotalSegmentator failed:', e)
        return False


def process_dataset(root, device='cpu', task='total', roi_subset=None, overwrite=False, keep_temp=False, fast=False):
    if os.path.isfile(root):
        logging.error('Input path is a file, expected folder: %s', root)
        return
    
    subjects = find_subject_folders(root)
    if not subjects:
        logging.error('No subject subfolders found under root: %s', root)
        return

    for subj in subjects:
        logging.info('Processing subject: %s', subj)
        ct_file = find_ct_file_in_folder(subj)
        if not ct_file:
            logging.warning('No CT found in %s - skipping (TotalSegmentator requires CT input)', subj)
            continue

        # create a temporary output folder inside the subject folder
        ts_out = os.path.join(subj, 'totalseg_tmp')
        if not os.path.isdir(ts_out):
            os.makedirs(ts_out, exist_ok=True)

        ok = run_totalseg_on_subject(ct_file, ts_out, device=device, task=task, roi_subset=roi_subset, fast=fast)
        if not ok:
            logging.error('Segmentation failed for subject %s', subj)
            if not keep_temp:
                shutil.rmtree(ts_out)
            continue

        moved = collect_segmentation_outputs(ts_out, subj, overwrite=overwrite, move_files=True)

        if not keep_temp:
            # remove remaining files
            try:
                shutil.rmtree(ts_out)
            except Exception as e:
                logging.warning('Could not remove temp folder %s: %s', ts_out, e)

        logging.info('Finished subject %s (moved %d organ files)', subj, len(moved))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run TotalSegmentator on CTs in dataset and extract per-organ nifti masks')
    parser.add_argument('--input', '-i', required=True, help='Root dataset folder (subjects as subfolders)')
    parser.add_argument('--device', default='gpu', help='Device for TotalSegmentator (cpu or gpu)')
    parser.add_argument('--task', default='total', help='TotalSegmentator task (default: total)')
    parser.add_argument('--roi-subset', default=None, help='Comma-separated list of organ names to restrict (optional)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing organ files')
    parser.add_argument('--keep-temp', action='store_true', help='Keep TotalSegmentator output folder for debugging')
    parser.add_argument('--fast', action='store_true', help='Use fast (lower resolution) TotalSegmentator mode')
    args = parser.parse_args()

    roi_subset = args.roi_subset.split(',') if args.roi_subset else None
    process_dataset(
        args.input, 
        device=args.device, 
        task=args.task, 
        roi_subset=roi_subset, 
        overwrite=args.overwrite, 
        keep_temp=args.keep_temp, 
        fast=args.fast
    )
