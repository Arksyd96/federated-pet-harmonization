import os
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm


def discover_subject_folders(root: str) -> List[str]:
    return [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]


def find_mip_files(folder: str, pattern: str) -> List[str]:
    return [
        os.path.join(folder, f) for f in sorted(os.listdir(folder)) 
        if pattern in f.lower() 
        and (f.lower().endswith('.nii') or f.lower().endswith('.nii.gz'))
    ]


def build_2d_extractor(binWidth: float = 0.25) -> featureextractor.RadiomicsFeatureExtractor:
    settings = {
        'binWidth': float(binWidth),
        'resampledPixelSpacing': None,
        'interpolator': sitk.sitkBSpline,
        'enableCExtensions': True,
        'force2D': True,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    # extractor.enableFeatureClassByName('shape2D')
    # extractor.enableFeatureClassByName('shape')
    # extractor.enableFeatureClassByName('glcm')
    # extractor.enableFeatureClassByName('glrlm')
    # extractor.enableFeatureClassByName('glszm')
    # extractor.enableFeatureClassByName('ngtdm')
    # extractor.enableFeatureClassByName('gldm')
    return extractor


def make_mask_from_image(img: sitk.Image, threshold: float = 1e-6) -> sitk.Image:
    mask = sitk.BinaryThreshold(img, lowerThreshold=threshold, upperThreshold=1e9, insideValue=1, outsideValue=0)
    return sitk.Cast(mask, sitk.sitkUInt8)


def extract_features_from_mip(image_path: str, extractor: featureextractor.RadiomicsFeatureExtractor, mask_threshold: float = 1e-3) -> Dict[str, Any]:
    img = sitk.ReadImage(image_path) # [W, 1, H]
    stats_arr = sitk.GetArrayFromImage(img).astype(np.float64)
    features: Dict[str, Any] = {
        'image_path': image_path,
        'min': np.min(stats_arr),
        'max': np.max(stats_arr),
        'mean': np.mean(stats_arr),
        'std': np.std(stats_arr),
    }

    mask = make_mask_from_image(img, threshold=mask_threshold)
    results = extractor.execute(img, mask)

    for k, v in results.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            features[k] = float(v)
    return features


def process_mip_dataset(root: str, out_dir: str, pattern: str, binWidth: float = 0.25, mask_threshold: float = 1e-6):
    os.makedirs(out_dir, exist_ok=True)
    subjects = discover_subject_folders(root)
    
    extractor = build_2d_extractor(binWidth=binWidth)

    rows = []
    for subj in tqdm(subjects, desc='Subjects', position=0, leave=True):
        subj_name = os.path.basename(subj)
        mips = find_mip_files(subj, pattern)
        for m in mips:
            feats = extract_features_from_mip(m, extractor, mask_threshold=mask_threshold)
            feats['subject'] = subj_name
            feats['file'] = os.path.basename(m)
            feats['path'] = m
            rows.append(feats)

    if os.path.isdir(out_dir):
        output_file = os.path.join(out_dir, 'mip_radiomics.csv')
    else:
        output_file = out_dir

    pd.DataFrame(rows).to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract radiomics features from 2D MIP images')
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--pattern', type=str, default='mip')
    parser.add_argument('--binwidth', type=float, default=0.25)
    parser.add_argument('--mask-threshold', type=float, default=1e-6)
    args = parser.parse_args()

    process_mip_dataset(
        args.input, args.output, 
        binWidth=args.binwidth, 
        pattern=args.pattern,
        mask_threshold=args.mask_threshold
    )
