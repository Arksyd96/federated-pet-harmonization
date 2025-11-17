#!/usr/bin/env python3
"""
Convertit des séries DICOM (CT, PET, EARL PET) en NIfTI en utilisant PyDicom + SimpleITK.
- Préserve spacing, origin et direction.
- Convertit les PET en SUV (estimation basée sur les champs DICOM standards si présents).

Usage:
python pet_ct_to_nifti.py --input /chemin/vers/dataset --output /chemin/vers/output


Dépendances:
pip install pydicom SimpleITK numpy tqdm


Notes:
- Le calcul de SUV est effectué quand les champs DICOM nécessaires sont présents:
PatientWeight (0010,1030), RadiopharmaceuticalInformationSequence (0054,0016)
avec RadionuclideTotalDose et RadionuclideHalfLife, et un temps d'injection.
- Le script applique RescaleSlope/Intercept si présents.
- Important: les DICOM PET peuvent stocker différentes unités/calibrations. Vérifiez les résultats sur un cas connu.
"""

import os
import sys
import argparse
import logging
import math
from datetime import datetime, timedelta


import numpy as np
import pydicom
import SimpleITK as sitk
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------- Utilities ------------------------
def parse_dicom_time(time_str):
    """Parse un temps DICOM (HHMMSS.frac) en datetime.time. Retourne None si invalide."""
    if not time_str:
        return None
    try:
        # time_str peut être 'HHMMSS' ou 'HHMMSS.FFFFFF'
        if '.' in time_str:
            base, frac = time_str.split('.')
        else:
            base, frac = time_str, '0'
        hour = int(base[0: 2])
        minute = int(base[2: 4]) if base.__len__() >= 4 else 0
        second = int(base[4: 6]) if base.__len__() >= 6 else 0
        micro = int((frac + '000000')[:6])
        return datetime(1900, 1, 1, hour, minute, second, micro)
    except Exception:
        return None


def compute_decay_corrected_injected_activity(ds: pydicom.Dataset) -> tuple:
    """Retourne l'activité injectée corrigée au temps d'acquisition (en Bq) si possible.
    Retourne (decayed_activity_Bq, details_dict) ou (None, details)
    """
    details = {}
    try:
        rph = ds.RadiopharmaceuticalInformationSequence[0]
    except Exception:
        details['error'] = 'RadiopharmaceuticalInformationSequence not found.'
        logging.warning(details['error'])
        return None, details
    
    # Radionuclide total dose (Bq)
    injected = float(getattr(rph, 'RadionuclideTotalDose', None))
    details['InjectedActivity_Bq'] = injected

    # half-life (s)
    half_life = float(getattr(rph, 'RadionuclideHalfLife', None))
    details['RadionuclideHalfLife_s'] = half_life

    # injection time
    injection_time_str = getattr(rph, 'RadiopharmaceuticalStartTime', None)
    details['InjectionTime_str'] = injection_time_str

    # acquisition time
    acquisition_time_str = getattr(ds, 'AcquisitionTime', None) or getattr(ds, 'SeriesTime', None) or getattr(ds, 'StudyTime', None)
    details['AcquisitionTime_str'] = acquisition_time_str

    if injected is None or half_life is None or injection_time_str is None or acquisition_time_str is None:
        details['error'] = 'Missing required DICOM tags for activity decay correction. Check details.'
        logging.warning(details['error'])
        return None, details
    
    # parse times
    injection_dt = parse_dicom_time(injection_time_str)
    acquisition_dt = parse_dicom_time(acquisition_time_str)
    if injection_dt is None or acquisition_dt is None:
        details['error'] = 'Failed to parse injection or acquisition time.'
        logging.warning(details['error'])
        return None, details
    
    # compute delta in seconds; time may wrap midnight -> handle
    delta_seconds = (acquisition_dt - injection_dt).total_seconds()
    if delta_seconds < 0:
        delta_seconds += 24 * 3600
    details['TimeSinceInjection_s'] = delta_seconds

    # decay correction
    decay_factor = np.exp(-np.log(2) * (delta_seconds / half_life))
    decayed_activity = injected * decay_factor
    details['decayed_activity_Bq'] = decay_factor
    return decayed_activity, details


def save_image_nifti(sitk_image: sitk.Image, out_path: str, extra_metadata: dict = None):
    """Sauvegarde une image SimpleITK en NIfTI avec des métadonnées supplémentaires."""
    if extra_metadata:
        for key, value in extra_metadata.items():
            sitk_image.SetMetaData(str(key), str(value))

    sitk.WriteImage(sitk_image, out_path)
    logging.info(f'Saved NIfTI image to: {out_path}')


def compute_suv_image(sitk_image: sitk.Image, dicom_filenames: list) -> tuple:
    """Calcule une image SUV à partir d'une image SimpleITK lue depuis une série DICOM.
    dicom_filenames: liste de fichiers (on utilisera le premier pour extraire les tags).
    Retourne (sitk_suv_image, metadata_dict)
    """
    ds = pydicom.dcmread(dicom_filenames[0], stop_before_pixels=True)
    metadata = {}

    # rescaling
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    metadata['RescaleSlope'] = slope
    metadata['RescaleIntercept'] = intercept

    # patient weight
    patient_weight_kg = getattr(ds, 'PatientWeight', None)
    if patient_weight_kg is None:
        try:
            sfg = ds.SharedFunctionalGroupsSequence[0]
            patient_weight_kg = float(sfg.PatientWeight)
        except Exception:
            logging.warning('Patient weight not found in DICOM tags.')
            return None, metadata
        
    patient_weight_kg = float(patient_weight_kg)
    metadata['PatientWeight_kg'] = patient_weight_kg

    # injected activity and half-life
    decayed_activity, details = compute_decay_corrected_injected_activity(ds)
    metadata.update(details)

    # sitk to numpy and rescale
    arr = sitk.GetArrayFromImage(sitk_image).astype(np.float64)
    arr = arr * slope + intercept
    metadata['applied_unit_assumption'] = 'Bq/mL'

    if patient_weight_kg is None or decayed_activity is None:
        metadata['SUV_computation'] = False
        logging.warning('Insufficient data for SUV computation (weight or activity missing).')
        return None, metadata
    
    # SUV = Bq/mL * patient_weight_kg * 1000 / decayed_activity (Bq)
    factor = (patient_weight_kg * 1000.0) / decayed_activity
    metadata['SUV_factor'] = factor
    suv_arr = arr * factor

    # creating sitk image
    suv_image = sitk.GetImageFromArray(suv_arr)
    suv_image.SetDirection(sitk_image.GetDirection())
    suv_image.SetOrigin(sitk_image.GetOrigin())
    suv_image.SetSpacing(sitk_image.GetSpacing())

    metadata['SUV_computation'] = True
    return suv_image, metadata


def process_series_folder(series_path: str, out_subject_path: str, modality: str):
    logging.info(f'-- Processing series folder: {series_path.split(os.sep)[-1]}; [md: {modality}]')

    try:
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(series_path) or []
    except Exception as e:
        series_IDs = []

    if not series_IDs:
        logging.warning(f'No DICOM series found in folder: {series_path}')
        return
    
    for series_id in series_IDs:
        filenames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(series_path, series_id)
        try:
            ds = pydicom.dcmread(filenames[0], stop_before_pixels=True)
        except Exception as e:
            logging.error(f'Failed to read DICOM file: {filenames[0]}. Error: {str(e)}')
            continue

        md = getattr(ds, 'Modality', '').upper()
        sdesc = getattr(ds, 'SeriesDescription', '')
        suid = getattr(ds, 'SeriesInstanceUID', series_id)

        out_name = f"{md}_{series_path.split(os.sep)[-1]}.nii.gz"
        out_path = os.path.join(out_subject_path, out_name)

        # read the image via simpleITK
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(filenames)
        image = reader.Execute()

        # compute SUV for PET and EARL images
        if modality == 'PET' or modality == 'EARL':
            logging.info(f'-- Computing SUV for {modality} series at: {series_id}')
            suv_image, meta = compute_suv_image(image, filenames)
            if suv_image is not None:
                logging.info(f'-- Saving SUV NIfTI to: {out_path} (with metadata)')
                save_image_nifti(suv_image, out_path, extra_metadata=meta)
            else:
                logging.warning(f'-- SUV computation failed for {modality} series: {series_id}.')
        
        else: # for CT
            logging.info(f'-- Saving {modality} NIfTI to: {out_path}')
            save_image_nifti(image, out_path, extra_metadata={
                'Modality': md,
                'SeriesDescription': sdesc
            })


def process_subject(subject_path: str, out_subject_path: str, curr_idx: int = 1, total_subjects: int = 1):
    logging.info('Processing subject: {} | {} / {}'.format(subject_path.split('/')[-1], curr_idx, total_subjects))

    immediate_subdirs = [p for p in os.listdir(subject_path)]
    if all(sd.lower().endswith('.dcm') for sd in immediate_subdirs):
        logging.info(f'Detected DICOM files directly under subject folder: {subject_path}')
        process_series_folder(subject_path, out_subject_path, modality=None)
        return

    for subdir in immediate_subdirs:
        modality = None
        if subdir.lower().startswith('ct'): modality = 'CT'
        if subdir.lower().startswith('tep'): modality = 'PET'
        if subdir.lower().startswith('tep') and 'earl' in subdir.lower(): modality = 'EARL'
        if modality is None:
            logging.warning(f'Skipping unknown modality folder: {subdir}')
            continue

        series_path = os.path.join(subject_path, subdir)
        if not os.path.isdir(series_path):
            continue

        process_series_folder(series_path, out_subject_path, modality=modality)
    return


def process_subjects_directory(root_path: str, out_root: str):
    root = os.path.abspath(root_path)
    out  = os.path.abspath(out_root)

    if not os.path.exists(out):
        os.makedirs(out)

    subjects_list = sorted(os.listdir(root))[:] # debugging

    for idx, subject in enumerate(subjects_list):
        subject_path = os.path.join(root, subject)
        if not os.path.isdir(subject_path):
            logging.warning(f'Skipping non-directory item: {subject_path}')
            continue

        out_subject_path = os.path.join(out, subject)
        if not os.path.exists(out_subject_path):
            os.makedirs(out_subject_path)

        process_subject(subject_path, out_subject_path, 
                        curr_idx=idx + 1, total_subjects=subjects_list.__len__())
        

# ----------------------- CLI ------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert DICOM CT/PET/EARL to NIfTI and compute PET SUV')
    parser.add_argument('--input', type=str, required=True, help='Path to input DICOM dataset [folder with subfolders per series]')
    parser.add_argument('--output', type=str, required=True, help='Path to output folder for NIfTI files [Same structure as input]')
    args = parser.parse_args()

    process_subjects_directory(args.input, args.output)


