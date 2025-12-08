#!/usr/bin/env python3
"""
Convertit des séries DICOM (CT, PET, EARL PET) en NIfTI en utilisant PyDicom + SimpleITK.
- Préserve spacing, origin et direction.
- Convertit les PET en SUV (estimation basée sur les champs DICOM standards si présents).
- Basée sur une structure de dossiers où chaque sujet a son propre dossier contenant des sous-dossiers
- Multi-threading pour accélérer le traitement.

Usage:
python pet_ct_to_nifti.py --input /chemin/vers/dataset --output /chemin/vers/output


Dépendances:
pip install pydicom SimpleITK numpy tqdm
"""

import os
import argparse
import logging
from datetime import datetime

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    
    # half-life (s)
    half_life = float(getattr(rph, 'RadionuclideHalfLife', None))
    details['RadionuclideHalfLife_s'] = half_life

    # Radionuclide total dose (Bq)
    injected = float(getattr(rph, 'RadionuclideTotalDose', None))
    details['InjectedActivity_Bq'] = injected

    # injection time
    injection_time_str = getattr(rph, 'RadiopharmaceuticalStartTime', None)
    details['InjectionTime_str'] = injection_time_str

    # acquisition time
    # acquisition_time_str = getattr(ds, 'AcquisitionTime', None)
    acquisition_time_str = getattr(ds, 'SeriesTime', None)
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


def compute_suv_image(sitk_image: sitk.Image, ds: pydicom.Dataset) -> tuple:
    """Calcule une image SUV à partir d'une image SimpleITK lue depuis une série DICOM.
    ds: metadata from one of the DICOM files in the series.
    Retourne (sitk_suv_image, metadata_dict)
    """
    metadata = {}

    assert getattr(ds, 'Units', None) == 'BQML', 'DICOM PET Units is not BQML, cannot compute SUV reliably.'

    # TODO: If Units are not BQML, implement the conversion function based
    # on the calibration factor and other parameters.

    # rescale metadata: we do not rescale as SimpleITK does it automatically
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

        sdesc = getattr(ds, 'SeriesDescription', '')
        suid = getattr(ds, 'SeriesInstanceUID', series_id)

        out_name = f"{modality}_{series_path.split(os.sep)[-1]}.nii.gz"
        out_path = os.path.join(out_subject_path, out_name)

        # read the image via simpleITK
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(filenames)
        image = reader.Execute()

        # compute SUV for PET and EARL images
        if modality == 'PET' or modality == 'EARL':
            suv_image, meta = compute_suv_image(image, ds)
            if suv_image is not None:
                save_image_nifti(suv_image, out_path, extra_metadata=meta)
            else:
                logging.warning(f'-- SUV computation failed for {modality} series: {series_id}.')
        
        else: # for CT
            save_image_nifti(image, out_path, extra_metadata={
                'Modality': modality,
                'SeriesDescription': sdesc
            })


def process_subject(subject_path: str, out_subject_path: str):
    immediate_subdirs = [p for p in os.listdir(subject_path)]

    if all(sd.lower().endswith('.dcm') for sd in immediate_subdirs):
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


def process_subjects_directory(root_path: str, out_root: str, num_workers: int = 1):
    root = os.path.abspath(root_path)
    out  = os.path.abspath(out_root)

    if not os.path.exists(out):
        os.makedirs(out)

    all_subjects = sorted([s for s in os.listdir(root) if os.path.isdir(os.path.join(root, s))])

    tasks = list()
    for subject in all_subjects:
        subject_path = os.path.join(root, subject)
        out_subject_path = os.path.join(out, subject)

        if not os.path.exists(out_subject_path):
            os.makedirs(out_subject_path)

        tasks.append((subject_path, out_subject_path))

    if num_workers is None or num_workers < 1:
        num_workers = max(1, multiprocessing.cpu_count() - 2)

    logging.info(f'Lancement du traitement sur {tasks.__len__()} avec {num_workers} workers... 🚀')
    logging.getLogger().setLevel(logging.WARNING)
    sitk.ProcessObject.SetGlobalWarningDisplay(False) # Annoying SimpleITK warnings, enable to debug

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_subject_wrapper, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing subjects'):
            result = future.result()
            if result is not None:
                tqdm.write(result)


def process_subject_wrapper(args):
    subject_path, out_subject_path = args
    
    try:
        process_subject(subject_path, out_subject_path)
        return None
    except Exception as e:
        return f"Erreur sur {subject_path}: {e}"

# ----------------------- CLI ------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert DICOM CT/PET/EARL to NIfTI and compute PET SUV')
    parser.add_argument('--input', '-i', type=str, required=True, 
                        help='Path to input DICOM dataset [folder with subfolders per series]')
    parser.add_argument('--output', '-o', type=str, required=True, 
                        help='Path to output folder for NIfTI files [Same structure as input]')
    parser.add_argument('--workers', '-w', type=int, default=None, # not required
                        help='Number of parallel workers (default: CPU count - 2)')
    args = parser.parse_args()

    process_subjects_directory(args.input, args.output, args.workers)


