#!/usr/bin/env python3
"""
Extracteur de métadonnées DICOM -> CSV (par patient)

But: prend le même repo DICOM que ton pipeline PET/CT et produit, pour chaque patient (ou dossier étude),
un fichier CSV listant une ligne par série DICOM avec toutes les métadonnées utiles pour
l'organisation multi-centre et la harmonisation (manufacturier, modèle, params acquisition, âge, poids,...).

Usage:
    python dicom_metadata_to_csv.py --input /chemin/vers/dicom_root --output /chemin/vers/out_csv

Dépendances:
    pip install pydicom pandas tqdm

Sortie:
    Pour chaque sous-dossier immédiat du root (patient/étude), on écrit `<out>/<folder.name>_metadata.csv`.
    Chaque CSV a une ligne par série et colonnes standardisées.

Remarques:
- Le script essaie d'être robuste: lit seulement l'entête DICOM (stop_before_pixels=True) sauf pour
  quelques calculs (extraction de positions pour vérifier le slice spacing).
- Certaines valeurs peuvent être absentes selon les vendors/scanners ; le script laisse des champs vides.
"""
import os
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------- Fonctions utilitaires -------------------------

def safe_get(ds, attr, default=None):
    """Retourne une valeur DICOM si présente, tente cast en str."""
    try:
        val = getattr(ds, attr, default)
        if val is None:
            return default
        # sequences -> keep count or repr
        if isinstance(val, pydicom.sequence.Sequence):
            return str(len(val))
        return str(val)
    except Exception:
        return default


def read_first_file_of_series(series_files):
    """Lit le premier fichier DICOM d'une série (header seulement)."""
    if not series_files:
        return None
    try:
        ds = pydicom.dcmread(series_files[0], stop_before_pixels=True)
        return ds
    except Exception:
        # essayer d'autres fichiers
        for f in series_files[1:10]:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
                return ds
            except Exception:
                continue
    return None


def get_series_file_list(subject_path: Path, series_id):
    """Retourne la liste de fichiers pour une SeriesInstanceUID donnée (Simple scan)."""
    files = []
    for p in subject_path.rglob('*'):
        if p.is_file():
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True)
                if getattr(ds, 'SeriesInstanceUID', None) == series_id:
                    files.append(str(p))
            except Exception:
                continue
    return sorted(files)


def check_slice_sampling(file_names):
    """(copié/réduit) Vérifie uniformité du spacing Z et retourne quelques stats.
    Retour: dict
    """
    positions = []
    iops = []
    for fn in file_names:
        try:
            ds = pydicom.dcmread(fn, stop_before_pixels=True)
            ipp = getattr(ds, 'ImagePositionPatient', None)
            iop = getattr(ds, 'ImageOrientationPatient', None)
            if ipp is None or iop is None:
                continue
            positions.append(np.array(ipp, dtype=float))
            iops.append(tuple([float(x) for x in iop]))
        except Exception:
            continue
    if len(positions) < 2:
        return {'n_slices': len(positions), 'status': 'not_enough_slices'}
    ref_iop = np.array(iops[0])
    row = ref_iop[0:3]
    col = ref_iop[3:6]
    normal = np.cross(row, col)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    proj = [float(np.dot(p, normal)) for p in positions]
    proj_sorted = np.sort(np.array(proj))
    diffs = np.diff(proj_sorted)
    positive_diffs = diffs[diffs > 1e-6]
    if len(positive_diffs) == 0:
        return {'n_slices': len(positions), 'status': 'identical_positions'}
    median = float(np.median(positive_diffs))
    mx = float(np.max(positive_diffs))
    mn = float(np.min(positive_diffs))
    std = float(np.std(positive_diffs))
    max_rel = float(np.max(np.abs(positive_diffs - median) / (median + 1e-12)))
    gap_factor = 1.5
    gaps = [(int(i), float(v), float(v/median)) for i, v in enumerate(positive_diffs) if v > gap_factor*median]
    status = 'ok' if (max_rel < 0.2 and len(gaps) == 0) else 'non_uniform'
    return {
        'n_slices': len(positions),
        'median_spacing': median,
        'min_spacing': mn,
        'max_spacing': mx,
        'std_spacing': std,
        'max_rel_dev': max_rel,
        'n_gaps': len(gaps),
        'gaps': gaps,
        'status': status
    }

# ------------------------- Champs DICOM à extraire -------------------------

# Liste de tags/attributs DICOM (noms d'attributs pydicom) qu'on extrait par défaut
COMMON_TAGS = [
    'PatientID', 'PatientAge', 'PatientSex', 'PatientWeight', 'PatientSize', 'StationName',
    'StudyInstanceUID', 'SeriesInstanceUID', 'StudyDescription', 'SeriesDescription', 'ProtocolName',
    'BodyPartExamined', 'Modality', 'SOPClassUID', 'SOPInstanceUID', 'FrameOfReferenceUID',
    'SeriesNumber', 'InstanceNumber', 'AcquisitionDate', 'AcquisitionTime', 'StudyDate', 'StudyTime',
    'KVP', 'XRayTubeCurrent', 'Exposure', 'ExposureTime', 'ConvolutionKernel', 'ReconstructionDiameter',
    'Rows', 'Columns', 'PixelSpacing', 'SliceThickness', 'SpacingBetweenSlices',
    'Units'
    # PET specific
    # 'RadiopharmaceuticalInformationSequence', 'RadionuclideCodeSequence',
]

# Certaines valeurs utiles sont dans les sequences -> on traite séparément

# ------------------------- Extraction métadonnées série -------------------------

def extract_series_metadata(series_id, series_files, include_sampling_check=False):
    """Construit un dict de métadonnées pour une série (une ligne du CSV)."""
    ds = read_first_file_of_series(series_files)
    out = {
        'SeriesInstanceUID': series_id, 
        'n_files': series_files.__len__()
    }
    if ds is None:
        out['read_error'] = True
        return out
    
    # Champs simples
    for tag in COMMON_TAGS:
        try:
            val = getattr(ds, tag, None)
            # pour sequences on garde info simplifiée
            if val is not None:
                if isinstance(val, pydicom.sequence.Sequence):
                    out[tag] = str(val.__len__())
                else:
                    out[tag] = str(val)
            else:
                out[tag] = None
        except Exception:
            out[tag] = None

    # Détails constructeur/modèle
    out['Manufacturer'] = safe_get(ds, 'Manufacturer', '')
    out['ManufacturerModelName'] = safe_get(ds, 'ManufacturerModelName', '')
    out['SoftwareVersions'] = safe_get(ds, 'SoftwareVersions', '')

    # Dates/temps d'acquisition normalisés
    out['AcquisitionDateTime'] = ''
    try:
        date = getattr(ds, 'AcquisitionDate', None) or getattr(ds, 'StudyDate', None)
        time = getattr(ds, 'AcquisitionTime', None) or getattr(ds, 'StudyTime', None)
        if date and time:
            out['AcquisitionDateTime'] = f"{date}T{time}"
    except Exception:
        pass

    # Patient info
    out['PatientAge'] = safe_get(ds, 'PatientAge', '')
    out['PatientWeight'] = safe_get(ds, 'PatientWeight', '')
    out['PatientSex'] = safe_get(ds, 'PatientSex', '')

    rphs = getattr(ds, 'RadiopharmaceuticalInformationSequence', None)

    if rphs and len(rphs) > 0:
        # Initialisation de listes pour stocker les valeurs de CHAQUE traceur
        list_names = []
        list_doses = []
        list_start_times = []
        
        # On boucle sur chaque élément 'item' dans la séquence 'rphs'
        for item in rphs:
            # 1. Extraction du Dose / Temps pour cet item spécifique
            dose = safe_get(item, 'RadionuclideTotalDose', 'NA')
            start_time = safe_get(item, 'RadiopharmaceuticalStartTime', 'NA')
            
            list_doses.append(str(dose))
            list_start_times.append(str(start_time))
            
            # 2. Extraction du NOM (Complexe car peut être texte ou code)
            # Priorité 1 : Le nom standardisé (Code Sequence)
            rph_code_seq = getattr(item, 'RadiopharmaceuticalCodeSequence', None)
            rph_name = "Unknown"
            
            if rph_code_seq and len(rph_code_seq) > 0:
                rph_name = safe_get(rph_code_seq[0], 'CodeMeaning', "Unknown")
            else:
                # Priorité 2 : Le nom texte libre
                rph_name = safe_get(item, 'Radiopharmaceutical', "Unknown")
                
            list_names.append(rph_name)

        # --- ENREGISTREMENT DANS OUT ---
        out['Radiopharmaceutical'] = " | ".join(list_names)
        out['RadionuclideTotalDose'] = " | ".join(list_doses)
        out['RadiopharmaceuticalStartTime'] = " | ".join(list_start_times)
        
        # Indicateur pratique pour filtrer plus tard
        out['TracerCount'] = len(rphs)

    else:
        # Cas où il n'y a pas d'info
        out['Radiopharmaceutical'] = None
        out['RadionuclideTotalDose'] = None
        out['TracerCount'] = 0

    # Calculs supplémentaires : n_slices, sampling_report
    if include_sampling_check:
        sampling = check_slice_sampling(series_files)
        out['n_slices'] = sampling.get('n_slices', None)
        out['median_spacing'] = sampling.get('median_spacing', None)
        out['max_rel_dev'] = sampling.get('max_rel_dev', None)
        out['n_gaps'] = sampling.get('n_gaps', None)
        out['sampling_status'] = sampling.get('status', None)

    return out

# ------------------------- Pipeline principale -------------------------

def collect_patient_metadata(subject_path: str, include_sampling_check: bool) -> pd.DataFrame:
    """Parcourt un dossier patient/étude et retourne un DataFrame contenant toutes les séries."""
    # identifier toutes les SeriesInstanceUID présentes
    subject = Path(subject_path)
    series_ids = set()
    for p in subject.rglob('*'):
        if p.is_file():
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True)
                sid = getattr(ds, 'SeriesInstanceUID', None)
                if sid:
                    series_ids.add(sid)
            except Exception:
                continue
    series_ids = sorted(list(series_ids))

    records = []
    for sid in series_ids:
        files = get_series_file_list(subject, sid)
        rec = extract_series_metadata(sid, files, include_sampling_check=include_sampling_check)
        records.append(rec)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    # Normaliser certaines colonnes si besoin
    return df


def process_subjects_directory(root_path: str, out_root: str, include_sampling_check: bool):
    root = os.path.abspath(root_path)
    out  = os.path.abspath(out_root)

    if not os.path.exists(out):
        os.makedirs(out)

    subjects_list = sorted(os.listdir(root))[:] # here to select n first patients

    for idx, subject in enumerate(subjects_list):
        subject_path = os.path.join(root, subject)
        if not os.path.isdir(subject_path):
            logging.warning(f'Skipping non-directory item: {subject_path}')
            continue

        total_subjects = subjects_list.__len__()
        progress = (idx + 1) / total_subjects * 100
        logging.info(f"Processing patient/study folder: {subject}; {idx + 1}/{total_subjects} [{progress:.2f}%]")
        df = collect_patient_metadata(subject_path, include_sampling_check=include_sampling_check)
        if df.empty:
            logging.warning(f"No series metadata for {subject}")
            continue

        csv_name = os.path.join(out, "{}.csv".format(subject))
        df.to_csv(csv_name, index=False)
        logging.info(f"Wrote metadata CSV: {csv_name} (rows={df.__len__()})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract DICOM metadata per patient into CSV files')
    parser.add_argument('--input', '-i', required=True, help='Root DICOM folder (one subfolder per patient/study)')
    parser.add_argument('--output', '-o', required=True, help='Output folder for CSV files')
    parser.add_argument('--include-sampling-check', action='store_true',
                        help='Include slice sampling uniformity check in the metadata')
    args = parser.parse_args()

    process_subjects_directory(args.input, args.output, args.include_sampling_check)
