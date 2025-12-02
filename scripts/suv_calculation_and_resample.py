"""
This module provides tools for converting DICOM images to NIfTI format, computing SUV
for PET and SPECT images and resamplimg images.
"""

import os
import sys
import logging
import datetime
from typing import Tuple

import pydicom
import numpy as np
import SimpleITK as sitk

from datetime import datetime


class NegativeDelay(Exception):
    """
    Exception to be raised when the computed delay between the radiopharmaceutical
    injection and the scan acquisition time is negative.
    """

def delay_between_injection_and_scan(file_dcm: pydicom.FileDataset):
    """
    Computes the delay between the radiopharmaceutical injection and the scan acquisition time,
    using the metadata from the DICOM file.

    Parameters:
        file_dcm (pydicom.FileDataset): A pydicom FileDataset object, 
        containing the metadata needed for SUV computation

    Returns:
        datetime.timedelta: The time difference between the injection and scan acquisition.
    """
    try :
        injection_date_time = datetime.strptime(
            file_dcm.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartDateTime, 
            '%Y%m%d%H%M%S')
    except ValueError:
        injection_date_time = datetime.strptime(
            file_dcm.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartDateTime,
            '%Y%m%d%H%M%S.%f')
    try : 
        scan_date_time = datetime.strptime(file_dcm.AcquisitionDate + file_dcm.AcquisitionTime,
                                            '%Y%m%d%H%M%S')
    except ValueError:
            scan_date_time = datetime.strptime(file_dcm.AcquisitionDate + file_dcm.AcquisitionTime,
                                                '%Y%m%d%H%M%S.%f')
    return scan_date_time - injection_date_time

def convert_to_BQML(image_array, file_dcm):
    volume = file_dcm.PixelSpacing[0] * file_dcm.PixelSpacing[1] * file_dcm.SpacingBetweenSlices
    rescale_slope = 10**9 / (16.49 * file_dcm.NumberOfFrames * 2 * file_dcm.RotationInformationSequence[0].ActualFrameDuration * volume * 10**(-3))
    return rescale_slope*image_array

def compute_suv(input_image :sitk.Image, file_dcm: pydicom.FileDataset):
    """
    Computes the SUV for the input image using metadata from the DICOM file.

    Parameters:
        input_image (sitk.Image): Input SimpleITK Image containing the activity.
        file_dcm (pydicom.FileDataset): Pydicom FileDataset containing the metadata needed for SUV 
        computation.

    Returns:
        sitk.Image: The computed SUV image.
    """

    image_array = sitk.GetArrayFromImage(input_image)

    units = file_dcm.get("Units", None)
    if units is None:
        image_array = convert_to_BQML(image_array, file_dcm)

    # Retrieving metadata for SUV computation
    patient_weight = float(file_dcm.PatientWeight) * 1000
    half_life = float(file_dcm.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
    injected_dose = float(file_dcm.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
    if file_dcm.Modality == "NM":
        injected_dose *=  1000000
    delay = delay_between_injection_and_scan(file_dcm).total_seconds()
    print(delay)
    if delay < 0:
        raise NegativeDelay("The delay between the radiopharmaceutical injection date and the scan "
                            "date is negative.This is not possible. There is an error in the "
                            "metadata.")

    decay = np.exp(-np.log(2) * ((delay) / half_life))
    injected_dose_decay = injected_dose * decay
    suv_image_array = image_array * patient_weight / injected_dose_decay

    suv_image = sitk.GetImageFromArray(suv_image_array)
    # Copying metadata (importantly: space metadata) to the new Image
    suv_image.CopyInformation(input_image)

    return suv_image

def resample_itk(input_image: sitk.Image, direction: Tuple, origin:Tuple, size:Tuple, spacing:Tuple, interpolator=sitk.sitkBSpline):
    """
    Resamples a SimpleITK Image towards a specified destination space.

    Parameters:
        input_image (sitk.Image): Input SimpleITK Image to be resampled.
        direction (Tuple): Destination direction.
        origin (Tuple): Destination origin.
        size (Tuple): Destination size.
        spacing (Tuple): Destination spacing.
        interpolator: SimpleITK interpolator to use. Default is sitk.sitkBSpline.

    Returns:
        sitk.Image: Resampled image.
    """
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(direction)
    resample.SetOutputOrigin(origin)
    resample.SetSize(size)
    resample.SetOutputSpacing(spacing)
    suv_image = resample.Execute(input_image)
    return suv_image

class MissingModalityError(Exception):
    """
    Exception to be raised when a modality is missing.
    """

class DicomToNifti:

    """
    Class for converting DICOM images to NIfTI format.
    """

    def __init__(
        self,
        ct_dicom_path: str,
        ct_nii_path: str,
        pet_dicom_path: str,
        pet_nii_path: str,
        spect_dicom_path: str,
        spect_nii_path: str
    ) -> None:

        """
        Initializes the DicomToNifti class with paths for DICOM and NIfTI files.

        Parameters:
            ct_dicom_path (str): Path to the CT DICOM files.
            ct_nii_path (str): Path to save the converted CT NIfTI file.
            pet_dicom_path (str): Path to the TEP DICOM files.
            pet_nii_path (str): Path to save the converted TEP NIfTI file.
            spect_dicom_path (str): Path to the SPECT DICOM files.
            spect_nii_path (str): Path to save the converted SPECT NIfTI file.
        """
        self.ct_dicom_path = ct_dicom_path
        self.ct_nii_path = ct_nii_path
        self.pet_dicom_path = pet_dicom_path
        self.pet_nii_path = pet_nii_path
        self.spect_dicom_path = spect_dicom_path
        self.spect_nii_path = spect_nii_path
        self.pt_new_size = None
        self.pt_new_orig= None
        self.pt_new_dir= None
        self.pt_spacing = None
        self.has_pet = False
        self.has_spect = False
        if (self.ct_dicom_path is None) or (self.ct_nii_path is None):
            raise MissingModalityError("Needs CT")
        if (self.pet_dicom_path is not None) and (self.pet_nii_path is not None):
            self.has_pet = True
        if (self.spect_dicom_path is not None) and (self.spect_nii_path is not None):
            self.has_spect = True
        if not(self.has_pet or self.has_spect):
            raise MissingModalityError("Needs at least a modality among PET and SPECT")
  
    def ct_dicom2nifti(self):

        """
        Converts CT DICOM files to NIfTI format and saves the image.
        """

        reader = sitk.ImageSeriesReader()
        ct_dicom_names = reader.GetGDCMSeriesFileNames(self.ct_dicom_path)
        reader.SetFileNames(ct_dicom_names)
        ct_image = reader.Execute()

        ct_image =resample_itk(ct_image, self.pt_new_dir, self.pt_new_orig,
                                 self.pt_new_size, self.pt_spacing)

        sitk.WriteImage(ct_image, self.ct_nii_path)

    def pet_dicom2nifti(self):
        """
        Converts TEP DICOM files to NIfTI format, computes the Standardized Uptake Value (SUV) for
        the PET image,resamples the image towards the destination space of CT, and saves the image.
        """

        reader = sitk.ImageSeriesReader()
        pet_dicom_names = reader.GetGDCMSeriesFileNames(self.pet_dicom_path)
        reader.SetFileNames(pet_dicom_names)
        pet_image = reader.Execute()

        file_dcm = pydicom.dcmread(os.path.join(self.pet_dicom_path,os.listdir(self.pet_dicom_path)[0]),
                                   stop_before_pixels=True)

        suv_image = compute_suv(pet_image, file_dcm)
        sitk.WriteImage(suv_image, self.pet_nii_path)

        self.pt_new_size = suv_image.GetSize()
        self.pt_new_orig= suv_image.GetOrigin()
        self.pt_new_dir= suv_image.GetDirection()
        self.pt_spacing = suv_image.GetSpacing()

    def spect_dicom2nifti(self):
        """
        Converts SPECT DICOM files to NIfTI format, computes the Standardized Uptake Value (SUV) 
        for the SPECT image,resamples the image towards the destination space of CT, and saves the 
        image.
        """

        reader = sitk.ImageFileReader()
        reader.SetFileName(str(os.path.join(self.spect_dicom_path, os.listdir(self.spect_dicom_path)[0])))
        sepct_image = reader.Execute()

        file_dcm = pydicom.dcmread(os.path.join(self.spect_dicom_path,os.listdir(self.spect_dicom_path)[0]),
                                   stop_before_pixels=True)

        suv_image = compute_suv(sepct_image, file_dcm)
        sitk.WriteImage(suv_image, self.spect_nii_path)

        self.pt_new_size = suv_image.GetSize()
        self.pt_new_orig= suv_image.GetOrigin()
        self.pt_new_dir= suv_image.GetDirection()
        self.pt_spacing = suv_image.GetSpacing()

if __name__ == "__main__":

    from pathlib import Path
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    database_path = Path("../datasets/EARL/")
    patients = ["019", "020", "021", "022", "023", "024", "025", "026", "027"]
    scans = ["PSMA", "C1"]

    for patient in patients:
        for scan in scans:
            logging.info("Calculation of SUV for the patient %s for the scan %s", patient, scan)

            modality = "PET" if scan in ["FDG", "PSMA"] else "SPECT"

            ct_dicom_path = database_path / f"patient_{patient}" / f"{scan}_{modality}" /"CT"
            functionnal_dicom_path = database_path / f"patient_{patient}" / f"{scan}_{modality}" / f"{modality}"
            ct_nifti_path = database_path / f"patient_{patient}" / f"{scan}_{modality}" / f"{scan}_CT_{patient}.nii.gz"
            functionnal_nifti_path = database_path / f"patient_{patient}" / f"{scan}_{modality}" / f"{scan}_{modality}_{patient}.nii.gz"


            if modality == "PET":
                dicom2nifti = DicomToNifti(ct_dicom_path, ct_nifti_path, functionnal_dicom_path, functionnal_nifti_path, None, None)
                dicom2nifti.pet_dicom2nifti()
                dicom2nifti.ct_dicom2nifti()
            else:
                dicom2nifti = DicomToNifti(ct_dicom_path, ct_nifti_path, None,  None, functionnal_dicom_path, functionnal_nifti_path)
                dicom2nifti.spect_dicom2nifti()
                dicom2nifti.ct_dicom2nifti()