import os
import numpy as np
from scipy import ndimage
import glob
import pydicom  # https://pydicom.github.io/
import dicom_numpy  # https://dicom-numpy.readthedocs.io/en/latest/
import numpy as np
import nibabel as nib  # https://nipy.org/nibabel/
import pandas as pd
import datetime

from config import config
from utils import utils


def resample_dicom(image, scan, spacing=[1.,1.,1.]):
    # Credit: https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
    
    old_spacing = map(float, (list(scan[0].PixelSpacing) + [float(scan[0].SliceThickness)]))
    old_spacing = np.array(list(old_spacing))

    resize_factor = old_spacing / spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor

    image = ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, old_spacing, new_spacing


def preprocess_mri(acc_dir, spacing, logger):
    """ Preprocess MRI DICOMs within an accession directory.
    """
    acc_num = acc_dir.split('/')[-2]
    logger.info(f"Processing MRI files under accession number: {acc_num}")
    logger.info(f"Accession dir: {acc_dir}")
    row = []

    # Read in dicom files
    dcm_files = glob.glob(acc_dir + "*.dcm")
    datasets = [pydicom.read_file(f) for f in dcm_files]  # all dcm files in a single accession number

    series_names = ["T1", "CT1", "T2", "FLAIR", "DIFFUSION"]

    if len(datasets) == 0:
        error_message = f'No DICOMs found: {acc_dir}'
        logger.error(error_message)
        row.extend([acc_num, np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan])
        for name in series_names:
            row.extend([error_message])
            row.extend(empty_series)
        return row

    # Get scanner manufacturer
    ds = datasets[0]
    scanner = ds.Manufacturer
    mri_datetime = ds.StudyDate + ds.StudyTime
    mri_datetime = datetime.datetime.strptime(mri_datetime, '%Y%m%d%H%M%S')
    study_desc = ds.StudyDescription.lower()
    patient_name = ds.PatientName
    patient_id = ds.PatientID
    patient_dob = ds.PatientBirthDate
    empty_series = [np.array([np.nan,np.nan,np.nan]),
                    np.array([np.nan,np.nan,np.nan]), np.nan]

    if ('neuro' not in study_desc and 'brain' not in study_desc) or ('spine' in study_desc):
        error_message = f'StudyDescription not brain or neuro: {study_desc}'
        logger.error(error_message)
        row.extend([acc_num, scanner, mri_datetime, study_desc, patient_id,
                    patient_name, patient_dob])
        for name in series_names:
            row.extend([error_message])
            row.extend(empty_series)
        return row

    # Get the sequence descriptions for each dicom
    SeriesDescriptions = [ds.SeriesDescription for ds in datasets]
    unique_series = sorted(utils.unique(SeriesDescriptions))  # list of unique series in accession number

    # Eliminate and segment the matches into groups of sequences
    unique_series_lower_all = [x.lower() for x in unique_series if isinstance(x, str)]
    unique_series_lower = [s for s in unique_series_lower_all if not any(xs in s for xs in ["cor", "sag", "scout", "t20"])]
    t1 = [s for s in unique_series_lower if "t1" in s and not "post" in s]
    t2 = [s for s in unique_series_lower if "t2" in s]
    flair = [s for s in unique_series_lower if "flair" in s]
    ct1 = [s for s in unique_series_lower if any(xs in s for xs in ["gd", "gad", "post", "contrast", "mpr"])]
    diff = [s for s in unique_series_lower if any(xs in s for xs in ["diff"])]

    matched_names = [t1, ct1, t2, flair, diff]

    # Grab best match from each sequence
    best_match = {}
    for matches, name in zip(matched_names, series_names):
        best_match[name] = np.nan
        for match in matches:
            if match in config.BEST_MATCHES[name]:
                best_match[name] = match

    # for the metadata file
    row.extend([acc_num, scanner, mri_datetime, study_desc, patient_id,
                patient_name, patient_dob])

    for series_name, series_match in best_match.items():

        # Get the index of the match if it exists
        try:
            unique_series_index = unique_series_lower_all.index(series_match)
        except ValueError:
            logger.warning(f"No matched {series_name} series for {acc_num}")
            logger.warning(f"Series found: {unique_series_lower_all}")
            error_message = "No matched series"
            row.extend([error_message,  np.array([np.nan,np.nan,np.nan]), np.array([np.nan,np.nan,np.nan]), np.nan])
            continue

        # Combine the dicoms, resample to 1x1x1, create NIFTI image, and save
        try:
            # Find all dicoms with that matched series
            all_dcm_in_series = [ds for ds in datasets if ds.SeriesDescription == unique_series[unique_series_index]]
            
            # Combine, resample, create 3D NIFTI
            voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(all_dcm_in_series)
            voxel_ndarray, old_spacing, new_spacing = resample_dicom(voxel_ndarray, all_dcm_in_series, spacing)
            new_image = nib.Nifti1Image(voxel_ndarray, affine=ijk_to_xyz)

            # Save
            if not os.path.exists(acc_dir + 'nifti'):
                os.makedirs(acc_dir + 'nifti')

            image_outfile = acc_dir + f"nifti/{series_name}.nii.gz"
            nib.save(new_image, image_outfile)
            row.extend([series_match, old_spacing, new_spacing, image_outfile])

        except Exception as e:
            error_message = str(e)
            logger.warning(f"Error during combining/resampling {series_name} series for {acc_num}:\n{error_message}")
            row.extend([error_message, np.array([np.nan,np.nan,np.nan]), np.array([np.nan,np.nan,np.nan]), np.nan])
    
    return row

