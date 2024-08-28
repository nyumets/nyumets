import os
import glob
import datetime
import numpy as np
import pandas as pd

import nibabel as nib
import pydicom
import dicom_numpy
from dicompylercore import dicomparser
from rt_utils import RTStructBuilder

from utils import utils, mri_utils
from config import config


def match_mri_dir(mri_dirs, rtstruct_path):
    """ Match MRI directory to RTSTRUCT file.
    """
    mri_path = None
    while mri_path == None:
        for mri_dir in mri_dirs:
            try:
                rtstruct = RTStructBuilder.create_from(
                              dicom_series_path=mri_dir, 
                              rt_struct_path=rtstruct_path
                           )
                return mri_dir
            except:
                continue
    return mri_dir


def get_tumor_metadata(rtstruct_path):
    """ Returns list of tumor names (indicated by type == 'NTV' in the DICOM
    metadata), datetime.
    """
    tumor_names = []
    rtstruct_datetime = datetime.time()
    
    try:
        dp = dicomparser.DicomParser(rtstruct_path)
        structures_dict = dp.GetStructures()

        for k, v in structures_dict.items():
            if v['type'] == 'PTV':
                tumor_names.append(v['name'])
                
        rtstruct_datetime = dp.GetSeriesDateTime()['date'] + dp.GetSeriesDateTime()['time']
        rtstruct_datetime = datetime.datetime.strptime(rtstruct_datetime, '%Y%m%d%H%M%S')

    except AttributeError:
        tumor_names = []
    
    return tumor_names, rtstruct_datetime


def extract_segmentations(rtstruct_path, mri_dir, tumor_names):
    """ Extracts segmentations from RTSTRUCT, combines all tumor segmentations,
    and returns a numpy array.
    """
    
    segmentations = np.array([])
    
    # Load RTSTRUCT
    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=mri_dir, 
        rt_struct_path=rtstruct_path
    )
    
    # Iterate through tumor names to get all segmentations
    for tumor_name in tumor_names:
        try:
            mask_3d = rtstruct.get_roi_mask_by_name(tumor_name)
        except AttributeError:
            continue

        if segmentations.any():
            # get intersection of tumor segmentations
            segmentations = np.logical_or(mask_3d, segmentations)
        else:
            segmentations = mask_3d
    
    return segmentations


def dcm2array(datasets, logger):
    """ Coverts a list of DICOM files into a numpy array. Chooses a series
    description of either T1, T2, or MPRAGE.
    """

    # Get the sequence descriptions for each dicom
    SeriesDescriptions = [ds.SeriesDescription for ds in datasets]

    # list of unique series in accession number
    unique_series = sorted(utils.unique(SeriesDescriptions))
    unique_series_lower_all = [x.lower() for x in unique_series if isinstance(x, str)]
    
    # choose a study description to pull (T1, T2, or MPRAGE)
    if len(unique_series) != 1:
        logger.warning("More than 1 MRI sequence found. Choosing the first with 'T1', 'T2', or 'MPR'")
        unique_series_lower = [s for s in unique_series_lower_all if not any(xs in s for xs in ["cor", "sag", "scout", "t20"])]
        best_series = [s for s in unique_series_lower if ("t1" in s) or ("t2" in s) or ("mpr" in s)]
        if len(best_series) > 0:
            chosen_series = best_series[0]
        else:
            logger.error('No appropriate MRI sequence found.')
            nii_outfile = None
   
    else:
        chosen_series = unique_series_lower_all[0]

    logger.info(f'Choosing MRI sequence: {chosen_series}')
    unique_series_index = unique_series_lower_all.index(chosen_series)
    
    # Combine DICOM slices
    all_dcm_in_series = [ds for ds in datasets if ds.SeriesDescription == unique_series[unique_series_index]]
    voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(all_dcm_in_series)
    
    return voxel_ndarray, all_dcm_in_series, ijk_to_xyz, chosen_series


def preprocess_rt(acc_dir, logger):
    """ Main preprocessing function for RTSTRUCT and associated MRI DICOM.
    """
    logger.info(f"Accession dir: {acc_dir}")

    out_mri_path = acc_dir + 'nifti/'

    if not os.path.exists(out_mri_path):
        try:
            os.mkdir(out_mri_path)
        except PermissionError as e:
            error = e
            logger.error(error)
            rows = [[acc_dir, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, error,
                    np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan]]
            return rows

    # Get MRI directory(s)
    mri_dirs = []
    for dcm_dir in os.listdir(acc_dir):
        if any(mri_key in dcm_dir for mri_key in config.MRI_PATH_KEYWORDS):
            mri_dirs.append(acc_dir + dcm_dir)

    logger.info(f'Found MRI directories: {mri_dirs}')

    # Get RTSTRUCT file(s)
    rtstruct_files = []
    for dcm_dir in os.listdir(acc_dir):
        if 'RTSTRUCT' in dcm_dir:
            rtstruct_files.append(glob.glob(acc_dir + dcm_dir + '/' + "*")[0])
        else:
            for r, d, f in os.walk(acc_dir + dcm_dir):
                for file in f:
                    if 'RTSTRUCT' in file:
                        rtstruct_files.append(acc_dir + dcm_dir + '/' + file)

    logger.info(f'Found RTSTRUCT DICOMs: {rtstruct_files}')

    # Initiate variables for metadata
    segmentations = np.array([])
    rtstruct_datetime = np.nan
    rtstruct_outfile = os.path.join(out_mri_path, 'segmentation_rt.nii.gz')
    rtmri_outfile = os.path.join(out_mri_path, 'rtMRI.nii.gz')
    rows = []
    error = np.nan
    scanner = np.nan
    patient_name = np.nan
    patient_id = np.nan
    patient_dob = np.nan
    mri_datetime = np.nan
    chosen_series = np.nan
    old_spacing = np.nan
    final_spacing = np.nan

    # If nothing found
    if len(mri_dirs) == 0 or len(rtstruct_files) == 0:
        rows = [[acc_dir, np.nan, rtstruct_files, mri_dirs, rtstruct_datetime, 
                 rtstruct_outfile, rtmri_outfile, error, scanner, patient_id,
                 patient_name, patient_dob, mri_datetime, chosen_series,
                 old_spacing, final_spacing]]
        return rows
    
    # If 1 to 1 match with RTSTRUCT and MRI directory
    elif len(rtstruct_files) == 1 and len(mri_dirs) == 1:
        rtstruct_path = rtstruct_files[0]
        mri_dir = mri_dirs[0]
        tumor_names, rtstruct_datetime = get_tumor_metadata(rtstruct_path)
        try:
            segmentations = extract_segmentations(rtstruct_path, mri_dir, tumor_names)
        except Exception as e:
            error = e
            logger.error(error)
        rows = [[acc_dir, t, rtstruct_path, mri_dir, rtstruct_datetime, 
                 rtstruct_outfile, rtmri_outfile, error] for t in tumor_names]

    else:
        # Iterate through RTSTRUCT files to get all tumor names and 
        # match MRI paths
        tumor_dict = {}
        rtstruct2mri = {}
        
        for i, rtstruct_path in enumerate(rtstruct_files):

            # Get all tumor segmentation names
            tumor_names, rtstruct_datetime = get_tumor_metadata(rtstruct_path)

            if len(tumor_names) > 0:
                # Match MRI directory to RTSTRUCT
                mri_path = match_mri_dir(mri_dirs, rtstruct_path)
                rtstruct2mri[rtstruct_path] = mri_path

                # Get all of the tumor metadata
                rows = []
                for t in tumor_names:
                    if t in tumor_dict.keys():
                        if (rtstruct_datetime - tumor_dict[t]['datetime']).days > 1:
                            # difference in RTSTRUCTs' datetime has been found
                            # TODO: add error; will need to deal with this manually
                            other_path = tumor_dict[t]['rtstruct_path']
                            error = f'DatetimeError: RTSTRUCTs have a difference of greater than 1 day'
                            logger.error(error)
                            
                        elif rtstruct_datetime < tumor_dict[t]['datetime']:
                            # if tumor has a newer segmentation,
                            # use that one instead
                            tumor_dict[t]['rtstruct_path'] = rtstruct_path
                            tumor_dict[t]['mri_path'] = mri_path
                            tumor_dict[t]['datetime'] = rtstruct_datetime

                    else:
                        # if no previous tumor of same name has been added
                        tumor_dict[t] = {}
                        tumor_dict[t]['rtstruct_path'] = rtstruct_path
                        tumor_dict[t]['mri_path'] = mri_path
                        tumor_dict[t]['datetime'] = rtstruct_datetime


        # Get all of the segmentations to be loaded for each RTSTRUCT
        rtstruct2tumor = {}
        for t in tumor_dict.keys():
            rtstruct_path = tumor_dict[t]['rtstruct_path']
            if rtstruct_path not in rtstruct2tumor.keys():
                rtstruct2tumor[rtstruct_path] = [t]
            else:
                rtstruct2tumor[rtstruct_path].append(t)

        # Load all segmentations to be processed
        for rtstruct_path, tumor_names in rtstruct2tumor.items():
            mri_dir = rtstruct2mri[rtstruct_path]
            try:
                rtstruct_segmentations = extract_segmentations(rtstruct_path, mri_dir, tumor_names)
            except Exception as e:
                error = e
                logger.error(error)
                rtstruct_segmentations = np.array([])

            if len(mri_dirs) == 1:
                if segmentations.any():
                    # get union of segmentations if multiple
                    segmentations = np.logical_or(segmentations, rtstruct_segmentations)
                else:
                    segmentations = rtstruct_segmentations

            else:
                if segmentations.any():
                    if mri_dir != baseline_mri:
                        # Register segmentations to baseline MRI
                        # TODO: for now, add error message as this is not an 
                        # issue with the current data
                        error = f'MRIError: More than 1 MRI with segmentations'
                        logger.error(error)

                    segmentations = np.logical_or(segmentations, rtstruct_segmentations)

                else:
                    segmentations = rtstruct_segmentations
                    baseline_mri = mri_dir
                    baseline_tumor_names = tumor_names

        for t, d in tumor_dict.items():
            row = [acc_dir, t, d['rtstruct_path'], d['mri_path'], d['datetime'],
                   rtstruct_outfile, rtmri_outfile, error]
            rows.append(row)
    
    # Preprocess segmentations
    if segmentations.any():

        # Concatenate segmentations and save as nifti
        segmentations = segmentations.astype(float)
        segmentations = np.swapaxes(segmentations,0,1)
        dcm_filelist = glob.glob(mri_dir + '/*.dcm')
        
        # all dcm files in a single accession number
        datasets = [pydicom.read_file(f) for f in dcm_filelist]
        mri_ndarray, all_dcm_in_series, affine, chosen_series = dcm2array(datasets, logger)

        # gather dicom metadata
        ds = datasets[0]
        scanner = ds.Manufacturer
        patient_name = ds.PatientName
        patient_id = ds.PatientID
        patient_dob = ds.PatientBirthDate
        mri_datetime = ds.StudyDate + ds.StudyTime
        #mri_datetime = datetime.datetime.strptime(mri_datetime, '%Y%m%d%H%M%S')

        # resample both to 1 mm^3
        spacing = map(float, (list(all_dcm_in_series[0].PixelSpacing) + [float(all_dcm_in_series[0].SliceThickness)]))
        final_spacing = np.array([1.,1.,1.])
        
        if not (np.array(list(spacing)) == final_spacing).all():
            old_spacing = np.array(list(spacing))
            logger.warning(f'Spacing is {old_spacing}, need to resample')
            mri_ndarray, old_spacing, new_spacing = mri_utils.resample_dicom(mri_ndarray, all_dcm_in_series)
            segmentations, old_spacing, new_spacing = mri_utils.resample_dicom(segmentations, all_dcm_in_series)
            logger.warning(f'New spacing is {new_spacing}')

            # segmentations must be binary after resampling
            # using arbitrary threshold of 0.0001
            segmentations[segmentations >= 0.0001] = 1.
            segmentations[segmentations < 0.0001] = 0.
        else:
            new_spacing = final_spacing

        for row in rows:
            row.extend([scanner, patient_id, patient_name, patient_dob, 
                        mri_datetime, chosen_series, old_spacing, new_spacing])

        # save both to nifti
        mri_nii = nib.Nifti1Image(mri_ndarray, affine=affine)
        rtstruct_nii = nib.Nifti1Image(segmentations, affine=affine)
        nib.save(mri_nii, rtmri_outfile)
        nib.save(rtstruct_nii, rtstruct_outfile)
    
    return rows
