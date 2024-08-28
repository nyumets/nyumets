import pandas as pd
import os
import numpy as np
import csv
import shutil
import math

import ants
from nipype.interfaces import fsl

from config import config
from utils import utils


def coregister_timepoint(row, logger, keep_old=True):
    accession_number = row[1].AccessionNumber
    if math.isnan(accession_number):
        accession_number = row[1].MRI_FileLocation.split('/')[-3].replace(" ", "_")
    else:
        accession_number = str(int(accession_number))
    
    scratch_superdir = os.path.join(config.SCRATCH_DIR, accession_number)
    if not os.path.exists(scratch_superdir):
        os.mkdir(scratch_superdir)
    
    scratch_outpath = None
    scratch_temppath = None

    logger.info(f'Processing accession number: {accession_number}')

    avail_sequences = []
    for col in row[1].index.tolist():
        if config.FILELOCATION_COL in col:
            if str(row[1][col]) != 'nan':
                avail_sequences.append(col.split('_')[0])

    num_avail_sequences = len(avail_sequences)

    outpath = None

    # Initialize the dictionaries to keep everything organized
    metadata_row = []
    orig_files = {}
    oriented_files = {}
    mat_files = {}
    reg_files = {}
    reference_file = None
    reference_sequence = None
    segmentation_rt = {}
    seg_filepath = None


    mrn = row[1]['PatientID']
    if math.isnan(mrn):
        mrn = row[1]['PatientID.1']
        if type(mrn) != str:
            mrn = str(mrn)
    else:
        mrn = str(int(mrn))
    
    pt_superdir = os.path.join(config.FINAL_DATASET_DIR, mrn)
    if not os.path.exists(pt_superdir):
        os.mkdir(pt_superdir)
    
    superdir = os.path.join(pt_superdir, accession_number)
    if not os.path.exists(superdir):
        os.mkdir(superdir)

    scratch_outpath = os.path.join(scratch_superdir, config.OUTDIR)
    scratch_temppath = os.path.join(scratch_superdir, config.TEMPDIR)
    
    outpath = os.path.join(superdir, config.OUTDIR)
    temppath = os.path.join(superdir, config.TEMPDIR)

    if not os.path.exists(scratch_outpath):
        os.mkdir(scratch_outpath)

    if not os.path.exists(scratch_temppath):
        os.mkdir(scratch_temppath)
    
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    if not os.path.exists(temppath):
        os.mkdir(temppath)
    
    # See what MRI and segmentation files exist
    for name in config.TARGET_FILES:
        if name in avail_sequences:
            name_fp = row[1][f'{name}_FileLocation']
            if 'MRI' in name:
                name_fp = name_fp.replace('  ', ' ')
                try:
                    temp_MRI_file = shutil.copy2(name_fp, scratch_temppath)
                    logger.info(f'Copied MRI from {name_fp} to {temp_MRI_file}')
                    orig_files[name] = temp_MRI_file
                except FileNotFoundError:
                    logger.error(f'No file found: {name_fp}')
                    num_avail_sequences -= 1
            else:
                orig_files[name] = name_fp

    if num_avail_sequences >= 2:

        # See what already exists
        existing_files = os.listdir(outpath)

        if isinstance(row[1]['RTSTRUCT_FileLocation'], str):
            orig_rtstruct_file = row[1]['RTSTRUCT_FileLocation']
            orig_rtstruct_file = orig_rtstruct_file.replace('  ', ' ')
            try:
                temp_rtstruct_file = shutil.copy2(orig_rtstruct_file, scratch_temppath)
                logger.info(f'Copied RTSTRUCT from {orig_rtstruct_file} to {temp_rtstruct_file}')
                segmentation_rt['orig'] = temp_rtstruct_file
            except FileNotFoundError:
                logger.error(f'No file found: {orig_rtstruct_file}')

        if keep_old:
            keep_files = []
            for old_file in existing_files:
                if 'MRI' in old_file:
                    name = 'MRI'
                else:
                    name = old_file.split('_')[0]
                old_filepath = os.path.join(outpath, old_file)
                if 'segmentation_rt' in old_file:
                    if '_reg' in old_file:
                        segmentation_rt['reg'] = old_filepath
                        keep_files.append(old_filepath)
                    elif '_r2s' in old_file:
                        segmentation_rt['r2s'] = old_filepath
                        keep_files.append(old_filepath)
                    elif '_noreg' in old_file:
                        segmentation_rt['orig'] = old_filepath
                        keep_files.append(old_filepath)
                elif '_ref' in old_file:
                    reference_file = old_filepath
                    reference_sequence = name
                    reg_files[name] = old_filepath
                    keep_files.append(old_filepath)
                elif '_reg' in old_file:
                    reg_files[name] = old_filepath
                    keep_files.append(old_filepath)
                elif '_r2s' in old_file:
                    oriented_files[name] = old_filepath
                    keep_files.append(old_filepath)
                elif '_noreg' in old_file:
                    orig_files[name] = old_filepath
                    keep_files.append(old_filepath)
            logger.info(f'Keeping existing files: {keep_files}')

        for name, fp in orig_files.items():
            if (name not in reg_files.keys()) and (name not in oriented_files.keys()) and name != reference_sequence:
                logger.info(f'Reorienting {name}')
                oriented_outfile = os.path.join(scratch_temppath, f'{name}_r2s.nii.gz')
                try:
                    reorient = fsl.Reorient2Std()
                    reorient.inputs.in_file = fp
                    reorient.inputs.out_file = oriented_outfile
                    reorient.inputs.output_type = 'NIFTI_GZ'
                    res = reorient.run()
                    oriented_files[name] = oriented_outfile
                except:
                    logger.error(f'Error while orienting {name}')

        if segmentation_rt:
            if ('reg' not in segmentation_rt.keys()) and ('r2s' not in segmentation_rt.keys()):
                logger.info(f'Reorienting segmentation_rt')
                oriented_outfile = os.path.join(scratch_temppath, 'segmentation_rt_r2s.nii.gz')
                try:
                    reorient = fsl.Reorient2Std()
                    reorient.inputs.in_file = segmentation_rt['orig']
                    reorient.inputs.out_file = oriented_outfile
                    reorient.inputs.output_type = 'NIFTI_GZ'
                    res = reorient.run()
                    segmentation_rt['r2s'] = oriented_outfile
                except:
                    logger.error(f'Error while orienting segmentation_rt')

        if reference_sequence == None:
            for ref in list(reversed(config.REFERENCE_PREFERRED_ORDER)):
                if ref in avail_sequences:
                    reference_sequence = ref
            
        if reference_sequence != None:
            logger.info(f'Reference file available: {reference_sequence}')
            if reference_file == None:
                reference_file = oriented_files[reference_sequence]
            ref_spacing = config.ISOTROPIC_SPACING

            try:
                ref_img = ants.image_read(reference_file)
                ref_img.set_spacing(ref_spacing)
                ref_outfile = os.path.join(scratch_outpath, f'{reference_sequence}_ref.nii.gz')
                ants.image_write(ref_img, ref_outfile)
                reg_files[reference_sequence] = ref_outfile
            
            except:
                logger.error(f'Error while loading reference {reference_sequence}')
                reference_sequence = None

        else:
            available_names = list(oriented_files.keys())
            logger.error(f'No reference file available: {available_names}')

        metadata_row = [accession_number, mrn, superdir,
                        list(orig_files.keys()), reference_sequence]

        if reference_sequence != None:
            for name, fp in oriented_files.items():
                if name not in reg_files.keys() and name != 'DIFFUSION' and name != reference_sequence:
                    reg_temppath = os.path.join(scratch_temppath, f'{name}_reg_files/')
                    if not os.path.exists(reg_temppath):
                        os.mkdir(reg_temppath)
                    new_spacing = config.ISOTROPIC_SPACING
                    # new_spacing = tuple(np.fromstring(row[1][f'{name}_NewSpacing'][1:-1], sep=' '))

                    # if the file is the rtMRI, register both the MRI and the
                    # segmentation
                    if 'MRI' in name:
                        logger.info(f'Registering {name} to {reference_sequence} as well as segmentation')
                        outfile = os.path.join(scratch_outpath, f'{name}_reg.nii.gz')
                        seg_outfile = os.path.join(scratch_outpath, 'segmentation_rt_reg.nii.gz')
                        logger.info(fp)
                        logger.info(segmentation_rt['r2s'])
                        
                        try:
                            moving_img = ants.image_read(fp)
                            moving_img.set_spacing(new_spacing)

                            seg_img = ants.image_read(segmentation_rt['r2s'])
                            seg_img.set_spacing(new_spacing)
                            mytx = ants.registration(fixed=ref_img,
                                                    moving=moving_img,
                                                    type_of_transform='SyN',
                                                    outprefix=reg_temppath)

                            warped_seg = ants.apply_transforms(fixed=ref_img, moving=seg_img, 
                                                                transformlist=mytx['fwdtransforms'])

                            warped_moving = mytx['warpedmovout']

                            ants.image_write(warped_moving, outfile)
                            ants.image_write(warped_seg, seg_outfile)

                            reg_files[name] = outfile
                            segmentation_rt['reg'] = seg_outfile
                            
                        except:
                            logger.error(f'Error while registering {name} and segmentation')

                    else:
                        logger.info(f'Registering {name} to {reference_sequence}')
                        outfile = os.path.join(scratch_outpath, f'{name}_reg.nii.gz')

                        try:
                            moving_img = ants.image_read(fp)
                            moving_img.set_spacing(new_spacing)

                            mytx = ants.registration(fixed=ref_img,
                                                     moving=moving_img,
                                                     type_of_transform='SyN',
                                                     outprefix=reg_temppath)

                            warped_moving = mytx['warpedmovout']

                            ants.image_write(warped_moving, outfile)

                            reg_files[name] = outfile
                            
                        except:
                            logger.error(f'Error while registering {name}')

        # if reg_files.keys() != oriented_files.keys():
        #     noreg_names = oriented_files.keys() - reg_files.keys()
        #     for name in noreg_names:
        #         reg_files[name] = oriented_files[name]
        #         outfile = os.path.join(scratch_outpath, f'{name}_noreg.nii.gz')
        #         new_spacing = config.ISOTROPIC_SPACING

        #         try:
        #             img = ants.image_read(oriented_files[name])
        #             img.set_spacing(new_spacing)

        #             ants.image_write(img, outfile)
        #             reg_files[name] = outfile
                
        #         except:
        #             logger.error(f'Error while loading {name}')


    else:
        metadata_row = [accession_number, mrn, superdir, avail_sequences, np.nan]
    
    # Append filepath to metadata
    if num_avail_sequences < 2:
        for name in config.TARGET_FILES:
            if name in avail_sequences:
                metadata_row.append(row[1][f'{avail_sequences[0]}_{config.FILELOCATION_COL}'])
            else:
                metadata_row.append(np.nan)
    else:
        for name in config.TARGET_FILES:
            if name in avail_sequences:
                try:
                    scratch_file = reg_files[name]
                    if not os.path.exists(scratch_file):
                        logger.error(f'File does not exist: {scratch_file}')
                        metadata_row.append(np.nan)
                    elif f'{name}_reg.nii.gz' in existing_files:
                        logger.info(f'File already exists: {scratch_file}')
                        metadata_row.append(scratch_file)
                    else:
                        logger.info(f'Copying file to outpath: {scratch_file}')
                        data_file = shutil.copy2(scratch_file, outpath)
                        metadata_row.append(data_file)
                        os.remove(scratch_file)

                except KeyError:
                    try:
                        scratch_file = oriented_files[name]
                        if not os.path.exists(scratch_file):
                            logger.error(f'File does not exist: {scratch_file}')
                            metadata_row.append(np.nan)
                        elif f'{name}_r2s.nii.gz' in existing_files:
                            logger.info(f'File already exists: {scratch_file}')
                            metadata_row.append(scratch_file)
                        else:
                            logger.info(f'Copying file to outpath: {scratch_file}')
                            data_file = shutil.copy2(scratch_file, outpath)
                            metadata_row.append(data_file)
                            os.remove(scratch_file)

                    except KeyError:
                        logger.error(f'No registered or oriented file found: {name}')
                        data_file = shutil.copy2(orig_files[name], outpath)
                        os.rename(data_file, data_file[:-7] + '_orig.nii.gz')
                        metadata_row.append(orig_files[name])

            else:
                metadata_row.append(np.nan)

    if segmentation_rt:
        try:
            scratch_file = segmentation_rt['reg']
            if not os.path.exists(scratch_file):
                logger.error(f'File does not exist: {scratch_file}')
                metadata_row.append(np.nan)
            elif f'segmentation_rt_reg.nii.gz' in existing_files:
                logger.info(f'File already exists: {scratch_file}')
                metadata_row.append(scratch_file)
            else:
                logger.info(f'Copying file to outpath: {scratch_file}')
                data_file = shutil.copy2(scratch_file, outpath)
                metadata_row.append(data_file)
                os.remove(scratch_file)
        except KeyError:
            try:
                scratch_file = segmentation_rt['r2s']
                if not os.path.exists(scratch_file):
                    logger.error(f'File does not exist: {scratch_file}')
                    metadata_row.append(np.nan)
                elif f'segmentation_rt_r2s.nii.gz' in existing_files:
                    logger.error(f'File already exists: {scratch_file}')
                    metadata_row.append(scratch_file)
                else:
                    logger.info(f'Copying file to outpath: {scratch_file}')
                    data_file = shutil.copy2(scratch_file, outpath)
                    metadata_row.append(data_file)
                    os.remove(scratch_file)
            except KeyError:
                logger.error(f'No registered or oriented file found: segmentation_rt')
                data_file = shutil.copy2(segmentation_rt['orig'], outpath)
                os.rename(data_file, data_file[:-7] + '_orig.nii.gz')
                metadata_row.append(data_file)
    else:
        logger.error(f'No segmentation_rt file')
        metadata_row.append(np.nan)
    

    # Copy temp directory
    if scratch_temppath is not None:
        shutil.copytree(scratch_temppath, temppath, dirs_exist_ok=True)
        shutil.rmtree(scratch_temppath)
    
    # Copy output directory
    if scratch_outpath is not None:
        logger.error(f'Some files moved to {temppath}')
        shutil.copytree(scratch_outpath, temppath, dirs_exist_ok=True)
        shutil.rmtree(scratch_outpath)

    return metadata_row
