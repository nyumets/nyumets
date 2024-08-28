import os
import yaml
import glob
import pandas as pd
from typing import List
from pathlib import Path


with open(Path(__file__).parents[2] / "config.yaml") as f:
    config = yaml.safe_load(f)


# TODO: represent as enum class
nyumets_modalities = {
    'T1_pre': 'T1',
    'T1_post': 'CT1', # spin-echo
    'T2': 'T2',
    'FLAIR': 'FLAIR',
    'T1_post_GK': 'RTSTRUCT_MRI',  # MPRAGE
    'segmentation': 'RTSTRUCT_segmentation',
    'T1_post_all': ['CT1', 'RTSTRUCT_MRI']
}

brats_modalities = {
    'T1_pre': 't1',
    'T1_post': 't1ce',
    'T2': 't2',
    'FLAIR': 'flair',
    'segmentation': 'seg'
}

stanford_modalities = {
    'T1_pre': '1',
    'T1_post': '2',  # spin-echo
    'FLAIR': '3',
    'T1_post_gradientecho': '0',  # MPRAGE
    'segmentation': 'seg',
    'T1_post_all': ['2', '0']
}


def get_nyumets_data(
    split: str = 'train',
    image_modalities: list = config['image_modalities'], # options: 'T1', 'T1_post', 'T1_post_GK', 'T2', 'FLAIR'
    keep_temporal: bool = True,
    expert_adjusted_only: bool = False,
    debug_subset: int = None,
    img_csv_path: str = config['nyumets']['img_metadata_csv'],
    split_csv_path: str = config['nyumets']['split_csv'],
    nyumets_dir_path: str = config['nyumets']['dir'],
):
    """
    Args:
        split: split of the data to return. Default is `train`.
        image_modalities: list of image modalities to load. Default is `['T1_pre', 'T1_post', 'T2', 'FLAIR']`.
        keep_temporal: bool whether to keep all images from all patients or whether to only keep one timepoint per patient. Default is True.
        expert_adjusted_only: bool whether to only return studies which have expert adjusted labels. Default is False.
        debug_subset: number of samples to use for debugging purposes. Default is `None` i.e. use full dataset.
        img_csv_path: path to nyumets image metadata csv.
        split_csv_path: path to nyumets split csv.
        nyumets_dir_path: path to nyumets data.
        
    Returns:
        data: list of dictionaries of data to be loaded into dataset.
    """
    
    img_df = pd.read_csv(img_csv_path)

    equivalent_cols = []
    img_cols = []
    for img_mod in image_modalities:
        col_name = nyumets_modalities[img_mod]
        if type(col_name) == list:
            equivalent_cols.append(col_name)
        elif type(col_name) == str:
            img_cols.append(col_name)

    label_column_name = nyumets_modalities['segmentation']
    all_cols = img_cols + [label_column_name]

    img_df = img_df.loc[img_df[all_cols].all(axis=1)]

    for cols in equivalent_cols:
        img_df = img_df.loc[img_df[cols].any(axis=1)]
    
    if not keep_temporal:
        img_df = img_df[img_df.duplicated(subset='PatientID', keep=False)]
    
    all_splits_df = pd.read_csv(split_csv_path)
    split_df = img_df[img_df['StudyID'].isin(all_splits_df[all_splits_df[split] == 1]['StudyID'])]
    
    if debug_subset is not None:
        split_df = split_df.head(debug_subset)
    
    data = []
    
    for row in split_df.iterrows():
        
        if expert_adjusted_only and not int(row[1]['ExpertAdjusted']):
            continue

        row_dict = {}
        row_dict['patient_id'] = str(int(row[1]['PatientID']))
        row_dict['study_id'] = str(int(row[1]['StudyID']))
        row_dict['expert_adjusted'] = int(row[1]['ExpertAdjusted'])

        split_dir = os.path.join(nyumets_dir_path, 'imaging/patientId', row_dict['patient_id'], 'studyId', row_dict['study_id'])

        for modality in image_modalities:
            mod_name = nyumets_modalities[modality]
            if type(mod_name) == list:
                for equivalent_seq in mod_name:
                    img_path = os.path.join(split_dir, f'{equivalent_seq}.nii.gz')
                    if os.path.exists(img_path):
                        row_dict[modality] = img_path
                    else:
                        continue
            elif type(mod_name) == str:
                img_path = os.path.join(split_dir, f'{mod_name}.nii.gz')
                if os.path.exists(img_path):
                    row_dict[modality] = img_path
                else:
                    continue

        label_path = os.path.join(split_dir, f'{label_column_name}.nii.gz')
        if os.path.exists(label_path):
            row_dict['label'] = label_path
        else:
            continue

        row_dict['timepoint'] = int(row[1]['RelativeTimepoint'])

        data.append(row_dict)
    
    return data


def get_brats21_data(
    split: str = 'train',
    image_modalities: list = config['image_modalities'],
    split_csv_path: str = config['brats21']['split_csv'],
    brats_dir_path: str = config['brats21']['dir'],
):
    """
    Retrieve BraTS 21 data in dictionary format for easy conversion to a Dataset.
    """
    split_df = pd.read_csv(split_csv_path)
    split_df = split_df[split_df['StudyID'].isin(split_df[split_df[split] == 1]['StudyID'])]
    
    label_name = brats_modalities['segmentation']

    data = []
    
    for row in split_df.iterrows():
        row_dict = {}
        row_dict['study_id'] = str(row[1]['StudyID'])
        
        split_dir = os.path.join(brats_dir_path, row_dict['study_id'])
        
        img_missing = False
        for modality in image_modalities:
            mod_name = brats_modalities[modality]
            img_path = os.path.join(split_dir, f"{os.path.basename(split_dir)}_{mod_name}.nii.gz")
            
            if os.path.exists(img_path):
                row_dict[modality] = img_path
            else:
                img_missing = True
                continue
        
        label_path = os.path.join(split_dir, f"{os.path.basename(split_dir)}_{label_name}.nii.gz")
        if os.path.exists(label_path):
            row_dict['label'] = label_path
        else:
            continue
        
        if not img_missing:
            data.append(row_dict)
    
    return data

def get_stanfordbrainmetsshare_data(
    split: str = 'train',  # options: 'all', 'train', 'val', 'test'
    image_modalities: list = config['image_modalities'],
    stanford_dir_path: str = config['stanford']['dir'],
    image_modality: str = 't1_post'
):  
    base_split_dir = os.path.join(stanford_dir_path, 'stanford_release_brainmask', 'mets_stanford_releaseMask_train')
    split_dirs = glob.glob(base_split_dir + '/*')

    if split == 'train':
        split_dirs = split_dirs[:73]
    elif split == 'val':
        split_dirs = split_dirs[74:93]
    elif split == 'test':
        split_dirs = split_dirs[94:]
    elif split != 'all':
        raise NotImplementedError(f'Split `{split}` not recognized.')
    
    data = []

    for met_dir in split_dirs:
        row_dict = {}
        row_dict['study_id'] = os.path.basename(met_dir)

        for img_mod in image_modalities:
            mod_name = stanford_modalities[img_mod]
            img_path = os.path.join(met_dir, f'{mod_name}', 'volume.nii.gz')

            if os.path.exists(img_path):
                row_dict[img_mod] = img_path

        seg_name = stanford_modalities['segmentation']
        seg_nifti = os.path.join(met_dir, f'{seg_name}', 'volume.nii.gz')
        if os.path.exists(seg_nifti):
            row_dict['label'] = seg_nifti
    
        data.append(row_dict)
    
    return data
