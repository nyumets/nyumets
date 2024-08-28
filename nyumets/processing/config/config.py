
MRI_DIRS = [
       "/gpfs/data/oermannlab/private_data/ObermannNeuro/dicom/",
       "/gpfs/data/oermannlab/private_data/ObermannNeuro2/dicom/",
       "/gpfs/data/oermannlab/ObermannNeuro3/dicom/"]

RT_DIRS = [
       "/gpfs/data/oermannlab/private_data/temporalMRI/gammaplan/",
       "/gpfs/data/oermannlab/private_data/temporalMRI/metastases_segmentations/",
       "/gpfs/data/oermannlab/private_data/Brainlab_exports/"
]

FINAL_DATASET_DIR = '/gpfs/data/oermannlab/private_data/NYUMets/dataset_v2/'

FINAL_METADATA_CSV = FINAL_DATASET_DIR + 'preprocess_metadata.csv'

ISOTROPIC_SPACING = [1., 1., 1.]

STANFORD_DATA_DIR = "/gpfs/data/oermannlab/public_data/brainmetshare-2/"

# CONFIG: preprocess_mri.py
PREPROCESS_MRI_HEADER = ["AccessionNumber",
                         "Scanner",
                         "ScanDate",
                         "StudyDescription",
                         "PatientID",
                         "PatientName",
                         "PatientBirthDate",
                         "T1",
                         "T1_OldSpacing",
                         "T1_NewSpacing",
                         "T1_FileLocation",
                         "CT1",
                         "CT1_OldSpacing",
                         "CT1_NewSpacing",
                         "CT1_FileLocation",
                         "T2",
                         "T2_OldSpacing",
                         "T2_NewSpacing",
                         "T2_FileLocation",
                         "FLAIR",
                         "FLAIR_OldSpacing",
                         "FLAIR_NewSpacing",
                         "FLAIR_FileLocation",
                         "DIFFUSION",
                         "DIFFUSION_OldSpacing",
                         "DIFFUSION_NewSpacing",
                         "DIFFUSION_FileLocation",
                        ]

PREPROCESS_MRI_FN = "preprocess_mri"

BEST_MATCHES = {
    'T1': ['ax t1', 'ax t1 pre', 'ax t1 pre_fbb', 't1 se ax brain_pre', 't1ax',
           'ax t1 fse', 't1 3d axial', 'ax t1 acr', 'ax t1 pre brain',
           'axial t1', 'ax t1 brain'],
    'T2': ['ax t2', 'ax t2_fbb', 't2 ax', 'ax t2 new', 't2 tse ax brain',
           'ax t2 fs', 't2 axial', 'axl t2 tse', 't2_3d axial', 'ax t2 acr',
           'axial t2', 'ax t2 brain post', 'ax t2 brain', 'ax t2_rr',
           'axi fse t2', 'post ax t2'],
    'CT1': ['ax t1 post_fbb', 't1 ax post', 'ax t1 fs post', 'ax t1 post',
            't1 se ax_post', 'post ax t1 fc', 'axl t1 post (5mm)',
            'ax t1 post acr', 't1 mprage ax (do not mpr sag!!!!!!!!!!!!!)',
            'axial t1 post', 'ax t1 brain post', 'ax t1 post', 'axi t1 se+c',
            'ax 3d mpr new-3t', 'ax mpr post', 'ax 3d mpr', 'ax mpr',
            'ax 3d mpr new'],
    'FLAIR': ['ax flair', 'ax flair_fbb', 'axial flair', '3d flair space',
              'ax t2 flair', 'flair ax', 'axi t2 flair', 'ax flair t2',
              't2 ax flair', 'ax t2 flair acr', 'ax flair no fs',
              'axial flair', 'ax flair brain post'],
    'DIFFUSION': ['ax diffusion mb2_adc', 'mb2_diffusion_adc',
                  'ax diffusion_adc', 'ax diff trace_adc', 
                  'ax diffusion sms_adc', 'ax diffusion_adc',
                  'mb2_diffusion_trace_adc', 'axl dwi_adc', 'ax dwi_adc', 
                  'ax diffusion brain_adc']
}


# CONFIG: preprocess_rt.py
PREPROCESS_RT_HEADER = ['SegPath',
                        'TumorName',
                        'RTSTRUCT_Path',
                        'MRI_Dir',
                        'RTSTRUCT_DateTime',
                        'RTSTRUCT_FileLocation',
                        'MRI_FileLocation',
                        'ErrorMessage',
                        'Scanner',
                        'PatientID',
                        'PatientName',
                        'PatientBirthDate',
                        'MRIScanDate',
                        'MRIStudyDescription',
                        'OldSpacing',
                        'NewSpacing'
                        ]

PREPROCESS_RT_FN = 'preprocess_rt'

MRI_PATH_KEYWORDS = ['MR', 'MPR', 'AX', '3D', 'T1']


# CONFIG: run_coregistration.py
MAPPED_CSV = '/gpfs/data/oermannlab/private_data/NYUMets/mri_rtstruct_metadata_mapped_20211115.csv'

COREG_FN = 'run_coregistration'

SCRATCH_DIR = '/gpfs/scratch/linkk01/'

TARGET_FILES = ['T1', 'CT1', 'T2', 'FLAIR', 'DIFFUSION', 'MRI']  # MRI files to coregister
SEGMENTATION_RT_FILE = 'segmentation_rt'
TARGET_EXTENSION = '.nii.gz'
REFERENCE_PREFERRED_ORDER = ['CT1', 'T1', 'T2', 'FLAIR']
FILELOCATION_COL = 'FileLocation'

INDIR = 'nifti'
OUTDIR = 'reg_out'
TEMPDIR = 'reg_temp'

RUN_COREGISTRATION_HEADER = ['AccessionNumber', 'MRN', 'Superdir',
                             'AvailableSequences', 'ReferenceSequence', 'T1_reg',
                             'CT1_reg', 'T2_reg', 'FLAIR_reg', 'DIFFUSION',
                             'rtMRI_reg', 'RTSTRUCT_reg']

