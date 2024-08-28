import glob
import argparse
import concurrent.futures
import numpy as np

from utils import mri_utils, metadata_utils
from config import config


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str,
                    default="/gpfs/data/oermannlab/ObermannNeuro3/dicom/",
                    help='directory of accession directories with DICOM files')
parser.add_argument('--spacing', type=list, default=config.ISOTROPIC_SPACING,
                    help='new spacing for resampling')
parser.add_argument('--n_splits', type=int, default=1)
parser.add_argument('--split_num', type=int, default=1)
parser.add_argument('--start_idx', type=int, default=0)



def main(args):    
    accension_dirs = glob.glob(args.dir + '*/')

    split_arr = np.array_split(accension_dirs, args.n_splits)
    acc_dirs_split = split_arr[args.split_num - 1]
    num_dirs_in_split = len(acc_dirs_split)

    preprocess_mri_split_fn = config.PREPROCESS_MRI_FN + f'_{args.split_num}'

    logger = metadata_utils.create_logger_handler(
        args.dir, preprocess_mri_split_fn)

    metadata_fp = args.dir + preprocess_mri_split_fn + '.csv'
    metadata_utils.write_row_to_metadata(config.PREPROCESS_MRI_HEADER, 
        metadata_fp)

    for i, acc_dir in enumerate(acc_dirs_split[args.start_idx:]):
        logger.info(f"Processing {i} out of {num_dirs_in_split}")

        # Run preprocessing function
        row = mri_utils.preprocess_mri(acc_dir, args.spacing, logger)
        
        logger.info(f"Writing row to metadata: {row}")
        metadata_utils.write_row_to_metadata(row, metadata_fp)
     
    
if __name__ in "__main__":
    args = parser.parse_args()
    main(args)
    
