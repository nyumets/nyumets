import os
import glob
import argparse
import numpy as np

from utils import rtstruct_utils, metadata_utils
from config import config


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str,
                    default="/gpfs/data/oermannlab/private_data/temporalMRI/gammaplan/",
                    help='directory of accession directories with RTSTRUCT directories')
parser.add_argument('--n_splits', type=int, default=1)
parser.add_argument('--split_num', type=int, default=1)
parser.add_argument('--start_idx', type=int, default=0)


def main(args):
    """ Extracts RTSTRUCT segmentations and associated MRI scans and
    resamples/saves to new directory with other structural MRI scans as .nii.gz
    for consistency.
    
    TODO: extract RTDOSE segmentations and metadata as well.
    """
    patient_dirs = glob.glob(args.dir + '*/')

    split_arr = np.array_split(patient_dirs, args.n_splits)
    pt_dirs_split = split_arr[args.split_num - 1]
    num_dirs_in_split = len(pt_dirs_split)

    preprocess_rt_split_fn = config.PREPROCESS_RT_FN + f'_{args.split_num}'

    logger = metadata_utils.create_logger_handler(
        args.dir, preprocess_rt_split_fn)

    metadata_fp = args.dir + preprocess_rt_split_fn + '.csv'
    metadata_utils.write_row_to_metadata(config.PREPROCESS_RT_HEADER, 
        metadata_fp)

    for i, pt_dir in enumerate(pt_dirs_split[args.start_idx:]):
        
        logger.info(f"Processing {i} out of {num_dirs_in_split} pts")
        logger.info(f"Patient dir: {pt_dir}")
        
        accension_dirs = glob.glob(pt_dir + '*/')
        
        for j, acc_dir in enumerate(accension_dirs):
            logger.info(f"Processing {i} out of {len(accension_dirs)} acc")

            # Run preprocessing function
            rows = rtstruct_utils.preprocess_rt(acc_dir, logger)
            
            for row in rows:
                logger.info(f"Writing row to metadata: {row}")
                metadata_utils.write_row_to_metadata(row, metadata_fp)
        

for __name__ in "__main__":
    args = parser.parse_args()
    main(args)
    