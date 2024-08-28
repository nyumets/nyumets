import os
import argparse
import pandas as pd
import numpy as np

from config import config
from utils import metadata_utils, coregister_utils


parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int, default=1)
parser.add_argument('--split_num', type=int, default=1)
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--keep_old', type=bool, default=True)


def main(args):

    df_all = pd.read_csv(config.MAPPED_CSV)

    split_arr = np.array_split(df_all, args.split)
    df_split = split_arr[args.split_num - 1]

    coreg_split_fn = config.COREG_FN + f'_{args.split_num}'

    if not os.path.exists(config.FINAL_DATASET_DIR):
        os.mkdir(config.FINAL_DATASET_DIR)
    
    logger = metadata_utils.create_logger_handler(config.FINAL_DATASET_DIR,
        coreg_split_fn)

    metadata_fp = config.FINAL_DATASET_DIR + coreg_split_fn + '.csv'
    metadata_utils.write_row_to_metadata(config.RUN_COREGISTRATION_HEADER, 
        metadata_fp)

    num_acc_in_split = df_split.shape[0]

    for i, acc_row in enumerate(df_split.iloc[args.start_idx:].iterrows()):
        
        logger.info(f"Processing {i+args.start_idx} out of {num_acc_in_split} accessions")
        
        # Run coregistration on all scans in timepoint for one patient
        metadata_row = coregister_utils.coregister_timepoint(acc_row, logger,
                                                             args.keep_old)

        logger.info(f"Writing row to metadata: {metadata_row}")
        metadata_utils.write_row_to_metadata(metadata_row, metadata_fp)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
            
 