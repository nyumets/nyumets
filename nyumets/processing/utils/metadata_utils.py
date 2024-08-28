import logging
import csv


def create_logger_handler(logging_dir, logging_fn):
    logger = logging.getLogger(name=logging_fn)
    logger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler(logging_dir + logging_fn + ".log")
    fileHandler.setLevel(logging.DEBUG)

    logger.addHandler(fileHandler)

    formatter = logging.Formatter('%(levelname)s:%(message)s')

    fileHandler.setFormatter(formatter)

    return logger


def write_row_to_metadata(row, metadata_fp):
    with open(metadata_fp, "a", newline="") as f:
        wr = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar=' ', quotechar='')
        wr.writerow(row)


def update_data_path(old_path):
    old_path_list = old_path.split('/')
    old_path_list.insert(4, 'private_data')
    new_path = '/'.join(old_path_list)
    return new_path
