import logging
import os
import warnings

LOG_FORMATTER = logging.Formatter(
        datefmt='%H:%M:%S %d.%m.%Y',
        fmt="%(asctime)s : %(levelname)s - %(message)s")

def setup_filehandler(filename):
    l_handler = logging.FileHandler(filename, mode="w")
    l_handler.setLevel(logging.WARNING)
    l_handler.setFormatter(LOG_FORMATTER)
    return l_handler

def setup_streamhandler():
    s_handler = logging.StreamHandler()
    s_handler.setLevel(logging.INFO)
    s_handler.setFormatter(LOG_FORMATTER)
    return s_handler


def configure_logger(out_dir=None):
    # ignore overflow errors, assume these to be mostly flagged data
    warnings.simplefilter("ignore")

    if out_dir is None:
        out_dir = "."
    else:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
    
    l_handler = setup_filehandler(
        os.path.join(out_dir, "xcrapping.log"))

    s_handler = setup_streamhandler()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.addHandler(l_handler)
    logger.addHandler(s_handler)
    return logger