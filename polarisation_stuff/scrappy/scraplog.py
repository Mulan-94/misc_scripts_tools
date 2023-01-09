import logging
import os
import warnings

# from utils import make_out_dir
from arguments import parser 

def configure_logger(out_dir):
    # ignore overflow errors, assume these to be mostly flagged data
    warnings.simplefilter("ignore")

    formatter = logging.Formatter(
        datefmt='%H:%M:%S %d.%m.%Y',
        fmt="%(asctime)s : %(levelname)s - %(message)s")


    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    l_handler = logging.FileHandler(
        os.path.join(out_dir, "xcrapping.log"), mode="w")
    l_handler.setLevel(logging.WARNING)
    l_handler.setFormatter(formatter)

    s_handler = logging.StreamHandler()
    s_handler.setLevel(logging.INFO)
    s_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.addHandler(l_handler)
    logger.addHandler(s_handler)
    return logger

# appedning the region size to the directory
# this is the default REGSIZE from scrappy
opts = parser().parse_args()
odir = opts.odir  
odir += "-s30" if opts.reg_size is None else f"-s{opts.reg_size}"
snitch = configure_logger(odir)