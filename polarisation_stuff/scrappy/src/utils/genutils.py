import os
import sys
import shutil
from time import perf_counter

import numpy as np

from utils.logger import LOG_FORMATTER, setup_streamhandler, logging

snitch = logging.getLogger(__name__)
snitch.setLevel("INFO")
snitch.addHandler(setup_streamhandler())

def make_out_dir(dir_name, delete_if_exists=False):
    if os.path.isdir(dir_name) and delete_if_exists:
        snitch.info(f"Deleting '{dir_name}' because it already exists")
        shutil.rmtree(dir_name)
        
    if not os.path.isdir(dir_name):
        snitch.info(f"Creating directory: {dir_name}")
        os.makedirs(dir_name)
    return os.path.abspath(dir_name)

def fullpath(*args):
    return os.path.join(*args)


def read_sorted_filnames(fname):
    with open(fname, "r") as post:
        items = post.readlines()
        items = [_.replace("\n", "") for _ in items]
    return items


def timer(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        snitch.info(f"'{func.__name__}' run in: {perf_counter()-start:.2f} sec")
        return result
    return wrapper


def read_npz(filename):
    with np.load(filename, allow_pickle=True) as dat:
        datas = dict(dat)
    return datas


def make_importable():
    import os
    import sys
    PATH = set(sys.path)
    PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir))
    if not PATH.issuperset(PROJECT_ROOT):
        sys.path.append(PROJECT_ROOT)


def does_specified_file_exist(*args):
    not_found = []
    for fname in args:
        if fname is None:
            continue
        if not os.path.isfile(fname):
            not_found.append(fname)

    if len(not_found)>0:
        snitch.error("The following files/dirs are required but were not found:")
        for fname in not_found:
            snitch.error(f"-> {fname}")
        snitch.error("Please ensure that they exist if intended for use")
        sys.exit()
    else:
        return True
    


# from https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
class dicto(dict):
    """Use dot operator to access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
