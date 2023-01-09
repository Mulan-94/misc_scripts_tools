import os
from time import perf_counter

from scraplog import snitch

def make_out_dir(dir_name):
    if not os.path.isdir(dir_name):
        snitch.info(f"Creating directory: {dir_name}")
        os.makedirs(dir_name)
    return os.path.relpath(dir_name)

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