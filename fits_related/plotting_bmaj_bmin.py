#############
## Lexy plotting BMAJ bmin BLABAL
#######################

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from astropy.io import fits
from ipdb import set_trace

plt.style.use("seaborn")

def read_fits(imname):
    outs = {}
    with fits.open(imname) as hdul:
        outs[imname] = hdul[0].header["BMAJ"], hdul[0].header["BMIN"], hdul[0].header["CRVAL3"]/1e9
    return outs


def read_cube_fits(imname):
    outs = []
    with fits.open(imname) as hdul:
        #set_trace()
        nfreqs = range(1, hdul[0].header["NAXIS3"]+1)
        for freq_id in nfreqs:
            outs.append(
                (hdul[0].header[f"BMAJ{freq_id}"], hdul[0].header[f"BMIN{freq_id}"],
                                freq_id))
    return outs


def plotme(freqs, bmajs, bmins, outname, title="All freqs"):

    fig, ax = plt.subplots(figsize=(16,9), ncols=2, sharex=True, sharey=True)
    ax[0].plot(freqs, bmajs, "bo", markersize=4)
    ax[0].set_xlabel("Freq GHz")
    ax[0].set_ylabel("BMAJ")
    ax[0].set_title(title)



    ax[1].plot(freqs, bmins, "bo", markersize=4)
    ax[1].set_xlabel("Freq GHz")
    ax[1].set_ylabel("BMIN")
    ax[1].set_title(title)


    fig.tight_layout()
    print(f"Saving file at: {outname}")
    fig.savefig(outname)


def get_params(pairs):
    bmajs = [_[0] for _ in pairs]
    bmins = [_[1] for _ in pairs]
    freqs = [_[2] for _ in pairs]
    return bmajs, bmins, freqs



def multiple_single_files(inf_name=None, files=None, oname=None):
    """
    read file containing filenames for the images to be processed
    these are stored in e.g eds.txt
    if a file containing these file names is not avaliable, 
    just use the files as they are
    """

    if inf_name:
        with open(inf_name, "r") as fil:
            data = fil.readlines()

        data = [_.strip("\n") for _ in data]
    else:
        data = files
    pairs = [read_fits(dat)[dat] for dat in data]
    bmajs, bmins, freqs = get_params(pairs)

    if oname is None:
        if files is not None:
            fbase = os.path.basename(os.path.commonprefix(files))
        else:
            fbase = os.path.basename(os.path.spiltext(inf_name)[0])

        stokes = os.path.splitext(files[0])[0].replace(fbase, "").split("-")[1]
        stokes = "I" if stokes=="image" else stokes
        
        oname = f"{fbase}-{stokes}-bmaj-bmin-vs-freq.png"
    plotme(freqs, bmajs, bmins, oname, title="All freqs")


def single_cube_file(cube_name, oname=None):
    ## in the case of multiple data cubes
    pairs = read_cube_fits(cube_name)
    bmajs, bmins, freqs = get_params(pairs)
    if oname is None:
        oname = f"bmaj-bmin-vs-freq-{cube_name}.png"
    plotme(freqs, bmajs, bmins, oname, title="All freqs")



def parser():
    ps = argparse.ArgumentParser()
    ps.add_argument("-c", "-cube", dest="cubes" , metavar="",
        nargs="*", type=str,  default=None,
        help="Input cubes"
    )

    ps.add_argument("-f", "-file", dest="files", metavar="",
        nargs="*", type=str,  default=None, action="append",
        help="""Input multiple files. If you have different groups of 
        images you want to workon, specify this argument mutliptle times. 
        e.g -f ../*[0-9][0-9][0-9][0-9]-I-*image* -f ../*[0-9][0-9][0-9][0-9]-Q-*image*
        """
    )

    ps.add_argument("-o", "--output", dest="output", metavar="",
        type=str, default=None, help="Name of output file"
    )

    return ps



if __name__ == "__main__":
    ps = parser().parse_args()

    if ps.cubes is not None:
        for cube in ps.cubes:
            single_cube_file(cube_name=cube, oname=ps.output)

    if ps.files is not None:
        for file_grp in ps.files:
            multiple_single_files(files=file_grp, oname=ps.output)


# How to run the script. Can be imported into ipython
#inf_name = "eds.txt"
#multiple_single_files(inf_name)
#
