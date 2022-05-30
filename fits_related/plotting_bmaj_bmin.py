#############
## Lexy plotting BMAJ bmin BLABAL
#######################

import matplotlib.pyplot as plt
import numpy as np
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
    fig.savefig(outname)


def get_params(pairs):
    bmajs = [_[0] for _ in pairs]
    bmins = [_[1] for _ in pairs]
    freqs = [_[2] for _ in pairs]
    return bmajs, bmins, freqs



def multiple_single_files(inf_name):
    # read file containing filenames for the images to be processed
    # these are stored in e.g eds.txt

    with open(inf_name, "r") as fil:
        data = fil.readlines()

    data = [_.strip("\n") for _ in data]
    pairs = [read_fits(dat)[dat] for dat in data]
    bmajs, bmins, freqs = get_params(pairs)
    plotme(freqs, bmajs, bmins, oname, title="All freqs")


def single_cube_file(cube_name):
    ## in the case of multiple data cubes
    pairs = read_cube_fits(cube_name)
    bmajs, bmins, freqs = get_params(pairs)
    oname = f"bmaj-bmin-vs-freq-{cube_name}.png"
    plotme(freqs, bmajs, bmins, oname, title="All freqs")


# How to run the script. Can be imported into ipython
#inf_name = "eds.txt"
#multiple_single_files(inf_name)
#

#cubes = [
#    #"Q-image-cubes.fits",
#    "testingeq_Q.fits",
#    "testingeq_Q2.fits"
#]
#
#for cube in cubes:
#    single_cube_file(cube)
