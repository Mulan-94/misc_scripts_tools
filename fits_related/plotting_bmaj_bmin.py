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


def read_cube_fits(imname, channels=None):
    outs = []

    with fits.open(imname) as hdul:
        if channels is not None:
            print(f"Selecting {len(ps.channels)} channels out of {hdul[0].header['NAXIS3']}")
            print(f"Channels: {channels}")
            channels = np.array(channels)
            channels += 1
        else:
            channels = range(1, hdul[0].header["NAXIS3"]+1)
        for freq_id in channels:
            outs.append(
                (hdul[0].header[f"BMAJ{freq_id}"], hdul[0].header[f"BMIN{freq_id}"],
                                freq_id))
    return outs


def plotme(freqs, bmajs, bmins, outname, title="All freqs"):

    fig, ax = plt.subplots(figsize=(16,9), ncols=2, sharex=True, sharey=True)
    ax[0].plot(freqs, bmajs, "bo", markersize=4)
    ax[0].axhline(bmajs.max(), linestyle="--", linewidth=1)

    maxs = freqs[np.where(bmajs == bmajs.max())]
    if maxs.size > 1:
        maxs = maxs[0]

    ax[0].annotate(f"{bmajs.max():.3}", 
        xy=(freqs[maxs], bmajs.max()), color="red")


    # #linear fit
    # da, de, di = np.polyfit(freqs, np.ma.masked_where(bmajs==0, bmajs), 2)
    # fit = da * np.square(freqs) + np.multiply(de,freqs) + di
    # # fit = np.poly1d(np.polyfit(freqs, bmajs, 1))(np.unique(freqs))
    # ax[0].plot(freqs, fit, "--")
    
    ax[0].set_xlabel("Freq GHz")
    ax[0].set_ylabel("BMAJ")
    ax[0].set_title(title)


    ax[1].plot(freqs, bmins, "bo", markersize=4)
    ax[1].axhline(bmins.max(), linestyle="--", linewidth=1)
    
    maxs = freqs[np.where(bmins == bmins.max())]
    if maxs.size > 1:
        maxs = maxs[0]
    ax[1].annotate(f"{bmins.max():.3}", 
        xy=(freqs[maxs], bmins.max()), 
        color="red")

    #linear fit
    # fit = np.poly1d(np.polyfit(freqs, bmins, 1))(np.unique(freqs))
    # ax[1].plot(freqs, fit, "--")
    
    ax[1].set_xlabel("Freq GHz")
    ax[1].set_ylabel("BMIN")
    ax[1].set_title(title)

    fig.tight_layout()
    print(f"Saving file at: {outname}")
    fig.savefig(outname)


def get_params(pairs):
    bmajs = np.array([_[0] for _ in pairs])
    bmins = np.array([_[1] for _ in pairs])
    freqs = np.array([_[2] for _ in pairs])
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


def single_cube_file(cube_name, oname=None, channels=None):
    ## in the case of multiple data cubes
    pairs = read_cube_fits(cube_name, channels=channels)
    bmajs, bmins, freqs = get_params(pairs)
    if oname is None:
        oname = f"bmaj-bmin-vs-freq-{cube_name}.png"
    plotme(freqs, bmajs, bmins, oname, title="All freqs")



def parser():
    ps = argparse.ArgumentParser()
    ps.add_argument("-c", "--cube", dest="cubes" , metavar="",
        nargs="*", type=str,  default=None,
        help="Input cubes"
    )

    ps.add_argument("-f", "--file", dest="files", metavar="",
        nargs="*", type=str,  default=None, action="append",
        help="""Input multiple files. If you have different groups of 
        images you want to workon, specify this argument mutliptle times. 
        e.g -f ../*[0-9][0-9][0-9][0-9]-I-*image* -f ../*[0-9][0-9][0-9][0-9]-Q-*image*
        """
    )

    ps.add_argument("-o", "--output", dest="output", metavar="",
        type=str, default=None, help="Name of output file"
    )

    ps.add_argument("-chans", "--channels", dest="channels", metavar="",
        nargs="+", type=int, default=None,
        help="Channels numbers to select. Specify as space separated list"
    )

    return ps


if __name__ == "__main__":
    ps = parser().parse_args()

    if ps.cubes is not None:
        for cube in ps.cubes:
            single_cube_file(cube_name=cube, oname=ps.output, channels=ps.channels)

    if ps.files is not None:
        for file_grp in ps.files:
            if ps.channels is not None:
                print(f"Selecting {len(ps.channels)} out of {len(file_grp)}")
                print(f"Channels: {ps.channels}")
                file_grp = [file_grp[c] for c in ps.channels]
            multiple_single_files(files=file_grp, oname=ps.output)


# How to run the script. Can be imported into ipython
#inf_name = "eds.txt"
#multiple_single_files(inf_name)
#
