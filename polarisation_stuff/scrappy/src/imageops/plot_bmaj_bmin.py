#############
## Lexy plotting BMAJ bmin BLABAL
#######################


import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from astropy.io import fits
from glob import glob
from natsort import natsorted


from utils.logger import logging, LOG_FORMATTER, setup_streamhandler
snitch = logging.getLogger(__name__)
snitch.addHandler(setup_streamhandler())
snitch.setLevel("INFO")

def read_fits(imname):
    outs = {}
    hdr = fits.getheader(imname)
    for _ in "BMAJ BMIN CRVAL3".split():
        if "C" in _:
            outs[_] = hdr[_] / 1e9
        else:    
            outs[_] = hdr[_]
    snitch.info(f"Reading         :{imname}")
    return outs


def read_cube_fits(imname, channels=None):
    outs = []

    with fits.open(imname) as hdul:
        if channels is not None:
            snitch.info(f"Selecting {len(ps.channels)} channels out of {hdul[0].header['NAXIS3']}")
            snitch.info(f"Channels: {channels}")
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
    snitch.info("Plotting beam dimensions")
    fig, ax = plt.subplots(figsize=(16,9), ncols=2, sharex=True)
    ax[0].plot(freqs, bmajs, "bo", markersize=4)
    ax[0].axhline(bmajs.max(), linestyle="--", linewidth=1)
    bm_max = np.argmax(bmajs)
    maxs = freqs[bm_max]
    #if maxs.size >= 1:
    #    maxs = maxs[0]
    ax[0].annotate(f"{bmajs.max():.3}", 
        xy=(maxs, bmajs.max()), color="red")

    
    ax[0].set_xlabel("Freq GHz")
    ax[0].set_ylabel("BMAJ")
    # ax[0].set_title(title)
    ax[0].minorticks_on()
    ax[0].grid(True)

    ax[1].plot(freqs, bmins, "bo", markersize=4)
    ax[1].axhline(bmins.max(), linestyle="--", linewidth=1)
    ax[1].minorticks_on()
    ax[1].grid(True)
    
    bm_max = np.argmax(bmins)
    maxs = freqs[bm_max]

    ax[1].annotate(f"{bmins.max():.3}", 
        xy=(maxs, bmins.max()), 
        color="red")
    
    ax[1].set_xlabel("Freq GHz")
    ax[1].set_ylabel("BMIN")
    # ax[1].set_title(title)

    fig.tight_layout()
    snitch.info(f"Saving file at: {outname}")
    fig.savefig(outname)


def get_params(pairs):
    snitch.info("Populating beam parameters for all images")
    bmajs = np.array([_["BMAJ"] for _ in pairs])
    bmins = np.array([_["BMIN"] for _ in pairs])
    freqs = np.array([_["CRVAL3"] for _ in pairs])
    return bmajs, bmins, freqs



def get_and_plot_beam_info(indir=None, search="*[0-9]-I-image.fits", dump=".",
    prefix=None, oname="beam_vs_freq.png"):
    """
    read file containing filenames for the images to be processed
    these are stored in e.g eds.txt
    if a file containing these file names is not avaliable, 
    just use the files as they are
    """

    data = natsorted(glob(os.path.join(indir, search)))

    pairs = [read_fits(dat) for dat in data]
    bmajs, bmins, freqs = get_params(pairs)


    # save this beam information into beams file
    np.savez(os.path.join(dump, "beams"), freqs=freqs, bmajs=bmajs, bmins=bmins)

    oname = os.path.join(dump, oname if prefix is None else f"{prefix}-{oname}")
    plotme(freqs, bmajs, bmins, oname, title="All freqs")

    return


# def single_cube_file(cube_name, oname=None, channels=None):
#     ## in the case of multiple data cubes
#     pairs = read_cube_fits(cube_name, channels=channels)
#     bmajs, bmins, freqs = get_params(pairs)
#     if oname is None:
#         oname = f"bmaj-bmin-vs-freq-{cube_name}.png"
#     plotme(freqs, bmajs, bmins, oname, title="All freqs")



#------------------------------------------============
# Some Basic automated channel selection using wsums
#------------------------------------------============
def read_wsums(image):
    hdr = fits.getheader(image)
    return hdr["WSCVWSUM"]


def channel_selection(folder, dump, threshold=0.5, autoselect=True, sel=None):
    """
    folder: str
        The directory containing your intended images
    threshold: float
        A value between zero and one. We look at the individual images wsum
        which is normalised by the maximum available. This threshold is checked
        against the normalised wsum. Values below the threshold are ignored.
    """
    snitch.info("Starting channel selection")

    images = natsorted(glob(os.path.join(folder, "*[0-9]-I-image.fits")))
    owsums = np.zeros(len(images))
    for i, image in enumerate(images):
        owsums[i] = read_wsums(image)

    ONAME = os.path.join(dump, "orig-wsums.txt")
    snitch.info(f"Saving WSUMS to: {ONAME}")
    np.savetxt(ONAME, owsums)

    
    if autoselect:
        wsums = np.round(owsums/owsums.max(), 2)
        not_sel, = np.where(np.ma.masked_less_equal(wsums, 0.5).mask==True)
        ONAME = os.path.join(dump, "not-selected-channels.txt")
        snitch.info(f"Saving channel not selected to: {ONAME}")
        with open(ONAME, "w") as file:
            file.writelines([f"{_}".zfill(4) + "\n" for _ in not_sel])

        sel, = np.where(np.ma.masked_less_equal(wsums, 0.5).mask==False)
        snitch.info(f"{sel.size} of {wsums.size} channels selected.")
        ONAME = os.path.join(dump, "selected-channels.txt")
        snitch.info(f"Saving channel sel to: {ONAME}")
        with open(ONAME, "w") as file:
            file.writelines([f"{_}".zfill(4) + "\n" for _ in sel])

        ONAME = os.path.join(dump, "wsums.txt")
        snitch.info(f"Saving WSUMS of selected channels to: {ONAME}")
        np.savetxt(ONAME, owsums[sel])
       

    else:
        snitch.info(f"Reading custom selected channel file: {sel}")
        sel = np.loadtxt(sel)
        not_sel = np.zeros(len(images))
        not_sel[sel] = 1
        not_sel, = np.where(not_sel==0)

        ONAME = os.path.join(dump, "wsums.txt")
        snitch.info(f"Saving WSUMS of selected channels to: {ONAME}")
        np.savetxt(ONAME, owsums[sel])
       

        ONAME = os.path.join(dump, "not-selected-channels.txt")
        snitch.info(f"Saving channel not selected to: {ONAME}")
        with open(ONAME, "w") as file:
            file.writelines([f"{_}".zfill(4) + "\n" for _ in not_sel])

    
    return sel, not_sel


def read_and_plot_beams2(folder, dump=".", prefix=None, beam_file="beams.npz", threshold=0.5, 
    sel=None, autoselect=True):
    """
    folder: str
        Where the input images are
    dump: str
        Where to dump the outputs
    """
    
    sel, not_sel = channel_selection(folder, dump, threshold=threshold,
        sel=sel, autoselect=autoselect)
    
    chans = [f"{_}".zfill(4) for _ in sel]
    
    bm = dict(np.load(os.path.join(dump, beam_file)))
    bma, bmi, freqs = bm["bmajs"], bm["bmins"], bm["freqs"]

    # write the largest beam available
    ONAME = os.path.join(dump, "beam-dims.txt")
    with open(ONAME, "w") as file:
        file.write(f"{bma[sel[0]]}\n")
        file.write(f"{bmi[sel[0]]}\n")
        file.write(f"{freqs[sel[0]]}\n")


    
    ONAME = os.path.join(dump, "frequencies.txt")
    snitch.info(f"Saving selected freqs : {ONAME}")
    np.savetxt(ONAME, freqs[sel])

    # chans = np.arange(bma.size)
    freqs = np.arange(bma.size)
    
    plt.close("all")
    fig, ax = plt.subplots(figsize=(16, 9), ncols=2, nrows=1, squeeze=False,
        sharex=True)

    ax[0, 1].plot(freqs[sel], bmi[sel], "bo", alpha=0.5, markersize=5)
    ax[0, 1].plot(freqs[not_sel], bmi[not_sel], "ro", alpha=0.5, markersize=5)
    ax[0, 1].minorticks_on()
    ax[0, 1].grid(True)
    ax[0, 1].set_ylabel("BMIN")
    # ax[0, 1].set_xlabel("Freq [GHz]")
    ax[0, 1].set_xlabel("Channel Number")

    ax[0, 0].plot(freqs[sel], bma[sel], "bo", alpha=0.5, markersize=5)
    ax[0, 0].plot(freqs[not_sel], bma[not_sel], "ro", alpha=0.5, markersize=5)
    ax[0, 0].minorticks_on()
    ax[0, 0].grid(True)
    ax[0, 0].set_ylabel("BMAJ")
    # ax[0, 0].set_xlabel("Freq [GHz]")
    ax[0, 0].set_xlabel("Channel Number")
    
    if prefix is None:
        ONAME = os.path.join(dump, "selected_beam_vs_freq.png")
    else:
        ONAME = os.path.join(dump, f"{prefix}-selected_beam_vs_freq.png")
    fig.savefig(ONAME, bbox_inches="tight", dpi=300)

    return
    


#------------------------------------------


def parser():
    ps = argparse.ArgumentParser()

    
    ps.add_argument("idir", metavar="", type=str,
        help="Directory where to find the input images"
    )

    ps.add_argument("-o", "--output", dest="output", metavar="",
        type=str, default=".", 
        help="Directory where to dump outputs. Default is current directory."
    )
    ps.add_argument("-search", "--search", type=str,
        help="Regex of what string to search in the directory",
        default="*[0-9q]-I-image.fits")

    ps.add_argument("-t", "--threshold", dest="threshold", metavar="",
        type=float, default=0.5, 
        help="""Channels with WSUM/WSUM.max() below this value will be excluded.
        Only used if -s is active"""
    )
    ps.add_argument("-sc", "--select-channels", dest="sel", default=None,
        type=str, help="Custom channel selections"
        )
    ps.add_argument("-as", "--auto-select", action="store_true", dest="auto_select", 
        help="Try and suggest valid channels for selection. Use in conjuction with '-t'.")
    ps.add_argument("-pre", "--prefix", type=str, default="00", 
        help="Prefix to append on the output images")
    return ps


def main():
    ps = parser().parse_args()

    # get info about beams and plot them
    get_and_plot_beam_info(ps.idir, search=ps.search, dump=ps.output)
       
    if ps.auto_select or ps.sel:
        read_and_plot_beams2(ps.idir, dump=ps.output, beam_file="beams.npz",
            threshold=ps.threshold, sel=ps.sel, autoselect=ps.auto_select)


def console():
    """A console run entry point for setup.cfg"""
    main()
    snitch.info("Bye :D !")
    

if __name__ == "__main__":
    console()
