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
    # save this beam information into beams file
    np.savez(os.path.join(os.path.dirname(os.path.realpath(outname)), "beams"),
        freqs=freqs, bmajs=bmajs, bmins=bmins)

    fig, ax = plt.subplots(figsize=(16,9), ncols=2, sharex=True, sharey=True)
    ax[0].plot(freqs, bmajs, "bo", markersize=4)
    ax[0].axhline(bmajs.max(), linestyle="--", linewidth=1)
    bm_max = np.argmax(bmajs)
    maxs = freqs[bm_max]
    #if maxs.size >= 1:
    #    maxs = maxs[0]
    ax[0].annotate(f"{bmajs.max():.3}", 
        xy=(maxs, bmajs.max()), color="red")


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
    
    bm_max = np.argmax(bmins)
    maxs = freqs[bm_max]

    ax[1].annotate(f"{bmins.max():.3}", 
        xy=(maxs, bmins.max()), 
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



# merging to model some beam trend blabla
#------------------------------------------

def model(x, data):
    res = np.ma.polyfit(x, data, deg=1)
    reg_line = np.poly1d(res)(x)
    return reg_line


def mask_them(mask, compress=True, **kwargs):
    for key, value in kwargs.items():
        kwargs[key] = np.ma.masked_array(data=kwargs[key], mask=mask)
        if compress:
            kwargs[key] = kwargs[key].compressed()
    return kwargs


def plot_beam_axis(**kwargs):
    plt.close("all")
    # ma_res,ma_lim,mi_res,mi_lim
    bmaj, bmin, freqs, ma_res, ma_lim, mi_res, mi_lim, ma_model,mi_model = list(
            map(kwargs.get, 
                ("bma,bmi,freqs,ma_res,ma_lim,"+
                "mi_res,mi_lim,ma_model,mi_model").split(",")))

    fig, ax = plt.subplots(figsize=(16, 9), ncols=2, nrows=3, squeeze=False,
        sharex=True, sharey="row", gridspec_kw={"wspace":0})

    ax[0, 0].plot(freqs, bmin, "bo", alpha=0.5, markersize=5)
    ax[0, 0].plot(freqs, mi_model, color="red")
    ax[0, 0].set_title("Valid channels: BMIN")
    for _ in np.nonzero(~bmin.mask)[0]:
        ax[0,0].text(freqs[_], bmin[_], f"{_}".zfill(2), fontsize=4, ha="center")

    ax[0, 1].plot(freqs, bmaj, "bo", alpha=0.5, markersize=5)
    ax[0, 1].plot(freqs, ma_model, color="red")
    ax[0, 1].set_title("Valid channels: BMAJ")
    ax[0, 1].xaxis.set_ticks(np.arange(0, bmaj.size, 5))
    for _ in np.nonzero(~bmaj.mask)[0]:
        ax[0,1].text(freqs[_], bmaj[_], f"{_}".zfill(2), fontsize=4, ha="center")

    ax[1, 0].plot(freqs.data, bmin.data, "bo", alpha=0.5, markersize=5)
    ax[1, 0].plot(freqs.data, mi_model.data, color="red")
    ax[1, 0].set_title("All Channels: BMIN")
    for _ in np.nonzero(~bmin.mask)[0]:
        ax[1,0].text(freqs[_], bmin[_], f"{_}".zfill(2), fontsize=4, ha="center")


    ax[1, 1].plot(freqs.data, bmaj.data, "bo", alpha=0.5, markersize=5)
    ax[1, 1].plot(freqs.data, ma_model.data, color="red")
    ax[1, 1].set_title("All Channels: BMAJ")
    for _ in np.nonzero(~bmaj.mask)[0]:
        ax[1,1].text(freqs[_], bmaj[_], f"{_}".zfill(2), fontsize=4, ha="center")

    ax[2, 0].stem(freqs.data, np.abs(mi_res.data))
    ax[2, 0].axhline(mi_lim, ls="--",color="k")
    ax[2, 1].stem(freqs.data, np.abs(ma_res.data))
    ax[2, 1].axhline(ma_lim, ls="--",color="k")
    ax[2, 0].set_title("Residuals")
    ax[2, 1].set_title("Residuals")
    ax[2, 1].set_xlabel("Frequncies")
    
    oname = kwargs.get("oname", "beams.png")
    fig.savefig(oname, bbox_inches="tight", dpi=300)


def model_beam_variation(freqs, bm, perc=65):
    """
    Get the regression line and return the limit value
    """
    mod = model(freqs, bm)
    residual = bm - mod
    lim = np.percentile(np.abs(residual.compressed()), perc)
    mask = np.ma.masked_greater(np.abs(residual), lim).mask
    return mod, residual, mask, lim


def read_and_plot_beams(fname="beams.npz", percentile=80):
    bm = dict(np.load(fname))

    bma, bmi, freqs = bm["bmajs"], bm["bmins"], bm["freqs"]
    freqs = np.arange(freqs.size)

    # remove the zero values
    mask = np.ma.masked_equal(bma, 0).mask
    masked = mask_them(mask, compress=False, bma=bma, bmi=bmi, freqs=freqs)
    bma, bmi, freqs = list(map(masked.get, "bma,bmi,freqs".split(",")))

    # model the bmaj and bmin: model, residual,mask, percentile_limit
    ma_model, ma_res, ma_mask, ma_lim = model_beam_variation(freqs, bma, perc=percentile)
    mi_model, mi_res, mi_mask, mi_lim = model_beam_variation(freqs, bmi, perc=percentile)

    mask = np.logical_or(mi_mask, ma_mask)

    masked = mask_them(mask ,compress=False, bma=bma, bmi=bmi, freqs=freqs,
        ma_model=ma_model, mi_model=mi_model)

    bma,bmi,freqs,ma_model,mi_model = list(
            map(masked.get, "bma,bmi,freqs,ma_model,mi_model".split(",")))

    chans = [f"{_}".zfill(4) for _ in np.nonzero(~mask)[0]]

    print("Valid channels: {}".format(len(chans)))
    print("Proposed selected channels: ", *chans)

   
    plot_beam_axis(bmi=bmi, bma=bma, freqs=freqs, ma_res=ma_res,
        ma_lim=ma_lim, mi_res=mi_res, mi_lim=mi_lim, ma_model=ma_model,
        mi_model=mi_model,
        oname=os.path.join(os.path.dirname(os.path.realpath(fname)), "beams.png"))

    with open("suggested-sel-chan.txt", "w") as fil:
        for chan in chans:
            fil.writelines(chan+"\n")




#------------------------------------------


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

    ps.add_argument("-p", "--percentile", dest="percentile", metavar="",
        type=float, default=80, help="For the selected channels, n-th percentile. Only data below this value will be included from the beam."
    )

    ps.add_argument("-s", "--select", action="store_true", dest="select", 
        help="Try and suggest valid channels. Use in conjuction with '-p'.")

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

    if ps.select:    
        read_and_plot_beams(
            fname=os.path.join(os.path.dirname(os.path.realpath(ps.output)), "beams.npz"),
            percentile=ps.percentile
            )


# How to run the script. Can be imported into ipython
#inf_name = "eds.txt"
#multiple_single_files(inf_name)
#
