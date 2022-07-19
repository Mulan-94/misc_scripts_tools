import numpy as np
import argparse
import matplotlib.pyplot as plt
import warnings

from astropy.io import fits
from astropy.wcs import WCS
from glob import glob
from matplotlib import ticker

import sys
import os
sys.path.append(f"{os.environ['HOME']}/git_repos/misc_scripts_n_tools/fits_related/")

import random_fits_utils as rfu
from ipdb import set_trace


warnings.filterwarnings("ignore", module="astropy")

FIGSIZE = (16,9)
EXT = ".svg"

def set_image_projection(image_obj):
    """
    image_obj:
        Matplotlib axis object from image e.g ax from
        fig, ax = plt.subplots(subplots_kw={"projection": wcs})
    """
    image_obj.get_transform('fk5')
    ra, dec = image_obj.coords
    ra.set_major_formatter('hh:mm:ss')
    dec.set_major_formatter('dd:mm:ss')
    image_obj.set_xlabel('J2000 Right Ascension')
    image_obj.set_ylabel('J2000 Declination')
    return image_obj


######################################################################################
## Plot polarised intensity for each channel

def plot_polarised_intensity(data=None, im_name=None, mask=None, oup=None, ref_image=None):
    """
    Input
    -----
    data:
        Numpy array containing image data. If this is specified, the rest of 
        the args are ignored.
    im_name
        Masked polarisation intensity im_name
    mask_name
        Name of mask to be applied
    """

    # if strings read the image, otherwise assume data is in numpy array
    if isinstance(data, str):
        data = rfu.read_image_cube(data, mask=False)["data"]
    if mask is not None and isinstance(mask, str):
        mask = rfu.read_image_cube(mask)["data"]
    
    chans = data.shape[0]
    mid_chan = int(np.median(np.arange(chans)))
    data = np.abs(data)

    ydim, xdim = np.where(mask == False)
    wiggle = 10
    xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
    ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle

    wcs = rfu.read_image_cube(ref_image)["wcs"]

    """
    fig, ax = plt.subplots(
        figsize=FIGSIZE, ncols=6, nrows=int(np.ceil(chans/6)), sharex=True,
        sharey=True, gridspec_kw={"wspace":0 , "hspace":0}, subplot_kw={'projection': wcs})
    ax = ax.flatten()

    for chan in range(chans):
        lpol = np.ma.masked_array(data=data[chan], mask=mask)
        image = ax[chan].imshow(lpol, origin="lower", cmap="coolwarm", vmin=0, vmax=0.3)
        plt.xlim(*xlims)
        plt.ylim(*ylims)

    fig.subplots_adjust(right=0.8)
    # left, bottom, width, height
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(image, cax=cbar_ax)
    fig.tight_layout()
    """

    fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
    lpol = np.ma.masked_array(data=data[mid_chan], mask=mask)
    image = ax.imshow(lpol, origin="lower", cmap="coolwarm", vmin=0, vmax=0.1)
    
    ax = set_image_projection(ax)
    ax.tick_params(axis="x", top=False)
    ax.tick_params(axis="y", right=False)
    
    ax.set_title(f"Polarised Intensity of Middle Channel: "+f"{mid_chan}".zfill(4))
    plt.xlim(*xlims)
    plt.ylim(*ylims)
    fig.colorbar(image, shrink=0.85)
    fig.tight_layout()
    oname = oup + f"-lin_pol{EXT}"
    fig.savefig(oname)
    print(f"Output is at: {oname}")


######################################################################################
## Plot fractional polzn for each channel

def plot_fractional_polzn(data, mask, oup=None, ref_image=None):
    """
    Input
    -----
    data:
        Numpy array containing image data.  Otherwise, it is the name of
        the iamge where the data will be gotten from 
    mask_name  mask data
        Name of mask to be applied. or amsk. This is amust fro mny own sake
    """
   
    print("Starting frac pol plot")
   
    # if strings read the image, otherwise assume data is in numpy array
    if isinstance(data, str):
        data = rfu.read_image_cube(data, mask=False)["data"]
    if mask is not None and isinstance(mask, str):
        mask = rfu.read_image_cube(mask)["data"]
    
    chans = data.shape[0]
    mid_chan = int(np.median(np.arange(chans)))
    data = np.abs(data)

    ydim, xdim = np.where(mask == False)
    wiggle = 10
    xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
    ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle

    wcs = rfu.read_image_cube(ref_image)["wcs"]
    
    """
    fig, ax = plt.subplots(
        figsize=FIGSIZE, ncols=6, nrows=int(np.ceil(chans/6)), sharex=True,
        sharey=True, gridspec_kw={"wspace":0 , "hspace":0})
    ax = ax.flatten()

    for chan in range(chans):
        fpol = np.ma.masked_array(data=data[chan], mask=mask)
        image = ax[chan].imshow(fpol, origin="lower", cmap="coolwarm", vmin=0, vmax=0.3)
        plt.xlim(*xlims)
        plt.ylim(*ylims)

    fig.subplots_adjust(right=0.8)
    # left, bottom, width, height
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(image, cax=cbar_ax)
    fig.tight_layout()
    """

    fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
    fpol = np.ma.masked_array(data=data[mid_chan], mask=mask)
    image = ax.imshow(fpol, origin="lower", cmap="coolwarm", vmin=0, vmax=0.3)
    
    ax = set_image_projection(ax)
    ax.tick_params(axis="x", top=False)
    ax.tick_params(axis="y", right=False)
    
    ax.set_title(f"Fractional polarisation of Middle Channel: "+f"{mid_chan}".zfill(4))
    plt.xlim(*xlims)
    plt.ylim(*ylims)
    fig.colorbar(image, shrink=0.85)
    fig.tight_layout()
    oname = oup + f"-fpol{EXT}"
    fig.savefig(oname)
    print(f"Output is at: {oname}")

    return
    


######################################################################################
## contours plotting and vectors
######################################################################################

# ref https://stackoverflow.com/questions/40939821/how-to-plot-a-vector-field-over-a-contour-plot-in-matplotlib


def add_contours(axis, data, levels=None):
    """
    contours: total intensity

    Input
    -----
    axis:
        Mpl axes object onto which to add contours
    data:
        Data from which contours will be calculated
    levels: tuple; optional
        Levels for the contours
    """

    # some functions for the contour levels that can be used
    sqrt = lambda x: np.sqrt(x)
    linear = lambda x: x
    square = lambda x: np.square(x)
    power = lambda x, alpha: ((alpha**x)-1) / (alpha -1)


    if levels is None:
        # setup contour levels factor of root two between n+1 and n
        lstep = 0.5
        # I'm setting the first level at 0.01, because 2**0 is 1
        levels = 2**np.arange(0, data.max()+10, lstep)*0.001
        levels = np.ma.masked_greater(levels, data.max()).compressed()

    # contour lines
    axis.contour(data, colors="k", linewidths=0.5, origin="lower", levels=levels)
    # ax.clabel(cs, inline=True, fontsize=10)


    #filled contours
    cs = axis.contourf(data, cmap="coolwarm", origin="lower", levels=levels,
                locator=ticker.LogLocator())
    plt.colorbar(cs)

    return axis
    

def add_magnetic_vectors(axis, fpol_data, pangle_data):
    """
    vector length: degree of poln (fpol)
    orientation: position angle
    """
    skip = 5
    slicex = slice(None, fpol_data.shape[0], skip)
    slicey = slice(None, fpol_data.shape[-1], skip)
    col, row = np.mgrid[slicex, slicey]

    # get M vector by rotating E vector by 90
    pangle_data = pangle_data[slicex, slicey] + (np.pi/2)
    fpol_data = fpol_data[slicex, slicey]

    # nornalize this
    
    fpol_data = np.ma.masked_greater(fpol_data, 1)
    scales = fpol_data / fpol_data.max()
    u = np.cos(pangle_data) * scales
    v = np.sin(pangle_data) * scales


    # ax.contourf(row, col, Z)
    # plt.axis([2200, 2770, 2050, 2250])
    qv = axis.quiver(
        row, col, u, v, angles="xy", pivot='tail', headlength=4,
        width=0.0008, scale=5, headwidth=0)

    return axis


def plot_intensity_vectors(i_name, fpol_name, pa_name, mask=None, oup=None):
    i_data = rfu.get_masked_data(i_name, mask)
    fpol_data = rfu.get_masked_data(fpol_name, mask)
    pa_data = rfu.get_masked_data(pa_name, mask)
    
    mask = i_data.mask

    ydim, xdim = np.where(mask == False)
    wiggle = 10
    xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
    ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle

    wcs = rfu.read_image_cube(i_name)["wcs"]

    fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
    ax = set_image_projection(ax)

    ax = add_contours(ax, i_data)
    ax = add_magnetic_vectors(ax, fpol_data, pa_data)
    ax.tick_params(axis="x", top=False)
    ax.tick_params(axis="y", right=False)
    plt.xlim(*xlims)
    plt.ylim(*ylims)
    # fig.tight_layout()
    oname = oup + f"-intense_mfield.png"
    print(f"Output is at: {oname}")
    fig.savefig(oname)



###################################
# plot lobes with the histograms left and right

def plot_rm_for_lobes(rot_meas_image, e_mask, w_mask, vmin=None, vmax=None, oup=None, ref_image=None):
    rot_meas = rfu.read_image_cube(rot_meas_image)["data"]
    w_lobe_mask = rfu.read_image_cube(w_mask, mask=True)["data"]
    e_lobe_mask = rfu.read_image_cube(e_mask, mask=True)["data"]

    w_lobe = rfu.get_masked_data(rot_meas_image, w_mask)
    e_lobe = rfu.get_masked_data(rot_meas_image, e_mask)
    
    lobes_mask = np.bitwise_and(w_lobe_mask, e_lobe_mask)
    lobes = np.ma.masked_array(rot_meas, mask=lobes_mask)


    fig = plt.figure(figsize=FIGSIZE)
    
    wcs = rfu.read_image_cube(ref_image or rot_meas_image)["wcs"]
    image = plt.subplot2grid((3,3), (1,0), rowspan=2, colspan=2, projection=wcs)
    ca = image.imshow(
        lobes, origin="lower", cmap="magma", vmin=vmin, vmax=vmax, aspect="equal")
    plt.colorbar(ca, location="right", shrink=0.90, pad=0.01, label="RM", drawedges=False)

    image = set_image_projection(image)

    # swich of these ticks
    image.tick_params(axis="x", top=False)
    image.tick_params(axis="y", right=False)
    # #image.axis("off")

    # so that I can zoom into the image easily and automatically
    ydim, xdim = np.where(lobes.mask == False)
    wiggle = 10
    plt.xlim(np.min(xdim)-wiggle, np.max(xdim)+wiggle)
    plt.ylim(np.min(ydim)-wiggle, np.max(ydim)+wiggle)

    west_hist = plt.subplot2grid((3,3), (1,2), rowspan=2, colspan=1)
    west_hist.hist(w_lobe.compressed(), bins=20, log=True,
        orientation="horizontal",fill=False, ls="--", lw=1, edgecolor="blue", 
        histtype="step")
    west_hist.yaxis.tick_right()

    west_hist.set_title("Western Lobe (Right Hand) RM Distribution")
    west_hist.xaxis.set_visible(False)

    east_hist = plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=1)
    east_hist.hist(e_lobe.compressed(), bins=20, log=False,
        orientation="vertical", fill=False, ls="--", lw=1, edgecolor="blue",
        histtype="step")
    east_hist.xaxis.tick_top()
    east_hist.set_title("Eastern Lobe (Left Hand) RM Distribution")
    east_hist.yaxis.set_visible(False)
    
    plt.subplots_adjust(wspace=.01, hspace=0)

    fig.tight_layout()
    oname = oup + f"-lobes_rm{EXT}"
    fig.savefig(oname)
    print(f"Output is at: {oname}")
    


def make_masks_from_ricks_data():

    fnames = [
    "band-l-and-c-LCPIC-10-NBL.RM.1.FITS",
    "band-l-and-c-LCPIC-10.RM10.2.FITS", #ricks_data.mask.fits
    "band-l-LPIC-ACBD-12.RM5.1.FITS",
    "band-l-and-c-LCPIC-10.RM10.1.FITS",
    "band-l-LPIC-ACBD-12.RM2.1.FITS",
    ]

    i_image = "I-MFS-image.fits"

    for fname in fnames:
        rfu.make_mask(fname, xy_dims=(572,572))


    fig, ax = plt.subplots(figsize=FIGSIZE, sharex=True, sharey=True, ncols=2, gridspec_kw={"wspace": 0 })
    ax[0].imshow(mask, origin="lower")
    ax[0].vlines(280, 0, 571)
    ax[0].hlines(280, 0, 571)
    ax[1].imshow(pica, origin="lower")
    ax[1].vlines(280, 0, 571)
    ax[1].hlines(280, 0, 571)
    # plt.show()


def parser():
    ps = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        description="Script to genernetae some plots for my paper")
    ps.add_argument("-rim", "--ref-i-image", dest="ref_image", 
        help="Reference I image for the contour plots")
    ps.add_argument("--cube-name", dest="cube_names", nargs="+",
        help="Input I q AND U same resolution cubes")
    ps.add_argument("--input-maps", dest="prefix", 
        required=True,
        help="Input prefix for the fpol rm and whatever maps")
    ps.add_argument("--mask-name", dest="mask_name",
        help="Name of the full field mask to use"
        )
    ps.add_argument("-elm", "--e-lobe-mask", dest="elobe_mask",
        help="Mask for the eastern lobe"
        )
    ps.add_argument("-wlm", "--w-lobe-mask", dest="wlobe_mask",
        help="Mask for the western lobe"
        )
    ps.add_argument("-o", "-output-dir", dest="output_dir",
        help="Where to dump the outputs"
        )
    return ps

######################################################################################
# Main
######################################################################################

if __name__ == "__main__":
    
    opts = parser().parse_args()

    # names for the various rm products
    map_names = {
        "amp" : f'{opts.prefix}-p0-peak-rm.fits',
        "angle" : f'{opts.prefix}-PA-pangle-at-peak-rm.fits',
        "fpol" : f'{opts.prefix}-FPOL-at-center-freq.fits',
        "rm" : f'{opts.prefix}-RM-depth-at-peak-rm.fits'
    }

    images = [map_names[_] for _ in "amp angle rm".split()]
    
    pica_i_data = rfu.get_masked_data(opts.ref_image, opts.mask_name)
    pangle_data = rfu.read_image_cube(map_names["angle"])["data"]
    pica_mask = pica_i_data.mask

    stokes = {}
    for cube in opts.cube_names:
        stoke = os.path.basename(cube)[0]
        stokes[stoke] = rfu.read_image_cube(cube)["data"]

    stokes["l_pol"] = stokes["q"] + 1j*stokes["u"]
    stokes["f_pol"] = (stokes["q"]/stokes["i"]) + ((1j*stokes["u"])/stokes["i"])
    
    # plot lobes and their dispersion
    plot_rm_for_lobes(
        rot_meas_image=map_names["rm"],
        e_mask=opts.elobe_mask, w_mask=opts.wlobe_mask,
        vmin=-100, vmax=100, oup=opts.prefix, ref_image=opts.ref_image)


    # plot fpol
    plot_fractional_polzn(stokes["f_pol"], oup=opts.prefix, mask=pica_mask,
        ref_image=opts.ref_image)
    
    plot_polarised_intensity(
        stokes["l_pol"], oup=opts.prefix, ref_image=opts.ref_image,
        mask=pica_mask)

    plot_intensity_vectors(
        opts.ref_image, map_names["fpol"], map_names["angle"],
        mask=opts.mask_name, oup=opts.prefix)


    """
    Running this script with
    
    python qu_pol/test_paper_plots.py --input-maps $prods/initial -rim i-mfs.fits --cube-name $conv_cubes/*-conv-image-cube.fits --mask-name $mask_dir/true_mask.fits -elm $mask_dir/east-lobe.fits -wlm $mask_dir/west-lobe.fits -o $prods/some-plots
    """

