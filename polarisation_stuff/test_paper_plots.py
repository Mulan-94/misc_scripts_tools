import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from glob import glob
from matplotlib import ticker

import sys
import os
sys.path.append(f"{os.environ['HOME']}/git_repos/misc_scripts_n_tools/fits_related/")

import random_fits_utils as rfu
from ipdb import set_trace

######################################################################################
## Plot polarised intensity for each channel

def plot_polarised_intensity(data=None, im_name=None, mask_name=None):
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
    if data is None:
        if mask is None:
            data = rfu.read_image_cube(im_name, mask=False)["data"]
        else:
            data = rfu.get_masked_data(im_name, mask)
    fig, ax = plt.subplots(ncols=6, nrows=5, sharex=True, sharey=True,
                           gridspec_kw={"wspace":0 , "hspace":0})
    ax = ax.flatten()
    chans = data.shape[0]
    for chan in range(chans):
        lpol = np.abs(data[chan])
        ax[chan].imshow(lpol, cmap="coolwarm", origin="lower")
    fig.tight_layout()
    fig.savefig("lin_pol.svg")
    plt.show()

######################################################################################
## Plot fractional polzn for each channel

def plot_fractional_polzn(data=None, im_name=None, mask_name=None):
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
    if data is None:
        if mask is None:
            data = rfu.read_image_cube(im_name, mask=False)["data"]
        else:
            data = rfu.get_masked_data(im_name, mask)

    fig, ax = plt.subplots(ncols=6, nrows=5, sharex=True, sharey=True,
                            gridspec_kw={"wspace":0 , "hspace":0})
    ax = ax.flatten()
    chans = data.shape[0]
    for chan in range(chans):
        fpol = np.abs(data[chan])
        ax[chan].imshow(np.log(fpol), origin="lower", cmap="coolwarm")
    fig.tight_layout()
    #plt.colorbar()
    fig.savefig("fpol.svg")
    plt.show()


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
        # setup contour levels
        lstep = 0.5
        levels = 2**np.arange(0, data.max()+10, lstep)*0.01
        levels = np.ma.masked_greater(levels, data.max()).compressed()

    # fig, ax = plt.subplots(figsize=(15,15))

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
    skip = 8
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
    axis.quiver(row, col, u, v, angles="xy", pivot='tail', headlength=4, width=0.0008, scale=10)

    return axis

def plot_intensity_vectors(i_name, fpol_name, pa_name, mask_name):
    i_data = rfu.get_masked_data(i_name, mask_name)
    fpol_data = rfu.get_masked_data(fpol_name, mask_name)
    pa_data = rfu.get_masked_data(pa_name, mask_name)
    
    fig, ax = plt.subplots(figsize=(15,15))
    ax = add_contours(ax, i_data)
    ax = add_magnetic_vectors(ax, fpol_data, pa_data)
    fig.tight_layout()
    fig.savefig("intense_mfield.svg")
    plt.show()




###################################
# plot lobes with the histograms left and right

def plot_rm_for_lobes(rot_meas_image, e_mask, w_mask, vmin=None, vmax=None):
    rot_meas = rfu.read_image_cube(rot_meas_image)["data"]
    w_lobe_mask = rfu.read_image_cube(w_mask, mask=True)["data"]
    e_lobe_mask = rfu.read_image_cube(e_mask, mask=True)["data"]

    w_lobe = rfu.get_masked_data(rot_meas_image, w_mask)
    e_lobe = rfu.get_masked_data(rot_meas_image, e_mask)
    
    lobes_mask = np.bitwise_and(w_lobe_mask, e_lobe_mask)
    lobes = np.ma.masked_array(rot_meas, mask=lobes_mask)


    fig = plt.figure(figsize=(10,5))

    wcs = rfu.read_image_cube(rot_meas_image)["wcs"]
    image = plt.subplot2grid((2,4), (0,1), rowspan=2, colspan=2, projection=wcs)
    ca = image.imshow(lobes, origin="lower", cmap="magma", vmin=vmin, vmax=vmax, aspect="equal")
    plt.colorbar(ca, location="bottom")
    image.get_transform('fk5')
    ra, dec = image.coords
    ra.set_major_formatter('hh:mm:ss')
    dec.set_major_formatter('dd:mm:ss')
    image.set_xlabel('J2000 Right Ascension')
    image.set_ylabel('J2000 Declination')

    # #image.axis("off")
    # plt.xlim(210, 360)
    # plt.ylim(230, 330)
    

    west_hist = plt.subplot2grid((2,4), (0,3), rowspan=2)
    west_hist.hist(w_lobe.compressed(), bins=20, log=True, orientation="horizontal")
    #west_hist.axis("off")

    east_hist = plt.subplot2grid((2,4), (0,0), rowspan=2)
    east_hist.hist(e_lobe.compressed(), bins=20, log=True, orientation="horizontal")
    east_hist.sharey(west_hist)
    #east_hist.axis("off")
    plt.subplots_adjust(wspace=.5, hspace=0)

    fig.tight_layout()
    fig.savefig("lobes_rm.svg")
    plt.show()


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

    set_trace()

    fig, ax = plt.subplots(figsize=(10,5), sharex=True, sharey=True, ncols=2, gridspec_kw={"wspace": 0 })
    ax[0].imshow(mask, origin="lower")
    ax[0].vlines(280, 0, 571)
    ax[0].hlines(280, 0, 571)
    ax[1].imshow(pica, origin="lower")
    ax[1].vlines(280, 0, 571)
    ax[1].hlines(280, 0, 571)
    plt.show()



######################################################################################
# Main
######################################################################################

prefix = "turbo"
postfix = {
    "amp" : f'{prefix}-p0-peak-rm.fits',
    "angle" : f'{prefix}-PA-pangle-at-peak-rm.fits',
    "fpol" : f'{prefix}-FPOL-at-center-freq.fits',
    "rm" : f'{prefix}-RM-depth-at-peak-rm.fits'
}

images = [postfix[_] for _ in "amp angle rm".split()]


cubes = glob("*-cubes*.fits")
stokes = {}
for cube in cubes:
    stoke = cube.split("-")[0].lower()
    stokes[stoke] = rfu.read_image_cube(cube)["data"]


stokes["l_pol"] = stokes["q"] + 1j*stokes["u"]
stokes["f_pol"] = (stokes["q"]/stokes["i"]) + ((1j*stokes["u"])/stokes["i"])


pica_i_image = "I-MFS.fits"
# pica_image = "I-hs-MFS.fits"
pica_mask = "true_mask_572.fits"

pica_i_data = rfu.get_masked_data(pica_i_image, pica_mask)

pangle_image = postfix["angle"]
pangle_data = rfu.read_image_cube(pangle_image)["data"]

fpol_image = postfix["fpol"]


# plot lobes and their dispersion
plot_rm_for_lobes(
    rot_meas_image=postfix["rm"],
    e_mask="masks/east-lobe-572.fits", w_mask="masks/west-lobe-572.fits",
    vmin=-100, vmax=100)


# plot fpol
# plot_fractional_polzn(stokes["f_pol"])
# plot_polarised_intensity(stokes["l_pol"])

plot_intensity_vectors(pica_i_image, fpol_image, pangle_image, pica_mask)

