import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from itertools import product
from difflib import get_close_matches
from copy import deepcopy


def get_pixel(fname):
    pix  = fits.getdata(fname).squeeze()[:, 2061, 2190]
    return pix

def process_pixel(data, x_pix, y_pix):
    pix_data = data[:, x_pix, y_pix]
    polynomial_fit()



def polynomial_fit(x, y_raw, degree):
    """
    Perform a polynomial fit on the input data
    Not that neither should contain Nan values. Zeroes are also discouraged
    Unless they form part of the raw data.

    parameters
    ----------
    x: np.array
        the independent variable (i.e. frequency for this specific script)
    y_raw: np.array
        the raw data being fitted
    degree: int
        Degree of poynomial fit

    Returns
    -------
    y_fit: np.array
        The fitted model to y
    coeff: tuple
        Coefficients to the fitted polynomial
    """
    if hasattr(y_raw, "mask"):
        coeffs = np.ma.polyfit(x, y_raw, degree)
    else:
        coeffs = np.polyfit(x, y_raw, degree)
    polynom = np.poly1d(coeffs)
    y_fit = polynom(x)
    return y_fit, tuple(coeffs.data)


def get_clean_header(fname=None, hdr=None):
    # remove extra axes
    # remove extra beam information
    # remove history
    # remove extra ctypes
    # removE WSC stuff
    # 
    if fname is not None:
        hdr = fits.getheader(fname)

    copy_hdr = deepcopy(hdr)
    for pref in "wsc bma bmi bpa history".upper().split():
        for key in hdr.keys():
            if (pref in key) and (key in copy_hdr):
                # print(f"Deleting this key: {key}")
                del copy_hdr[key]
    
    return copy_hdr



def plot_mod_vs_raw(raw, model, freq, fname="mod-vs-raw.png"):
    """
    raw: np.array
        The raw data for a single LOS
    model: np.array
        Modelled data fro that LOS
    freq: np.array
        The X-axis data (frequency)
    """
    print("Plotting Fitted model and raw data for a single LOS")

    plt.close("all")
    fig, ax= plt.subplots(figsize=(16,9))
    ax.plot(freq, raw, "ko", "Raw data")
    ax.plot(freq, model, "r", label='Model')
    ax.minorticks_on()
    fig.legend()
    fig.savefig(fname, dpi=600)
    print(f"Saving figure to: {fname}")
    return

def write_fitsfile(fname, data, hdr):
    fits.writeto(fname, data, header=hdr, overwrite=True)


def split_cube(fname, model_data, cube_hdr, exclude=None):
    """
    Generate single channelised images rather than a stacked cube
    """
    clean_hdr = get_clean_header(hdr=cube_hdr)
    
    for chan in range(model_data.shape[0]):
        if (exclude is not None) and (chan in exclude):
            continue

        clean_hdr.update({
            f"BMAJ": cube_hdr.get(f"BMAJ{chan+1}"),
            f"BMIN": cube_hdr.get(f"BMIN{chan+1}"),
            f"BPA": cube_hdr.get(f"BPA{chan+1}")
        })

        # name the images per channel like wsclean
        oname = fname+("-" + f"{chan}".zfill(4) + ".fits")

        print(f"Writing:    {oname}")

        write_fitsfile(oname, model_data[chan], clean_hdr)



def argument_parser():
    parser = argparse.ArgumentParser(
        description="Fit a simple taylor polynomial on Stokes I cube, generating a model I")
    parser.add_argument("cube", type=str,
        help="Name of the input I cube to be modelled")
    parser.add_argument("freqs", type=str,
        help="Text file containing respective frequencies. The freqs should be in Hz")

    parser.add_argument("-mask", "--mask-name", dest="mask", type=str, default=None,
        help="Region within which to perform fits. This is a FITS MASK")

    parser.add_argument("-ex", "--exclude-channels", dest="exclude", type=str, default=None, 
        help="""Channels that should be excluded while fitting the image cube. 
        This should be a text file containing those channel numbers. Data 
        corresponding to the specified channels will be set to zero.""")
    
    parser.add_argument("-u", "--unstack", dest="unstack", action="store_true",
        help="""Unstack the output model cube as single channelised images. If -ex was specified
        those channels will also be excluded.""")
    
    parser.add_argument("-deg", "--degree", dest="deg", type=int, default=2,
        help="Degree of the taylor polynimial to be fitted.")

    parser.add_argument("-o", "--output-dir", dest="odir", type=str, default="fitness",
        help="Directory where to dump the output files.")

    return parser


def main():
    opts = argument_parser().parse_args()

    fname = opts.cube
    mask = opts.mask
    freq_fname = opts.freqs
    deg = opts.deg
    odir = opts.odir


    if not os.path.isdir(odir):
        print(f"Creating directory: {odir}")
        os.makedirs(odir)


    # Read the freq file in GHZ
    freqs = np.loadtxt(freq_fname) /1e9
    # REad the image data
    image_data = fits.getdata(fname).squeeze()

    if opts.exclude is not None:
        exclude = np.loadtxt(opts.exclude).astype(int)
        image_data[exclude] = np.nan
        image_data = np.ma.masked_invalid(image_data)
        np.ma.set_fill_value(image_data, np.nan)

        freqs[exclude] = np.nan
        freqs = np.ma.masked_invalid(freqs)
        np.ma.set_fill_value(freqs, np.nan)


    # initialise 3-d arrays where models will be stored
    # these will be shared arrays
    model_data = np.zeros_like(image_data)
    # 3-d arrays where coefficients will be stored
    # e.g deg 3 will have three coefficients + a constant
    coeffs_data = np.zeros((deg+1, *image_data.shape[1:]))

    if mask is not None:
        mask_data = fits.getdata(mask).squeeze()
        ys, xs = np.where(mask_data > 0 )
        pixels = zip(ys, xs)
    else:
        # since input image is cube, pick shape containing the x y axis only
        ys, xs = image_data.shape[1:]
        ys, xs = range(ys), range(xs)
        pixels = product(ys, xs)

    
    for ypix, xpix in pixels:
        print(f"pixel x: {xpix}, y: {ypix}")
        model, coeffs = polynomial_fit(x=freqs, y_raw=image_data[:, ypix, xpix], degree=deg)
        model_data[:, ypix, xpix] = model

        # check actuall y if coefficients will work ok here
        coeffs_data[:, ypix, xpix] = coeffs

    if hasattr(model_data, "mask"):
        model_data = model_data.filled()
        
    hdr = fits.getheader(fname)
    
    if opts.unstack:
        split_cube(os.path.join(odir, f"i-model-deg-{deg}"), model_data, hdr, exclude=exclude)
    
    write_fitsfile(os.path.join(odir, f"i-model-deg-{deg}.fits"), model_data, hdr)

    hdr = get_clean_header(fname=fname)
    for _ in range(len(coeffs)):
        write_fitsfile(os.path.join(odir, f"coefficient-{_}.fits"), coeffs_data[_], hdr)

    print("done")

def console():
    """A console run entry point for setup.cfg"""
    main()
    snitch.info("Bye :D !")

if __name__ == "__main__":
    console()
    
    """
    python new-fit.py inputs/i-image-cube.fits frequencies.txt \
        -mask ../../../surround.fits -deg 3 -o modeli-t3


    python new-fit.py inputs/i-full-cube.fits all-freqs.txt \
        -mask ../../../surround.fits -deg 3 -o fitting-t3-full-c \
        -ex not-sel.txt -u
    """