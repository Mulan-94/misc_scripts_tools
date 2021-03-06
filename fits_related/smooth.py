#! /bin/python3
import argparse
import os
import re
import logging
import numpy as np

from astropy.io import fits
from glob import glob
from casacore.tables import table
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import dask.array as da

from ipdb import set_trace

logging.basicConfig()
snitch = logging.getLogger("sade")
snitch.setLevel(logging.INFO)


def get_arguments():
    parser = argparse.ArgumentParser(description="Refine model images")
    reqs = parser.add_argument_group("Required arguments")
    reqs.add_argument("--ms", dest="ms_name", required=True, metavar="",
        help="Input MS. Used for getting reference frequency")
    reqs.add_argument("-ip", "--input-prefix", dest="input_prefix",
        required=True, metavar="",
        help="The input image prefix. The same as the one used for wsclean")
    reqs.add_argument("-co", "--channels-out", dest="channels_out", metavar="",
        required=True, type=int,
        help="Number of channels to generate out")
    reqs.add_argument("-od", "--output-dir", dest="output_dir", default=None,
        metavar="",
        help="Where to put the output files.")
    reqs.add_argument("-order", "--polynomial-order", dest="poly_order",
        default=None, metavar="", type=int,
        help="Order of the spectral polynomial")
    reqs.add_argument("-nthreads", dest="nthreads", default=10, metavar="",
        type=int, help="Number of threads to use in this")
    reqs.add_argument("-stokes", dest="stokes",
        default="I", metavar="", type=str,
        help="""Which stokes model to extrapolate. Write as single streing e.g
        IQUV. Default 'I'""")
    return parser


def get_ms_ref_freq(ms_name):
    snitch.info("Getting reference frequency from MS")
    with table(f"{ms_name}::SPECTRAL_WINDOW", ack=False) as spw_subtab:
        ref_freq, = spw_subtab.getcol("REF_FREQUENCY")
    return ref_freq


def read_input_image_header(im_name):
    """
    Parameters
    ----------
    im_name: :obj:`string`
        Image name

    Output
    ------
    info: dict
        Dictionary containing center frequency, frequency delta and image wsum
    """
    snitch.debug(f"Reading image: {im_name} header")
    info = {}

    info["name"] = im_name
   
    with fits.open(im_name, readonly=True) as hdu_list:
        # print(f"There are:{len(hdu_list)} HDUs in this image")
        for hdu in hdu_list:
            naxis = hdu.header["NAXIS"]
            # get the center frequency
            for i in range(1, naxis+1):
                if hdu.header[f"CUNIT{i}"].lower() == "hz":
                    info["freq"] = hdu.header[f"CRVAL{i}"]
                    info["freq_delta"] = hdu.header[f"CDELT{i}"]

            #get the wsum
            info["wsum"] = hdu.header["WSCVWSUM"]
            info["data"] = hdu.data

    return info


def get_band_start_and_band_width(freq_delta, first_freq, last_freq):
    """
    Parameters
    ----------
    freq_delta: float
        The value contained in cdelt. Difference between the different
        consecutive bands
    first_freq: float
        Center frequency for the very first image in the band. ie. band 0 image
    last_freq: float
        Center frequency for the last image in the band. ie. band -1 image

    Output
    ------
    band_start: float
        Where the band starts
    band_width: float
        Size of the band
    """
    snitch.info("Calculating the band starting frequency and  band width")
    band_delta = freq_delta/2
    band_start = first_freq - band_delta
    band_stop = last_freq + band_delta
    band_width = band_stop - band_start
    return band_start, band_width


def gen_out_freqs(band_start, band_width, n_bands, return_cdelt=False):
    """
    Parameters
    ----------
    band_start: int or float
        Where the band starts from
    band_width: int or float
        Size of the band
    n_bands: int
        Number of output bands you want
    return_cdelt: bool
        Whether or not to return cdelt

    Output
    ------
    center_freqs: list or array
        iterable containing output center frequencies
    cdelt: int or float
        Frequency delta
    """
    snitch.info("Generating output center frequencies")
    cdelt = band_width/n_bands
    first_center_freq = band_start + (cdelt/2)

    center_freqs = [first_center_freq]
    for i in range(n_bands-1):
        center_freqs.append(center_freqs[-1] + cdelt)

    center_freqs = np.array(center_freqs)
    if return_cdelt:
        return center_freqs, cdelt
    else:
        return center_freqs


def concat_models(models):
    snitch.info(f"Concatenating {len(models)} model images")
    return np.concatenate(models, axis=1).squeeze()


def interp_cube(model, wsums, infreqs, outfreqs, ref_freq, spectral_poly_order):
    """
    model: array
        model array containing model image's data for each stokes parameter
    wsums: list or array
        concatenated wsums for each of stokes parameters
    infreqs: list or array
        a list of input frequencies for the images?
    outfreqs: int
        Number of output frequencies i.e how many frequencies you want out
    ref_freq: float
        The reference frequency. A frequency representative of this 
        spectral window, usually the sky frequency corresponding to the DC edge
        of the baseband. Used by the calibration system if a fixed scaling
        frequency is required or **in algorithms to identify the observing band**. 
        see https://casa.nrao.edu/casadocs/casa-5.1.1/reference-material/measurement-set
    spectral_poly_order: int
        the order of the spectral polynomial
    """

    snitch.info("Starting frequency interpolation")

    nchan = outfreqs.size
    nband, nx, ny = model.shape
    mask = np.any(model, axis=0)

    # components excluding zeros
    beta = np.zeros_like(model)
    beta[:, mask] = model[:, mask]
    beta = beta.reshape(beta.shape[0], beta.size//beta.shape[0])

    # convert this to a dask array
    beta = da.from_array(beta, chunks=(beta.shape[0], 10_000_000//beta.shape[0]))
    # beta = model[:, mask]

    
    if spectral_poly_order > infreqs.size:
        raise ValueError("spectral-poly-order can't be larger than nband")

    # we are given frequencies at bin centers, convert to bin edges
    #delta_freq is the same as CDELt value in the image header
    delta_freq = infreqs[1] - infreqs[0] 

    wlow = (infreqs - delta_freq/2.0)/ref_freq
    whigh = (infreqs + delta_freq/2.0)/ref_freq
    wdiff = whigh - wlow

    # set design matrix for each component
    # look at Offringa and Smirnov 1706.06786
    Xfit = np.zeros([nband, spectral_poly_order])
    for i in range(1, spectral_poly_order+1):
        Xfit[:, i-1] = (whigh**i - wlow**i)/(i*wdiff)

    # dirty_comps = Xfit.T.dot(wsums*beta)
    dirty_comps = da.dot(Xfit.T, wsums*beta)

    # hess_comps = Xfit.T.dot(wsums*Xfit)
    hess_comps = da.dot(Xfit.T, wsums*Xfit)

    # comps = np.linalg.solve(hess_comps, dirty_comps)
    comps = da.linalg.solve(hess_comps, dirty_comps)

    w = outfreqs/ref_freq

    # Xeval = np.zeros((nchan, spectral_poly_order))
    # for c in range(spectral_poly_order):
    #     Xeval[:, c] = w**c
    
    Xeval = w[:, np.newaxis]**np.arange(spectral_poly_order)[np.newaxis, :]

    # betaout = Xeval.dot(comps)
    betaout = da.dot(Xeval, comps)

    betaout = betaout.reshape(betaout.shape[0], nx, ny)
    
    modelout = np.zeros((nchan, nx, ny))
    # modelout = betaout[:, mask]
    # modelout = modelout.reshape(nchan, nx, ny)
    modelout = betaout
    return modelout


def gen_fits_file_from_template(template_fits, center_freq, cdelt, new_data, out_fits):
    with fits.open(template_fits, mode="readonly") as temp_hdu_list:
        temp_hdu, = temp_hdu_list

        #update the center frequency
        for i in range(1, temp_hdu.header["NAXIS"]+1):
            if temp_hdu.header[f"CUNIT{i}"].lower() == "hz":
                temp_hdu.header[f"CRVAL{i}"] = center_freq
                temp_hdu.header[f"CDELT{i}"] = cdelt
      
        #update with the new data
        if temp_hdu.data.ndim == 4:
            temp_hdu.data[0,0] = new_data
        elif temp_hdu.data.ndim == 3:
            temp_hdu.data[0] = new_data
        elif temp_hdu.data.ndim == 2:
            temp_hdu.data = new_data
        temp_hdu_list.writeto(out_fits, overwrite=True)
    snitch.info(f"New file written to: {out_fits}")
    return


def write_model_out(chan_num, temp_fname, output_dir, cdelt, models, freqs):
    outname = os.path.basename(temp_fname)
    outname = re.sub(r"-(\d){4}-", "-"+f"{chan_num}".zfill(4)+"-", outname)
    outname = os.path.join(output_dir, outname)
    gen_fits_file_from_template(
        temp_fname, freqs[chan_num], cdelt,
        models[chan_num], outname)

def main():
    args = get_arguments().parse_args()

    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.input_prefix), "smooth_out") 
    else:
        output_dir = args.output_dir
    
    if not os.path.isdir(output_dir):
        snitch.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    ref_freq = get_ms_ref_freq(args.ms_name)  

    snitch.info(f"Specified -stokes: {args.stokes.upper()}")
    for stokes in args.stokes.upper():
        snitch.info(f"Running Stoke's {stokes}")
        
        input_pref = os.path.abspath(args.input_prefix)
        images_list = sorted(glob(
                os.path.join(input_pref, f"*00*{stokes}-model*.fits")))
        if len(images_list) == 0:
            images_list = sorted(glob(
                os.path.join(input_pref, f"*00*-model*.fits")))
        
        if len(images_list) == 0:
            continue

        im_heads = []

        for im_name in images_list:
            im_header = read_input_image_header(im_name)
            im_heads.append(im_header)
        
        bstart, bwidth = get_band_start_and_band_width(
            im_heads[0]["freq_delta"], im_heads[0]["freq"], im_heads[-1]["freq"])

        model = concat_models([image_item["data"] for image_item in im_heads])
        out_freqs = gen_out_freqs(bstart, bwidth, args.channels_out)
        new_cdelt = out_freqs[1] - out_freqs[0]

        # gather the wsums and center frequencies
        w_sums = np.array([item["wsum"] for item in im_heads])
        w_sums = w_sums[:, np.newaxis]
        input_center_freqs = np.array([item["freq"] for item in im_heads])

        out_model = interp_cube(model, w_sums, input_center_freqs, out_freqs,
                                ref_freq, args.poly_order)

        # for chan in range(args.channels_out):
        #     outname = os.path.basename(images_list[0])
        #     outname = re.sub(r"(\d){4}", f"{chan}".zfill(4), outname)
        #     outname = os.path.join(output_dir, outname)
        #     gen_fits_file_from_template(
        #         images_list[0], out_freqs[chan], new_cdelt, out_model[chan],
        #         outname)
        
        results = []
        with ThreadPoolExecutor(args.nthreads) as executor:
            results = executor.map(
                partial(write_model_out, temp_fname=images_list[0],
                        output_dir=output_dir, cdelt=new_cdelt,
                        models=out_model, freqs=out_freqs), 
                range(args.channels_out))

        results = list(results)
        
        snitch.info("Done")


if __name__ == "__main__":
    main()
