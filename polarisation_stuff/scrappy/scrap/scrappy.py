import os
import numpy as np
import subprocess
import sys

from concurrent import futures
from functools import partial
from itertools import product
from multiprocessing import Array

PATH = set(sys.path)
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir))
if not PATH.issuperset(PROJECT_ROOT):
    sys.path.append(PROJECT_ROOT)


from utils.genutils import fullpath, make_out_dir
from utils.mathutils import is_infinite, are_all_nan, nancount, are_all_zeroes
from utils.rmmath import (polarised_snr, linear_polzn_error, frac_polzn,
    frac_polzn_error, linear_polzn, polzn_angle, polzn_angle_error,lambda_sq)


from scrap.scraplog import snitch
from scrap.arguments import parser
from scrap.image_utils import (read_fits_image, region_flux_jybm, region_is_above_thresh,
    read_regions_as_pixels, make_default_regions, make_noise_region_file,
    parse_valid_region_candidates, write_regions, image_noise, get_wcs)
from scrap.plotting import overlay_regions_on_source_plot


def initialise_globals(odir="scrappy-out"):
    global ODIR, RDIR, PLOT_DIR, LOS_DIR, RFILE, CURRENT_RFILE, NRFILE
    global OVERWRITE, MFT, REG_SIZE, NWORKERS

    ODIR = fullpath(os.curdir, odir)

    RDIR = fullpath(ODIR, "regions")
    PLOT_DIR =  fullpath(ODIR, "plots")
    LOS_DIR = fullpath(ODIR, "los-data")

    RFILE = fullpath(RDIR, "regions")
    CURRENT_RFILE = RFILE
    NRFILE = fullpath(RDIR, "noise-region")

    OVERWRITE = True
    MFT = 0.7
    REG_SIZE = 30
    NWORKERS = 16
    return


def make_image_lists(image_dir):
    images = dict()
    for stokes in "IQU":
        imlist = subprocess.check_output(
            f"ls -v {image_dir}/*-[0-9][0-9][0-9][0-9]-{stokes}-image.fits",
            shell=True)
        images[stokes] = imlist.decode().split("\n")[:-1]

    imlist = zip(images["I"], images["Q"], images["U"])
    return list(imlist)

def make_syncd_data_stores(size, syncd=True):
    """
    size: int
        Number of regions available
    """

    # i.e I, Q, U
    dtype = "d"

    per_image = dict()
    for stoke in "IQU":
        per_image[f"{stoke}"] = dtype
        per_image[f"{stoke}_err"] = dtype


    general = {
        "chan_width": dtype,
        "fpol": dtype,
        "fpol_err": dtype,
        "lpol_err": dtype,
        "pangle": dtype,
        "pangle_err": dtype,
        "snr": dtype,
        "freqs": dtype,
        "lambda_sq": dtype
        # storing a boolean as unsigned char
        # https://docs.python.org/3/library/array.html#module-array
        }

    if syncd:

        outs = {key: Array(dt, np.ones(size, dtype=dt)*np.nan)
                for key, dt in per_image.items()}
        outs.update({key: Array(dt, np.ones(size, dtype=dt)*np.nan)
                for key, dt in general.items()})
        outs["mask"] = Array('B', np.zeros(size, dtype=bool))
    else:
        outs = {key: np.ones(size, dtype=dt)*np.nan
                for key, dt in per_image.items()}
        outs.update({key: np.ones(size, dtype=dt)*np.nan
                for key, dt in general.items()})
        outs["mask"] = np.zeros(size, dtype=bool)
    return outs


# def parse_valid_fpol_region_per_chan(i_data, q_data, u_data, regs, noise_reg, threshold):
#     """
#     Only store regions which are above the specified threshold
#     Algorithm
#     1. Calculate poln power
#     2. Calulate error in 1
#     3. Check if snr > threhsold
#     4. If snr > threshold
#         - calculate fpol
#         - Only store if fpol is btwn 0 and 1
#     5. Store the
#         - flux_jynm
#         - image noise
#         - freq
#         - lambda squared
#         - i, q and u errors
#         - fpol errors
#         - 
#     """
#     noise_reg = read_regions_as_pixels(noise_file)
    
#     i_noise = image_noise(noise_reg, i_data)
#     q_noise = image_noise(noise_reg, q_data)
#     u_noise = image_noise(noise_reg, u_data)


#     valids = []
#     for _i, reg in enumerate(regs):
#         signal_i = region_flux_jybm(reg, i_data)
#         signal_q = region_flux_jybm(reg, q_data)
#         signal_u = region_flux_jybm(reg, u_data)

#         snr = polarised_snr(signal_q, signal_u, q_noise, u_noise)
#         noise = linear_polzn_error(signal_q, signal_u, q_noise, u_noise)

#         if is_infinite(noise):
#             snitch.info(f"Skipping region: {reg.meta['text']}. Image is not sensible")
#             continue

#         if region_is_above_thresh(snr=snr, noise=noise, threshold=threshold):
#             # fpol = frac_polzn(signal_i, signal_q, signal_u)
            
#             # if fpol>1 or fpol<=0:
#             #     snitch.info(f"Skipping region: {reg.meta['text']}. "+\
#             #             "Fractional polarisation is above limit")
#             #     continue

#             # # store signal and noise info
#             info.append(dict(flux_jybm=signal, noise=noise))
#             valids.append(
#                 "circle({:.6f}, {:.6f}, {:.6f}\") # text={{reg_{}}}".format(
#                 reg.center.ra.deg, reg.center.dec.deg, reg.radius.arcsec, _i))

#     if len(valids) > 0:
#         snitch.info(f"Found {len(valids)}/{len(regs)} valid region candidates")
#         valid_candidates_file = RFILE + "-valid-candidates.reg"
#         valid_candidates_file = write_regions(valid_candidates_file, valids)
#     else:
#         snitch.warning("No valid regions were found")

#     return valid_candidates_file
    

def parse_valid_fpol_region_per_region(triplets, cidx, reg, noise_reg, threshold):
    """
    triplets: tuple
        A tuple with (i,q,u) image name
    cidx: int
        Index number of the channel          
    """    
    global sds
    # function starts here; loop over the channelised images
    i_im, q_im, u_im = triplets
    i_data = read_fits_image(i_im)
    q_data = read_fits_image(q_im)
    u_data = read_fits_image(u_im)

    i_noise = image_noise(noise_reg, i_data.data)
    q_noise = image_noise(noise_reg, q_data.data)
    u_noise = image_noise(noise_reg, u_data.data)

    # check 1: Is the image valid ?
    # remember initial values are set in sds intializer
    channel = i_im.split('-')[-3]
    if is_infinite(i_noise) or are_all_nan(i_noise) or are_all_zeroes(i_noise):
        snitch.info(f"Skipping channel: {channel} images. They are not sensible.")
        # mask this data 
        sds["mask"][cidx] = True
        return False


    signal_i = region_flux_jybm(reg, i_data.data)
    signal_q = region_flux_jybm(reg, q_data.data)
    signal_u = region_flux_jybm(reg, u_data.data)

    snr = polarised_snr(signal_q, signal_u, q_noise, u_noise)
    noise = linear_polzn_error(signal_q, signal_u, q_noise, u_noise)

 
    # Check 2: Is the region above the threshold ?
    if region_is_above_thresh(snr=snr, noise=noise, threshold=threshold):
        # store signal and noise info
        sds["I"][cidx] = signal_i 
        sds["I_err"][cidx] = i_noise 
        
        sds["Q"][cidx] = signal_q
        sds["Q_err"][cidx] = q_noise 

        sds["U"][cidx] = signal_u
        sds["U_err"][cidx] = u_noise 
 
        
        fpol = frac_polzn(signal_i, signal_q, signal_u)
        sds["fpol"][cidx] = fpol
        sds["fpol_err"][cidx] = frac_polzn_error(signal_i, signal_q, signal_u,
                                        i_noise, q_noise, u_noise)

        sds["lpol_err"][cidx] = noise
        sds["pangle"][cidx] = polzn_angle(signal_q, signal_u)
        sds["pangle_err"][cidx] = polzn_angle_error(signal_q, signal_u, q_noise, u_noise)
        sds["snr"][cidx] = snr
        sds["chan_width"][cidx] = i_data.header["CDELT3"]
        sds["freqs"][cidx] = i_data.freq
        sds["lambda_sq"][cidx] = lambda_sq(i_data.freq, i_data.header["CDELT3"])

        # check 3: is fpol above zero?
        # create a mask for when fpol less than 0 or less than1
        sds["mask"][cidx] = True if fpol<0 or fpol>1 else False
        return True

    else:
        snitch.info(
            f"Skipping channel: {channel} in LoS: {reg.meta['text']}. " +
            f"SNR {snr:.2f} < {threshold}")
        # mask this data 
        sds["mask"][cidx] = True
        return False   


def make_per_region_data_output(images, reg_file, noise_file, threshold, wcs_ref):
    """
    images: list
        List containing tuples with the channelise I,Q,U images. 
        ie. [(chan1_I, chan2_Q, chan1_U), ...]
    reg_file: str
        File name of the file containing the regions to be evaluated
    noise_file: str
        File containning the noise region
    """

    #using one of the input images for wcs 
    wcs = get_wcs(wcs_ref)

    regs = read_regions_as_pixels(reg_file, wcs=wcs)
    noise_reg, = read_regions_as_pixels(noise_file, wcs=wcs)


    valid_regions = []
    count = 1
    for ridx, reg in enumerate(regs):

        # create some data store that'll store region data
        global sds

        sds = make_syncd_data_stores(len(images), syncd=True)       
        snitch.info(f"Region: {reg.meta['text']}")

        with futures.ProcessPoolExecutor(max_workers=NWORKERS) as executor:
            results = list(executor.map(
                    partial(
                        parse_valid_fpol_region_per_region,
                           reg=reg, noise_reg=noise_reg,
                            threshold=threshold),
                    images, range(len(images))
                  ))


        ##################################################################
        # Debug them
        ##################################################################

        # results = []
        # sds = make_syncd_data_stores(len(images), syncd=False) 
        # for chan, triplet in enumerate(images):
        #     results.append(parse_valid_fpol_region_per_region(triplet,
        #             cidx=chan, reg=reg,
        #             noise_reg=noise_reg, threshold=threshold))

        ##################################################################
        
        # only accept if
        # 1. not all data is masked/flagged and
        # 2.flagged data <= MFT%
        n_masked = np.array(sds["mask"]).sum()
        n_chans = len(images)
        if n_masked != n_chans and n_masked <= n_chans*MFT:
            # adding lpol here because complex arrays dont work with multiproc array
            sds["lpol"] = linear_polzn(np.array(sds["Q"]), np.array(sds["U"]))
            
            snitch.warning(f"{reg.meta['text']}: flagged {n_masked}/{n_chans} points")
        
            outfile = fullpath(LOS_DIR, f"reg_{count}")
            np.savez(outfile, **sds)
            count += 1

            sky = reg.to_sky(wcs)
            valid_regions.append(
                "circle({:.6f}, {:.6f}, {:.6f}\")".format(
                    sky.center.ra.deg, sky.center.dec.deg, sky.radius.arcsec))
        else:
            snitch.warning(f"Skipping region {reg.meta['text']} because either:")
            snitch.warning("(1) Too much data was flagged, or not " +
                f"enough data: >{MFT*100}%, flagged: {(n_masked/n_chans)*100:.2f}%")
            snitch.warning(f"(2) All data is flagged; there's no valid "+
                            "data in this region.")

    # write the valid regions into a file
    valid_regs_file = write_regions(RFILE+"-valid.reg", valid_regions, reg_id=True)
    
    snitch.info(f"Done saving data files at: {LOS_DIR}")
    snitch.info( "--------------------")

    # returns the weeds
    return valid_regs_file


def step1_default_regions(reg_size, wcs_ref, x_range, y_range, threshold=1, rnoise=None):
    # Step 1. Make the default regions, ie within the source dimensions
    # we establish the sources bounds here, no parsing involved
    global RDIR, CURRENT_RFILE, NRFILE

    RDIR = make_out_dir(RDIR)

    CURRENT_RFILE = make_default_regions(*x_range, *y_range, reg_size,
                        wcs_ref, RFILE, overwrite=OVERWRITE)

    NRFILE = make_noise_region_file(name=NRFILE, reg_xy=None)

    overlay_regions_on_source_plot(
        CURRENT_RFILE, wcs_ref,
        rnoise or NRFILE, threshold)
    return


def step2_valid_reg_canidates(wcs_ref, threshold, rnoise=None):
    # Step 2: Determines which regions meet the requried threshold
    # we use the I-MFS image here. Basiclally just map the source extent
    global CURRENT_RFILE, NRFILE

    CURRENT_RFILE = parse_valid_region_candidates(wcs_ref, CURRENT_RFILE,
            NRFILE, threshold, noise=rnoise, overwrite=OVERWRITE)
    
    overlay_regions_on_source_plot(
        CURRENT_RFILE, wcs_ref,
        rnoise or NRFILE, threshold)

    return


def step3_valid_los_regs(image_dir, threshold, wcs_ref, rnoise=None):
    global CURRENT_RFILE, LOS_DIR, NRFILE
    
    LOS_DIR = make_out_dir(LOS_DIR)

    # Step 3: we also need the regional los data
    # Scrappy here

    images = make_image_lists(image_dir)

    # Step 3: Generate the regional data files
    CURRENT_RFILE = make_per_region_data_output(images, CURRENT_RFILE, NRFILE,
                        threshold=threshold, wcs_ref=wcs_ref)

    overlay_regions_on_source_plot(
        CURRENT_RFILE, wcs_ref,
        rnoise or NRFILE, threshold)

    return

def step4_plots():
    PLOT_DIR = make_out_dir(PLOT_DIR)



def main():
    opts = parser().parse_args()

    # doing it this way to modify odir
    if opts.reg_size is None:
        reg_size = REG_SIZE
    else:
        reg_size = opts.reg_size

    if opts.odir is not None:
        ODIR = opts.odir
    
    # I want the file names to contain region sizes
    ODIR += f"-s{reg_size}"
    ODIR = make_out_dir(ODIR)
    initialise_globals(ODIR)
    OVERWRITE = opts.noverwrite

    if opts.todo:
        todo = list(opts.todo)
    else:
        todo = list("rl")


    ##########################################
    #  For regions
    ##########################################

    if opts.rfile is not None:
        RFILE = opts.rfile
        snitch.info(f"Region file: {RFILE}")

    if opts.reg_size is None:
        reg_size = REG_SIZE
    else:
        reg_size = opts.reg_size

    if opts.wcs_ref is None:
        wcs_ref = "i-mfs.fits"
    else:
        wcs_ref = opts.wcs_ref

    #incase user wants specific noise for region generation
    if opts.rnoise is not None:
        rnoise = opts.rnoise
    else:
        rnoise = None

    if opts.x_range is None:
        # left to right: ra in degrees
        pictor_x = (80.04166306500294, 79.84454319889994)
        x_range = pictor_x
    else:
        x_range = opts.x_range

    if opts.y_range is None:
        #  bottom to top dec in degrees
        pictor_y = (-45.81799666164118, -45.73325018138195)
        y_range = pictor_y
    else:
        y_range = opts.y_range

    if opts.threshold is None:
        threshold = 3
    else:
        threshold = opts.threshold

    if opts.nworkers is not None:
        NWORKERS = opts.nworkers

    # For regions
    if opts.regions_only or "r" in todo:

        step1_default_regions(reg_size, wcs_ref, x_range, y_range,
            threshold=threshold, rnoise=rnoise)
        step2_valid_reg_canidates(wcs_ref, threshold, rnoise=rnoise)
    
    # For scrappy
    if opts.los_only or "l" in todo:
 

        if opts.image_dir is not None:
            image_dir = opts.image_dir
            step3_valid_los_regs(image_dir, threshold, wcs_ref, rnoise=rnoise)
        else:
            snitch.error("No directory for the input images was provided")
            snitch.error("Please specifiy using --image dir")
            snitch.error("Byeee!")
            sys.exit()


    # For plots
    if opts.plots_only or "p" in todo:
        # step4_plots()
        pass

    return


if __name__ == "__main__":
    main()
    snitch.info("Bye :D !")

    """
    python scrappy.py -id imgs -od testing -ref-image
        imgs/i-mfs.fits --threshold 3

    # test regions
    python scrappy.py -od testing-regs -ref-image
        imgs/i-mfs.fits --threshold 3 -rs 40 -todo r -idir imgs

    # test LOS
    python scrappy.py -od testing-LOS -ref-image
        imgs/i-mfs.fits --threshold 3 -rs 40 -todo rl -idir imgs


    python qu_pol/scrappy/scrappy.py -rs 3 -idir 
        --threshold 10 -odir $prods/scrap-outputs 
        -ref-image i-mfs.fits -mrn 0.0006 -todo rl
    """