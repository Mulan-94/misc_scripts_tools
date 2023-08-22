import os
import numpy as np
import subprocess
import shutil
import sys

from concurrent import futures
from functools import partial
from glob import glob
from itertools import product
from multiprocessing import Array
from natsort import natsorted

PATH = set(sys.path)
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir))
if not PATH.issuperset(PROJECT_ROOT):
    sys.path.append(PROJECT_ROOT)


from utils.genutils import fullpath, make_out_dir, does_specified_file_exist
from utils.mathutils import (is_infinite, are_all_nan, nancount, are_all_zeroes,
    signal_to_noise)
from utils.rmmath import (polarised_snr, linear_polzn_error, frac_polzn,
    frac_polzn_error, linear_polzn, polzn_angle, polzn_angle_error,lambda_sq)


from scrap.scraplog import snitch
from scrap.arguments import parser
from scrap.image_utils import (read_fits_image, read_fits_cube, region_flux_jybm,
    region_is_above_thresh,
    read_regions_as_pixels, make_default_regions, make_noise_region_file,
    parse_valid_region_candidates, write_regions, image_noise, get_wcs)
from scrap.plotting import overlay_regions_on_source_plot


def initialise_globals(odir="scrappy-out"):
    global ODIR, RDIR, PLOT_DIR, LOS_DIR, RFILE, CURRENT_RFILE, NRFILE
    global OVERWRITE, MFT, REG_SIZE, NWORKERS, USE_POLZD_SNR

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
    USE_POLZD_SNR = False
    return


def make_image_lists(image_dir):
    if  type(image_dir)== list:
        cubes = True
        snitch.info(f"Cube mode: The following cubes have been found:")
        imlist = image_dir
        for _ in imlist:
            snitch.info(f"-> {_}: {does_specified_file_exist(_)}")
        return imlist, cubes
    
    cubes = False
    images = dict()
    for stokes in "IQU":
        imlist = subprocess.check_output(
            f"ls -v {image_dir}/*-[0-9][0-9][0-9][0-9]-{stokes}-image.fits",
            shell=True)
        images[stokes] = imlist.decode().split("\n")[:-1]
    
    snitch.info("Single channel mode.")

    imlist = zip(images["I"], images["Q"], images["U"])
    return list(imlist), cubes

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
        "psnr": dtype,
        "freqs": dtype,
        "lambda_sq": dtype,
        "noise": dtype,
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



def parse_valid_fpol_region_per_region(triplets, cidx, reg, noise_reg,
    threshold, global_noise):
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

    # Add some of the free standing data for completeness
    # these should never have nan values
    sds["chan_width"][cidx] = i_data.header["CDELT3"]
    sds["freqs"][cidx] = i_data.freq
    sds["lambda_sq"][cidx] = lambda_sq(i_data.freq, i_data.header["CDELT3"])
    sds["noise"][cidx] = global_noise

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

    p_snr = polarised_snr(signal_q, signal_u, q_noise, u_noise)
    i_snr = signal_to_noise(signal_i, global_noise)

    sds["snr"][cidx] = i_snr
    sds["psnr"][cidx] = p_snr

    if USE_POLZD_SNR:
        snr = p_snr
    else:
        # get snr of total intensity to global total intensity noise
        snr = i_snr

    # Check 2: Is the region above the threshold ?
    if region_is_above_thresh(snr=snr, threshold=threshold):
        # store signal and noise info
        sds["I"][cidx] = signal_i 
        sds["I_err"][cidx] = i_noise 
         
        sds["Q"][cidx] = signal_q
        sds["Q_err"][cidx] = q_noise 

        sds["U"][cidx] = signal_u
        sds["U_err"][cidx] = u_noise 


        # N.B we DO NOT store or lpol because it's a complex number
        #  that cannot be held in the shared array!!
        
        # this function does fractional q and u inside       
        fpol = frac_polzn(signal_i, signal_q, signal_u)
        sds["fpol"][cidx] = fpol
        sds["fpol_err"][cidx] = frac_polzn_error(signal_i, signal_q, signal_u,
                                        i_noise, q_noise, u_noise)

        sds["lpol_err"][cidx] = linear_polzn_error(signal_q,
                                    signal_u, q_noise, u_noise)
        sds["pangle"][cidx] = polzn_angle(signal_q/signal_i, signal_u/signal_i)
        sds["pangle_err"][cidx] = polzn_angle_error(signal_q/signal_i,
                                    signal_u/signal_i, q_noise, u_noise)

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


def make_per_region_data_output(images, reg_file, noise_file, threshold, noise_ref):
    """
    images: list
        List containing tuples with the channelise I,Q,U images. 
        ie. [(chan1_I, chan2_Q, chan1_U), ...]
    reg_file: str
        File name of the file containing the regions to be evaluated
    noise_file: str
        File containning the noise region
    """

    #using the noise reference image to get the wcs and the global reference noise
    wcs = get_wcs(noise_ref)

    regs = read_regions_as_pixels(reg_file, wcs=wcs)
    noise_reg, = read_regions_as_pixels(noise_file, wcs=wcs)

    # get the single reference global noise from the noise reference image
    global_noise = image_noise(noise_reg, read_fits_image(noise_ref).data)

    valid_regions = []
    count = 1
    
    
    for ridx, reg in enumerate(regs):

        # create some data store that'll store region data
        global sds

        sds = make_syncd_data_stores(len(images), syncd=True)   

        # for orphaned regions  
        if "text" not in reg.meta:
            reg.meta["text"] = f"reg_{ridx+1}" 

        snitch.info(f"Region: {reg.meta.get('text', ridx)}")

        if not DEBUG:
            with futures.ProcessPoolExecutor(max_workers=NWORKERS) as executor:
                results = list(executor.map(
                        partial(
                            parse_valid_fpol_region_per_region,
                            reg=reg, noise_reg=noise_reg,
                                threshold=threshold, global_noise=global_noise),
                        images, range(len(images))
                    ))

        else:
            # ##################################################################
            # # Debug them
            # ##################################################################
            results = []
            sds = make_syncd_data_stores(len(images), syncd=False) 
            for chan, triplet in enumerate(images):
                results.append(parse_valid_fpol_region_per_region(triplet,
                        cidx=chan, reg=reg, global_noise=global_noise,
                        noise_reg=noise_reg, threshold=threshold))

        # ##################################################################
        
        # only accept if
        # 1. not all data is masked/flagged and
        # 2.flagged data <= MFT%
        n_masked = np.array(sds["mask"]).sum()
        n_chans = len(images)
        if n_masked != n_chans and n_masked <= n_chans*MFT:
            # adding lpol here because complex arrays dont work with multiproc array
            sds["lpol"] = linear_polzn(
                np.array(sds["Q"])/np.array(sds["I"]), 
                np.array(sds["U"])/np.array(sds["I"]))
            
            if n_masked > 0:
                snitch.warning(f"{reg.meta['text']}: flagged {n_masked}/{n_chans} points")
        
            outfile = fullpath(LOS_DIR, f"reg_{count}")
            
            if "tag" not in reg.meta:
                reg.meta["tag"] = ["g1"]
            
            sds["tag"] = ','.join(reg.meta["tag"])
            
            np.savez(outfile, **sds)
            count += 1

            sky = reg.to_sky(wcs)
            valid_regions.append(
                "circle({:.6f}, {:.6f}, {:.6f}\") # tag={{{}}}".format(
                    sky.center.ra.deg, sky.center.dec.deg, sky.radius.arcsec,
                    ','.join(reg.meta["tag"])))
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


def cubes_make_per_region_data_output(triplets, reg_file, noise_file, threshold,
    noise_ref, freqs=None):

    wcs = get_wcs(noise_ref)

    regs = read_regions_as_pixels(reg_file, wcs=wcs)
    noise_reg, = read_regions_as_pixels(noise_file, wcs=wcs)

    # get the single reference global noise from the noise reference image
    global_noise = image_noise(noise_reg, read_fits_image(noise_ref).data)


    i_im_cube, q_im_cube, u_im_cube = sorted(triplets)

    freqs = np.loadtxt(freqs)

    i_data = read_fits_cube(i_im_cube, freqs=freqs, stokes="I")
    q_data = read_fits_cube(q_im_cube, freqs=freqs, stokes="Q")
    u_data = read_fits_cube(u_im_cube, freqs=freqs, stokes="U")

    gens = make_syncd_data_stores(freqs.size, syncd=False) 
    gens["chan_width"] = i_data.chan_width
    gens["freqs"] = i_data.freq
    gens["lambda_sq"] = lambda_sq(i_data.freq, i_data.chan_width)
    gens["noise"].fill(global_noise)


    valid_regions = []

    triplets = (i_data, q_data, u_data)

    if not DEBUG:
        snitch.info("Starting pool")
        with futures.ThreadPoolExecutor(max_workers=NWORKERS) as executor:
            valid_regions = list(executor.map(
                    partial(
                        cubes_parse_valid_fpol_region_per_region,
                        triplets=triplets, wcs=wcs, 
                        noise_reg=noise_reg, threshold=threshold, datas=gens),
                    regs
            ))
    else:
        for reg in regs:
            reg_coord = cubes_parse_valid_fpol_region_per_region(reg=reg,
                triplets=triplets, wcs=wcs, noise_reg=noise_reg,
                threshold=threshold, datas=gens)
            valid_regions.append(reg_coord)
    
    valid_regions = [_ for _ in valid_regions if _]

    # rename files to proper numbers
    for i, los in enumerate(natsorted(glob(f"{LOS_DIR}/*.npz")), 1):
        os.rename(los, os.path.join(LOS_DIR, f"reg_{i}.npz"))


    # write the valid regions into a file
    valid_regs_file = write_regions(RFILE+"-valid.reg", valid_regions, reg_id=True)
    
    snitch.info(f"Done saving data files at: {LOS_DIR}")
    snitch.info( "--------------------")

    # returns the weeds
    return valid_regs_file


def cubes_parse_valid_fpol_region_per_region(reg, triplets, wcs, noise_reg,
    threshold, datas):
    """
    triplets: pass the cube data at this point
    freqs: an array containing the input freqquencies
    """
    snitch.info(f"Reg: {reg.meta['text']}")

    datas = dict(datas)

    i_data, q_data, u_data = triplets
    signal_i = region_flux_jybm(reg, i_data.data)
    signal_q = region_flux_jybm(reg, q_data.data)
    signal_u = region_flux_jybm(reg, u_data.data)

    i_noise = image_noise(noise_reg, i_data.data)
    q_noise = image_noise(noise_reg, q_data.data)
    u_noise = image_noise(noise_reg, u_data.data)

    global_noise = datas["noise"]

    p_snr = polarised_snr(signal_q, signal_u, q_noise, u_noise)
    i_snr = signal_to_noise(signal_i, global_noise)

    datas["snr"] = i_snr
    datas["psnr"] = p_snr
    
    datas["I"] = signal_i 
    datas["I_err"] = i_noise 
        
    datas["Q"] = signal_q
    datas["Q_err"] = q_noise 

    datas["U"] = signal_u
    datas["U_err"] = u_noise 

    datas["lpol"] = linear_polzn(signal_q/signal_i, signal_u/signal_i)
    datas["fpol"] = frac_polzn(signal_i, signal_q, signal_u)
    datas["fpol_err"] = frac_polzn_error(signal_i, signal_q, signal_u,
                                        i_noise, q_noise, u_noise)
    datas["lpol_err"] = linear_polzn_error(signal_q,
                                signal_u, q_noise, u_noise)
    datas["pangle"] = polzn_angle(signal_q/signal_i, signal_u/signal_i)
    datas["pangle_err"] = polzn_angle_error(signal_q/signal_i,
                                signal_u/signal_i, q_noise, u_noise)



     # check 1: Is the image valid ?
    # remember initial values are set in sds intializer
    if np.all(is_infinite(i_noise)) or are_all_nan(i_noise) or are_all_zeroes(i_noise):
        snitch.info(f"Skipping LoS: {reg.meta['text']}. Its data is not sensible.")
        # mask this data 
        return False

    # get snr of total intensity to global total intensity noise
    snr = p_snr if USE_POLZD_SNR else i_snr

    
    # Check 2: Is the region above the threshold ?
    datas["mask"][snr<threshold] = True
    mask = datas["mask"]

    if mask.sum() == mask.size:
        snitch.info(f"{reg.meta['text']}: All data was flagged. Skipping.")
        return False


    if mask.sum()>0:
        snitch.info(
            f"{reg.meta['text']}: flagged {mask.sum()}/{mask.size} of the "+
            "channels which are lower than the threshold.")

    if "tag" not in reg.meta:
        reg.meta["tag"] = ["g1"]
    
    datas["tag"] = ','.join(reg.meta["tag"])

    outfile = fullpath(LOS_DIR, f"{reg.meta['text']}-tmp")
    np.savez(outfile, **datas)


    sky = reg.to_sky(wcs)
    reg_coord = "circle({:.6f}, {:.6f}, {:.6f}\") # tag={{{}}}".format(
            sky.center.ra.deg, sky.center.dec.deg, sky.radius.arcsec,
            ','.join(reg.meta["tag"]))
    
    return reg_coord



def step1_default_regions(reg_size, wcs_ref, bounds, threshold=1, rnoise=None):
    # Step 1. Make the default regions, ie within the source dimensions
    # we establish the sources bounds here, no parsing involved
    global RDIR, CURRENT_RFILE, NRFILE

    RDIR = make_out_dir(RDIR)

    CURRENT_RFILE = make_default_regions(bounds, reg_size,
                        wcs_ref, RFILE, overwrite=OVERWRITE)

    NRFILE = make_noise_region_file(name=NRFILE, reg_xy=None)

    overlay_regions_on_source_plot(
        CURRENT_RFILE, wcs_ref,
        rnoise or NRFILE, threshold)
    return


def step2_valid_reg_candidates(wcs_ref, noise_ref, threshold, rnoise=None):
    # Step 2: Determines which regions meet the requried threshold
    # we use the I-MFS image here. Basiclally just map the source extent
    global CURRENT_RFILE, NRFILE

    CURRENT_RFILE = parse_valid_region_candidates(noise_ref, CURRENT_RFILE,
            NRFILE, threshold, noise=rnoise, overwrite=OVERWRITE)
    
    overlay_regions_on_source_plot(
        CURRENT_RFILE, wcs_ref,
        rnoise or NRFILE, threshold)

    return


def step3_valid_los_regs(image_dir, threshold, noise_ref, rnoise=None, freq_file=None):
    global CURRENT_RFILE, LOS_DIR, NRFILE
    
    LOS_DIR = make_out_dir(LOS_DIR, delete_if_exists=True)

    # Step 3: we also need the regional los data
    # Scrappy here

    images, cubes = make_image_lists(image_dir)

    # # Step 3: Generate the regional data files
    if not cubes:
        CURRENT_RFILE = make_per_region_data_output(images, CURRENT_RFILE, NRFILE,
                        threshold=threshold, noise_ref=noise_ref)
    else:
        CURRENT_RFILE = cubes_make_per_region_data_output(images, CURRENT_RFILE, NRFILE,
                        threshold=threshold, noise_ref=noise_ref, freqs=freq_file)

    overlay_regions_on_source_plot(CURRENT_RFILE, noise_ref, rnoise or NRFILE,
        threshold)

    return

def step4_plots():
    PLOT_DIR = make_out_dir(PLOT_DIR, delete_if_exists=True)



def main():
    global RDIR, RFILE, CURRENT_RFILE, NRFILE, DEBUG
    opts = parser().parse_args()

    # doing it this way to modify odir
    if opts.reg_size is None:
        reg_size = REG_SIZE
    else:
        reg_size = opts.reg_size

    if opts.odir is not None:
        ODIR = opts.odir
    
    # I want the file names to contain region sizes
    ODIR = make_out_dir(ODIR)
    initialise_globals(ODIR)
    OVERWRITE = opts.noverwrite
    DEBUG = opts.debug

    if opts.todo:
        todo = list(opts.todo)
    else:
        todo = list("rl")

    req_files = [opts.freq_file, opts.rfile, opts.nrfile, opts.wcs_ref, opts.mask,
        opts.noise_ref]
    req_files.extend(opts.cubes) if opts.cubes else ""
    
    if does_specified_file_exist(*req_files):
        snitch.info("All specified input files have been found")


    ##########################################
    #  For regions
    ##########################################
    if opts.rfile is not None:
        snitch.info(f"Region file: {opts.rfile}")
        # copy custom region file to usual destination
        RDIR = make_out_dir(RDIR)
        RFILE = fullpath(RDIR, "regions-default")
        CURRENT_RFILE = RFILE
        shutil.copy(opts.rfile, RFILE+".reg")
    
    if opts.nrfile is not None:
        snitch.info(f"Region file: {opts.nrfile}")
        RDIR = make_out_dir(RDIR)
        NRFILE = fullpath(RDIR, "noise-region")
        shutil.copy(opts.nrfile, NRFILE+".reg")

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

    if opts.mask is not None:
        bounds = opts.mask
    else:
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
        bounds = [*x_range, *y_range]

    if opts.noise_ref is not None:
        noise_ref = opts.noise_ref
    else:
        noise_ref = "i-mfs.fits"

    if opts.threshold is None:
        threshold = 3
    else:
        threshold = opts.threshold

    if opts.nworkers is not None:
        NWORKERS = opts.nworkers

    if opts.polzd_snr:
        USE_POLZD_SNR = opts.polzd_snr

    # For regions
    if opts.regions_only or "r" in todo and opts.rfile is None:
        step1_default_regions(reg_size, wcs_ref, bounds,
            threshold=threshold, rnoise=rnoise)
        step2_valid_reg_candidates(wcs_ref, noise_ref, threshold, rnoise=rnoise)
    
    # For scrappy
    if opts.los_only or "l" in todo:
 
        if opts.image_dir is not None and os.path.isdir(opts.image_dir):
            image_dir = opts.image_dir
            step3_valid_los_regs(image_dir, threshold, wcs_ref, rnoise=rnoise,
                freq_file=opts.freq_file)
        elif opts.cubes is not None:
            if opts.freq_file is None:
                snitch.error(
                    f"Cubes have been specified. Please specify the associated"+
                    "Frequency text file using --freq-file.")
                sys.exit()

            image_dir = opts.cubes
            step3_valid_los_regs(image_dir, threshold, wcs_ref, rnoise=rnoise,
                freq_file=opts.freq_file)
        else:
            snitch.error(
                f"Check that the specified --image-dir is correct: "+
                "'{opts.image_dir}' not found")
            snitch.error("No directory for the input images was provided")
            snitch.error("Please specifiy using --image dir")
            snitch.error("Byeee!")
            sys.exit()


    # For plots
    if opts.plots_only or "p" in todo:
        # step4_plots()
        pass

    return

def console():
    """A console run entry point for setup.cfg"""
    main()
    snitch.info("Bye :D !")


if __name__ == "__main__":
    console()

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