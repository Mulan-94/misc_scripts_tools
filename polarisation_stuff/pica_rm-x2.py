#! /usr/bin/env python

"""
Courtesy of Lerato
Edited by Lexy
"""
import argparse
import logging
import os
import time
import sys
import warnings

import astropy.io.fits as pyfits
import numpy as np

from multiprocessing import Pool
from scipy import signal
from concurrent import futures
from functools import partial
from ipdb import set_trace

def configure_logger(out_dir):
# ignore overflow errors, assume these to be mostly flagged data
    warnings.simplefilter("ignore")

    formatter = logging.Formatter(
        datefmt='%H:%M:%S %d.%m.%Y',
        fmt="%(asctime)s : %(levelname)s - %(message)s")
    
    l_handler = logging.FileHandler(
        os.path.join(out_dir, "turbo-rm-x2.log"), mode="w")
    l_handler.setLevel(logging.INFO)
    l_handler.setFormatter(formatter)

    s_handler = logging.StreamHandler()
    s_handler.setLevel(logging.INFO)
    s_handler.setFormatter(formatter)

    logger = logging.getLogger("turbo-rm")
    logger.setLevel(logging.INFO)

    logger.addHandler(l_handler)
    logger.addHandler(s_handler)
    return logger


def faraday_to_lambda(lam2, phi_range, pol_lam2):

    """
    Computes Faraday Spectra using RM-Synthesis 
    as defined by Brentjens and de Bruyn (2005)

    """

    N = len(lam2)
    l20 = lam2.mean()
    fdata = np.zeros([len(phi_range)], dtype=complex)
    

    for k, phi in enumerate(phi_range):
        fdata[k] = pow(N, -1) * np.sum(pol_lam2 * 
                np.exp(-2j * (lam2-l20) * phi))   
    return fdata



def rm_clean(lam2, phi_range, fspectrum, niter=500, gain=0.1):

    fwhm = (3.8/ abs(lam2[0]-lam2[-1]))
    sigma = (fwhm/2.35482)
    Gauss = np.exp(-0.5 * (phi_range/sigma)**2) 

    # I am padding here to avoid edge effects.
    
    pad = abs(phi_range[-1]) * 2
    dpad = abs(phi_range[0]-phi_range[1])
    phi_pad = np.arange(-pad, pad, dpad)
    dshift = int(pad/(2.0 * dpad))

    rmsf_fixed = faraday_to_lambda(lam2, phi_pad, 1) 
    components = np.zeros([len(phi_range)], dtype=complex)

    for n in range(niter):
        temp = np.zeros([len(phi_range)], dtype=complex)
        f_amp = np.absolute(fspectrum)
        ind = np.where(f_amp == f_amp.max())[0]
        f_comp = fspectrum[ind[0]] * gain
        temp[ind[0]] = f_comp
        components += temp         
    
        dirac = np.zeros(len(phi_range))
        dirac[ind[0]] = 1
        rmsf = signal.convolve(rmsf_fixed, dirac, mode='same')
        rmsf = rmsf[dshift:-dshift+1]
 
        fspectrum -= f_comp * rmsf

    Fres = fspectrum
    fclean = signal.convolve(components, Gauss, mode='same') + Fres

    return fclean, components


def rm_synthesis(lam, pdata, phi_max=500, phi_step=5, niter=1000, gain=0.1, plot=False):


    phi_range =  np.arange(-phi_max, phi_max+phi_step, phi_step)
    # this ensures that the middle value is zero. 
    fdirty = faraday_to_lambda(lam, phi_range, pdata)
    
    
    fclean, fcomp = rm_clean(lam, phi_range, fdirty, 
                    niter=niter, gain=gain)


    return phi_range, fclean



def read_data(image, freq=True):
                                                                              
    """ Read and return image data: ra, dec and freq only"""
    try:
        with pyfits.open(image) as hdu:
            imagedata = hdu[0].data
            header = hdu[0].header
        imslice = np.zeros(imagedata.ndim, dtype=int).tolist()
        imslice[-1] = slice(None)
        imslice[-2] = slice(None)
        if freq:
            imslice[-3] = slice(None)
        snitch.info('>>> Image %s loaded sucessfully. '%image)
        return imagedata[tuple(imslice)], header
    except OSError:
        sys.exit('>>> could not load %s. Aborting...'%image)


def call_fitting(args, wavelengths=None, max_depth=500, niters=100, phi_step=1):
    """
    Returns

    p0_map
    PA_map
    RM_map
    fp_map

    peak_rm_amp: p0_map
        Amplitude of the clean peak RM
    pol_angle: PA_map
        Polarisation angle at clean peak
    peak_depth
        Faraday depth at clean RM peak
    cleaned_amp
        Amplitude of cleaned the entire Farady  spectra
    peak_fpol
        Fpol at the center freq, or the peak at all freqs


    pdata: is the total linearly polarised intensity
    fpdata: is the fractional polarisation
    """
    x, y = args

    snitch.info(f"Processing pixel {x} x {y}")
    # all freqs single pixel
    single_pxl_poln = pdata[:, x, y]    
    ind_nans =  ~np.isnan(abs(single_pxl_poln))
    single_pxl_poln = single_pxl_poln[ind_nans]
    wave_sq =  wavelengths[ind_nans]

    # RM-synth and RM-CLEAN on this single pixel. We pass in the complex polarised data
    phi_range, fcleaned = rm_synthesis(wave_sq, single_pxl_poln, phi_max=max_depth,
        phi_step=phi_step, niter=niters, gain=0.1, plot=False) 

    abs_fclean = np.abs(fcleaned)

    # get the peak index of the clean faraday spectra
    peak_idx =  np.where(abs_fclean == abs_fclean.max())[0][0]

    # get the depth at which RM is peak. Is this the estimated RM?
    peak_depth = phi_range[peak_idx]

    # get the peak complex fclean
    peak_complex_rm = fcleaned[peak_idx]

    #this gives the estimated intrinsic fractional polarisation??
    peak_rm_amp = np.abs(peak_complex_rm)

    # get the polarisation angle at the peak fday spectrum (intrinsic fpol?)
    # of that component. Note that this is from the cleaned FDF
    pol_angle = 0.5 * np.arctan2(peak_complex_rm.imag, peak_complex_rm.real)

    # Trying to get the frac pol at max polarised intensity for this x,y pixel
    # nfp = np.abs(fp_data[:, x, y])
    nfp = fp_data[:, x, y]

    # get location of peak polzn intensity
    peak_lpol = np.abs(single_pxl_poln).max()
    max_pol_idx = np.where(np.abs(single_pxl_poln)==peak_lpol)[0][0]  
    # peak_fpol = nfp[max_pol_idx]
    peak_fpol = nfp[-1]
    snitch.info(f"x: {y}, y:{x}, max polzn intensity @ chan: {max_pol_idx}")
    return peak_rm_amp, pol_angle, peak_depth, abs(fcleaned), peak_fpol, peak_lpol


def modify_fits_header(header, ctype='RM', unit='rad/m/m'):

    """
    Modify header   
    """
    hdr = header.copy()
    new_hdr = {'naxis3': 1, 'cunit3': unit, 
               'ctype3':ctype,'bunit': unit}

    hdr.update(new_hdr) 
    return hdr


def read_mask(fits):

    """get pixel coordinates for a mask"""
    maskdata, mhdr = read_data(fits, freq=False)
    xpix, ypix = np.where(maskdata > 0)

    return xpix, ypix

def arg_parser():
    parser = argparse.ArgumentParser(description='Performs linear ' 
             'least squares fitting to Q and U image.')

    add = parser.add_argument
    add('-q', '--qcube', dest='qfits', help='Stokes Q cube (fits)', type=str)
    add('-u', '--ucube', dest='ufits', help='Stokes U cube (fits)', type=str)
    add('-i', '--icube', dest='ifits', help='Stokes I cube (fits)', type=str)
    add('-f', '--freq', dest='freq', help='Frequency file (text)', type=str)

    add('-ncore', '--ncore', dest='numProcessor',
        help='number of cores to use. Default 60.', default=60, type=int)
    add('-mask', '--maskfits', dest='maskfits',
        help='A mask image (fits)', default=None, type=str)
    add('-o', '--prefix', dest='prefix', help='This is a prefix for output files.', type=str)
    add("-niters", "--niters", dest="niters",
        help="Number of clean iterations. Default 1000", default=1000, type=int)
    add("-md", "--max-depth", dest="max_depth",
        help="Maximum Farady depth to fit for. Default 500", default=500, type=int)
    add("--depth-step", dest="depth_step",
        help="Depth stepping. Default 1", default=1, type=int)
   
    return parser.parse_args()


def main():
    args = arg_parser()

    # making these global because of multiprocess non-sharing.
    # Global variables can not be modified and shared between different processes
    global pdata, fp_data, snitch

    snitch = configure_logger(".")
    
    try:
        frequencies = np.loadtxt(args.freq)
    except ValueError:
        snitch.error(">>> Problem found with frequency file. It should be a text file")
        sys.exit(">>> Exiting. See log file 'turbo-rm.log'")
    

    wavelengths =  (299792458.0/frequencies)**2 
    qdata, qhdr = read_data(args.qfits) # run Q-cube 
    udata, uhdr = read_data(args.ufits) # run U-cube

    pdata = qdata + 1j*udata

    if args.ifits:
        idata, ihdr = read_data(args.ifits) # run I-cube
        fp_data = np.abs(pdata) / idata

 
    N1, N2, N3 = qdata.shape
    p0_map = np.zeros([N2, N3])
    PA_map = np.zeros([N2, N3])
    RM_map = np.zeros([N2, N3])
    fp_map = np.zeros([N2, N3])
    lp_map = np.zeros([N2, N3])

    
    #q_cube = np.zeros([N1, N2, N3])
    #u_cube = np.zeros([N1, N2, N3])

    if args.maskfits:
        x, y = read_mask(args.maskfits)
    
    else:
        x, y = np.indices((N2, N3))
        x = x.flatten()
        y = y.flatten()


    xy = list(zip(x, y))

    results = []
    with futures.ProcessPoolExecutor(args.numProcessor) as executor:
        results = executor.map(
            partial(
                call_fitting,wavelengths=wavelengths, max_depth=args.max_depth,
                niters=args.niters, phi_step=args.depth_step), 
            xy)

    # # test bedding
    # for _v in xy:
    #     results.append(call_fitting(_v, wavelengths=wavelengths))

    
    results = list(results)
    for _, an in enumerate(xy):        
        # # making new pixels with the new the poln angle and the
        p0_map[an] = results[_][0]
        PA_map[an] = results[_][1]
        RM_map[an] = results[_][2]
        fp_map[an] = results[_][4]
        lp_map[an] = results[_][5]

    
    
    # now save the fits
    # intrisic fractional polarisation
    p0_hdr = modify_fits_header(qhdr, ctype='p', unit='ratio')

    #polarisation angle
    PA_hdr = modify_fits_header(qhdr, ctype='PA', unit='radians')

    # RM - depth at the peak faraday spectrum
    RM_hdr = modify_fits_header(qhdr, ctype='RM', unit='rad/m^2')

    # polarised flux
    pf_hdr = modify_fits_header(qhdr, ctype='FPOL', unit='x100%')

    # polarise intensity
    lp_hdr = modify_fits_header(qhdr, ctype='LPOL', unit='unit')

    # Amplitude of the clean peak RM (intrinsic fpol)
    pyfits.writeto(
        args.prefix + '-p0-peak-FDF.fits', p0_map, p0_hdr, overwrite=True)

    # pol angle at peak RM (polarisation angle)
    pyfits.writeto(
        args.prefix + '-PA-pangle-at-peak-rm.fits', PA_map, PA_hdr,
        overwrite=True)

    # Depth at peak RM (Actual RM at the peak intrisic fpol)
    pyfits.writeto(
        args.prefix + '-RM-depth-at-peak-rm.fits', RM_map, RM_hdr,
        overwrite=True)

    #frac pol at central freq
    pyfits.writeto(
        args.prefix + '-FPOL-at-max-lpol.fits', fp_map, pf_hdr,
        overwrite=True)
    
     #frac pol at central freq
    pyfits.writeto(
        args.prefix + '-max-LPOL.fits', lp_map, lp_hdr,
        overwrite=True)

    snitch.info("Donesies!")

    #pyfits.writeto(args.prefix + '-qFAR.fits', q_cube, qhdr, overwrite=True)
    #pyfits.writeto(args.prefix + '-uFAR.fits', u_cube, qhdr, overwrite=True)




if __name__=='__main__':

    main()


    """
    Running
    
    time python pica_rm-x2.py -q Q-image-cubes.fits -u U-image-cubes.fits -i I-image-cubes.fits -ncore 120 -o turbo -f Frequencies-PicA-Masked.txt -mask true_mask_572.fits

    python pica_rm-x2.py -q Q-image-cubes.fits -u U-image-cubes.fits -i I-image-cubes.fits -ncore 120 -o from_rick/outputs/with-NHS-data-mask -f Frequencies-PicA-Masked.txt -mask masks/nhs-mask-572.fits

    python pica_rm-x2.py -q Q-image-cubes.fits -u U-image-cubes.fits -i I-image-cubes.fits -ncore 120 -o from_rick/outputs/with-ricks-data-mask -f Frequencies-PicA-Masked.txt -mask from_rick/masks/ricks-data-mask.fits

    """
    