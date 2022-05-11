#! /usr/bin/env python

"""
Courtesy of Lerato
Edited by Lexy
"""

import numpy
import time
import astropy.io.fits as pyfits
import sys
from multiprocessing import Pool
import argparse
import os
from scipy import signal

from concurrent import futures
from functools import partial
from ipdb import set_trace



def Faraday2Lambda(lam2, phi_range, pol_lam2):

    """
    Computes Faraday Spectra using RM-Synthesis 
    as defined by Brentjens and de Bruyn (2005)

    """

    N = len(lam2)
    l20 = lam2.mean()
    fdata = numpy.zeros([len(phi_range)], dtype=complex)
    

    for k, phi in enumerate(phi_range):
        fdata[k] = pow(N, -1) * numpy.sum(pol_lam2 * 
                numpy.exp(-2j * (lam2-l20) * phi))   
    return fdata



def RMClean(lam2, phi_range, fspectrum, 
          niter=500, gain=0.1):

    fwhm = (3.8/ abs(lam2[0]-lam2[-1]))
    sigma = (fwhm/2.35482)
    Gauss = numpy.exp(-0.5 * (phi_range/sigma)**2) 

    # I am padding here to avoid edge effects.
    
    pad = abs(phi_range[-1]) * 2
    dpad = abs(phi_range[0]-phi_range[1])
    phi_pad = numpy.arange(-pad, pad, dpad)
    dshift = int(pad/(2.0 * dpad))

    rmsf_fixed = Faraday2Lambda(lam2, phi_pad, 1) 
    components = numpy.zeros([len(phi_range)], dtype=complex)

    for n in range(niter):
        temp = numpy.zeros([len(phi_range)], dtype=complex)
        f_amp = numpy.absolute(fspectrum)
        ind = numpy.where(f_amp == f_amp.max())[0]
        f_comp = fspectrum[ind[0]] * gain
        temp[ind[0]] = f_comp
        components += temp         
    
        dirac = numpy.zeros(len(phi_range))
        dirac[ind[0]] = 1
        rmsf = signal.convolve(rmsf_fixed, dirac, mode='same')
        rmsf = rmsf[dshift:-dshift+1]
 
        fspectrum -= f_comp * rmsf

    Fres = fspectrum
    fclean = signal.convolve(components, Gauss, mode='same') + Fres

    return fclean, components


def rm_synthesis(lam, pdata, phi_max=500, phi_step=5,
         niter=1000, gain=0.1, plot=False):


    phi_range =  numpy.arange(-phi_max, phi_max+phi_step, phi_step)
    # this ensures that the middle value is zero. 
    fdirty = Faraday2Lambda(lam, phi_range, pdata)
    
    
    fclean, fcomp = RMClean(lam, phi_range, fdirty, 
                    niter=niter, gain=gain)


    return phi_range, fclean



def read_data(image, freq=True):
                                                                              
    """ Read and return image data: ra, dec and freq only"""
    try:
        with pyfits.open(image) as hdu:
            imagedata = hdu[0].data
            header = hdu[0].header
        imslice = numpy.zeros(imagedata.ndim, dtype=int).tolist()
        imslice[-1] = slice(None)
        imslice[-2] = slice(None)
        if freq:
            imslice[-3] = slice(None)
        print('>>> Image %s loaded sucessfully. '%image)
        return imagedata[tuple(imslice)], header
    except OSError:
        sys.exit('>>> could not load %s. Aborting...'%image)


def call_fitting(args, wavelengths=None):
    x, y = args

    print(f"Processing pixel {x} x {y}")
    # all freqs single pixel
    Pol = pdata[:, x, y]    
    ind_nans =  ~numpy.isnan(abs(Pol))
    Pol = Pol[ind_nans]
    wavelength2 =  wavelengths[ind_nans]

    # RM synth on this single pixel    
    phi_range, fcleaned = rm_synthesis(wavelength2, Pol, 
         phi_max=500, phi_step=5, niter=500, gain=0.1, plot=False) 

    
    famp = abs(fcleaned)
    # get the peak index of the clean faraday spectra
    ind =  numpy.where(famp == famp.max())[0]

    # get the peak RM depth
    RM = phi_range[ind][0]

    # get the peak complex fclean
    fmax = fcleaned[ind][0]
    p0 = numpy.absolute(fmax)

    # get the polarisation angle at the peak
    PA = 0.5 *numpy.arctan2(fmax.imag, fmax.real)
    return p0, PA, RM, famp


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
    xpix, ypix = numpy.where(maskdata > 0)

    return xpix, ypix

def arg_parser():
    parser = argparse.ArgumentParser(description='Performs linear ' 
             'least squares fitting to Q and U image.')

    add = parser.add_argument
    add('-q', '--qcube', dest='qfits', help='Stokes Q cube (fits)')
    add('-u', '--ucube', dest='ufits', help='Stokes U cube (fits)')
    add('-i', '--icube', dest='ifits', help='Stokes I cube (fits)')
    add('-f', '--freq', dest='freq', help='Frequency file (text)')  
    add('-ncore', '--numpr', dest='numProcessor', help='number of cores to use. Default 1.', 
        default=60, type=int)
    add('-mask', '--maskfits', dest='maskfits', help='A mask image (fits)', default=None)
    add('-o', '--prefix', dest='prefix', help='This is a prefix for output files.')
   
    return parser.parse_args()


def main():
    args = arg_parser()

    global pdata, p0_map, PA_map, RM_map
    
    try:
        frequencies = numpy.loadtxt(args.freq)
    except ValueError:
        print(">>> Problem found with frequency file. It should be a text file")
        sys.exit(">>> Exiting. See log file %s" %LOG_FILENAME)
    

    wavelengths =  (299792458.0/frequencies)**2 
    qdata, qhdr = read_data(args.qfits) # run Q-cube 
    udata, uhdr = read_data(args.ufits) # run U-cube

    if args.ifits:
        idata, ihdr = read_data(args.ifits) # run I-cube
        fpdata = (qdata/idata) + 1j * (udata/idata)

    
    

    pdata = qdata + 1j * udata
 
    N1, N2, N3 = qdata.shape
    p0_map = numpy.zeros([N2, N3])
    PA_map = numpy.zeros([N2, N3])
    RM_map = numpy.zeros([N2, N3])

    

    #q_cube = numpy.zeros([N1, N2, N3])
    #u_cube = numpy.zeros([N1, N2, N3])

    if args.maskfits:
        x, y = read_mask(args.maskfits)
    
    else:
        x, y = numpy.indices((N2, N3))
        x = x.flatten()
        y = y.flatten()


    xy = list(zip(x, y))

    results = []
    with futures.ProcessPoolExecutor(args.numProcessor) as executor:
        results = executor.map(partial(call_fitting,wavelengths=wavelengths), xy)

    
    results = list(results)
    for _, an in enumerate(xy):        
        # # making new pixels with the new the poln angle and the
        p0_map[an] = results[_][0]
        PA_map[an] = results[_][1]
        RM_map[an] = results[_][2]
    
    # now save the fits
    p0_hdr = modify_fits_header(qhdr, ctype='p', unit='ratio')
    PA_hdr = modify_fits_header(qhdr, ctype='PA', unit='radians')
    RM_hdr = modify_fits_header(qhdr, ctype='RM', unit='rad/m/m')
    pyfits.writeto(args.prefix + '-p0.fits', p0_map, p0_hdr, overwrite=True)
    pyfits.writeto(args.prefix + '-PA.fits', PA_map, PA_hdr, overwrite=True)
    pyfits.writeto(args.prefix + '-RM.fits', RM_map, RM_hdr, overwrite=True)
    pyfits.writeto(args.prefix + '-FPOL.fits', RM_map, RM_hdr, overwrite=True)

    print("Donesies!")

    #pyfits.writeto(args.prefix + '-qFAR.fits', q_cube, qhdr, overwrite=True)
    #pyfits.writeto(args.prefix + '-uFAR.fits', u_cube, qhdr, overwrite=True)




if __name__=='__main__':

    main()


    """
    Running
    time python pica_rm-x2.py -q Q-image-cubes.fits -u U-image-cubes.fits -i I-image-cubes.fits -ncore 120 -o turbo -f Frequencies-PicA-Masked.txt -mask 572-mask.fits


    """