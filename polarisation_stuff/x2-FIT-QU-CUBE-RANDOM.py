#! /usr/bin/env python

import time
import sys
import argparse
import os
import logging

import numpy as np

from astropy.io import fits
from lmfit import minimize, Parameters, Minimizer
from multiprocessing import Pool
from concurrent import futures
from functools import partial
from ipdb import set_trace



def read_data(image, freq=True):
                                                                              
    """ Read and return image data: ra, dec and freq only"""
    try:
        with fits.open(image) as hdu:
            imagedata = hdu[0].data
            header = hdu[0].header
        imslice = np.zeros(imagedata.ndim, dtype=int).tolist()
        imslice[-1] = slice(None)
        imslice[-2] = slice(None)
        if freq:
            imslice[-3] = slice(None)
        print('>>> Image %s loaded sucessfully. '%image)
        return imagedata[tuple(imslice)], header
    except OSError:
        sys.exit('>>> could not load %s. Aborting...'%image)


def check_shape(qfits, ufits, frequencies):
    """
    Checks the shape of the cubes and frequency file
    """
    qhdr =  fits.getheader(qfits)
    uhdr =  fits.getheader(ufits)
    print(qhdr['naxis3'], uhdr['naxis3'], frequencies.shape)
    errors = []
    axis = ['naxis1', 'naxis2', 'naxis3']
    if qhdr['naxis'] < 3 or uhdr['naxis'] < 3:
        errors.append('The dimensions of Q = %d and U = %d, not >=3.' %(
               qhdr['naxis'], uhdr['naxis']))
    if qhdr['naxis'] != uhdr['naxis']:
        if qhdr['naxis'] >= 3 and uhdr['naxis'] >=3:
             for ax in axis:
                 if qhdr[ax] != uhdr[ax]:
                      errors.append('%s for Q is != %d of U.' %(ax, qhdr[ax], uhdr[ax]))
    if qhdr[axis[2]] != len(frequencies) or uhdr[axis[2]] != len(frequencies):
        errors.append('Freq-axis of the cubes differ from that of the '
           'frequency file provided.')
            
    return errors


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



def realimag(array):
     return np.array([(x.real, x.imag) for x in array]).flatten()


def model(param, model=False, eps=False, extras=None):

    """linear model """
    p0 = param['p0']
    RM = param['RM']
    PA = param['PA']
    dRM = param['dRM']
    #DRM = param['DRM']
    waves = extras.get("waves")
    fpol = extras.get("fpol")
    sigma_qu = extras.get("sigma_qu")
    p = p0 * np.exp(2j * (PA  + RM * waves) )  * np.exp(-2 * dRM**2 * waves**2) #np.sinc(DRM * wavelengths)
    if model:
        return p
    if eps:
        residual = realimag(fpol) -  realimag(p)
        sigma = realimag(sigma_qu)
        return  np.sqrt( residual**2/ sigma**2 )
    else:
        diff =  realimag(fpol) -  realimag(p)
        return  diff



def call_model(p0=0.5, PA=1, PA_min=-np.pi/2.0, PA_max= np.pi/2.0, RM=10, 
              RM_min=-12500, RM_max=12500, dRM=50, dRM_min=0, dRM_max=2500,
              extras=None):
     params = Parameters()
     params.add('p0', value=p0, min=0.0001, max=1.0)
     params.add('PA', value=0, min=PA_min, max=PA_max)
     params.add('RM', value=RM, min=RM_min, max=RM_max)
     params.add('dRM', value=dRM, min=dRM_min, max=dRM_max)
     #params.add('DRM', value=50, min=-3000, max=3000)

     kw = {}
     kw['ftol'] = 1e-30

     start = time.time()
     mini =  Minimizer(partial(model, extras=extras), params)
     #fit_residual = mini.minimize(method='nedger')
     fit_residual = mini.minimize(method='leastsq', params=params)
     end = time.time()
     return fit_residual


def call_fitting2(xy_pix, wavelengths=None, noise_q=None, noise_u=None, noise_i=None):

    x, y = xy_pix
    fpol = np.ma.masked_invalid(fpol_data[:, x, y])
    inv_mask = fpol.mask
    fpol = fpol.compressed()

    q = np.ma.masked_array(data=qdata[:, x, y], mask=inv_mask).compressed()
    u = np.ma.masked_array(data=udata[:, x, y], mask=inv_mask).compressed()
    i = np.ma.masked_array(data=idata[:, x, y], mask=inv_mask).compressed()
    # fpol = fpol_data[:, x, y]
    fq = np.ma.masked_array(data=frac_q[:, x, y], mask=inv_mask).compressed()
    fu = np.ma.masked_array(data=frac_u[:, x, y], mask=inv_mask).compressed()

    waves = np.ma.masked_array(data=wavelengths, mask=inv_mask).compressed()
    noise_q =  np.ma.masked_array(data=noise_q, mask=inv_mask).compressed()
    noise_u =  np.ma.masked_array(data=noise_u, mask=inv_mask).compressed()
    noise_i =  np.ma.masked_array(data=noise_i, mask=inv_mask).compressed()

    sigma_q = abs(fq) * ( (noise_q/q)**2 + (noise_i/i)**2 )**0.5
    sigma_u = abs(fu) * ( (noise_u/u)**2 + (noise_i/i)**2 )**0.5
    sigma_qu = sigma_q + 1j * sigma_u

    theta = 0.5 *  np.arctan2(fpol[-1].imag, fpol[-1].real)
    rm = rmdata[x, y]
    fpol0 = pmax_data[x, y]
    extras = dict(waves=waves, fpol=fpol, sigma_qu=sigma_qu)
    fit_residual =  call_model(p0=fpol0, PA=theta, RM=rm, extras=extras)
 
    outs = dict()
    outs["p0"] = fit_residual.params['p0'].value
    outs["p0_err"] = fit_residual.params['p0'].stderr
    outs["pa"] = fit_residual.params['PA'].value
    outs["pa_err"] = fit_residual.params['PA'].stderr
    outs["rm"] = fit_residual.params['RM'].value
    outs["rm_err"] = fit_residual.params['RM'].stderr   
    outs["drm"] = fit_residual.params['dRM'].value
    outs["drm_err"] = fit_residual.params['dRM'].stderr
    outs["red_chi_sq"] = fit_residual.redchi
    outs["chi_sq"] = fit_residual.chisqr
    outs["aic"] = fit_residual.aic
    outs["bic"] = fit_residual.bic
    logging.debug(f'Processing pixel {x} x {y}')
    print(f'Processing pixel {x} x {y}')

    return outs


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Performs linear ' 
             'least squares fitting to Q and U image.')

    add = parser.add_argument
    add('-q', '--qcube', dest='qfits', help='Stokes Q cube (fits)')
    add('-u', '--ucube', dest='ufits', help='Stokes U cube (fits)')
    add('-i', '--icube', dest='ifits', help='Stokes I cube (fits)')
    add('-f', '--freq', dest='freq', help='Frequency file (text)')  
    add('-n', '--noise', dest='noise', help='noise text file. Q, U and I in a single file.')
    add('-ncore', '--numpr', dest='numProcessor', help='number of cores to use. Default 1.', 
        default=10, type=int)
    add('-mask', '--maskfits', dest='maskfits', help='A mask image (fits). This package '
        'comes with a tool called cleanmask to allow the user to create a mask, for more info '
        'cleanmask -h', default=None)
    add('-rmimg', '--rm-image', dest='rm_image', help='First guess RM image (fits)', default=None)
    add('-pmaximg', '--pmax-image', dest='pmax_image', help='First guess FPOL peak image (fits)', default=None)
    add('-rmch', '--rmchannel', dest='remove_channel',  help='list of channels to remove', 
        type=str, action='append', default=None)
    add('-o', '--prefix', dest='prefix', help='This is a prefix for output files.')
   
    args = parser.parse_args()

    LOG_FILENAME = args.prefix + '.log'
    logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)
    

    try:
        frequencies = np.loadtxt(args.freq)
    except ValueError:
        logging.debug(">>> Problem found with frequency file. It should be a text file")
        sys.exit(">>> Exiting. See log file %s" %LOG_FILENAME)

    errors = check_shape(args.qfits, args.ufits, frequencies)
    if len(errors) > 0:
        logging.debug(errors)
        sys.exit(">>> Exiting. See log file %s" %LOG_FILENAME)

    
    global idata, udata, qdata, rmdata, wavelengths, pmax_data, noise_q, noise_u, noise_i, fpol_data, frac_q, frac_u

    wavelengths =  (299792458.0/frequencies)**2 
    qdata, qhdr = read_data(args.qfits) # run Q-cube 
    udata, uhdr = read_data(args.ufits) # run U-cube
    idata, ihdr = read_data(args.ifits) # run I-cube

    frac_q, frac_u = qdata/idata, udata/idata 
    fpol_data = frac_q + 1j * frac_u

    if args.rm_image:
       rmdata, hdr = read_data(args.rm_image, freq=False)
    if args.pmax_image:
       pmax_data, hdr = read_data(args.pmax_image, freq=False)


    noise = np.loadtxt(args.noise)
    print(noise.shape)
    noise_q = noise[:, 0]
    noise_u = noise[:, 1]
    noise_i = noise[:, 2]

    # make empty 2-D arrays to hold future data
    N_wave, N_x, N_y = qdata.shape
    p0_cube = np.zeros([N_x, N_y])
    PA_cube = np.zeros([N_x, N_y])
    RM_cube = np.zeros([N_x, N_y])
    dRM_cube = np.zeros([N_x, N_y])
    #DRM_cube = np.zeros([N_x, N_y])
    p0err_cube = np.zeros([N_x, N_y])
    PAerr_cube = np.zeros([N_x, N_y])
    RMerr_cube = np.zeros([N_x, N_y])
    dRMerr_cube = np.zeros([N_x, N_y])
    #DRMerr_cube = np.zeros([N_x, N_y])
    REDUCED_CHISQR = np.zeros([N_x, N_y])
    CHISQR = np.zeros([N_x, N_y])
    AIC = np.zeros([N_x, N_y])
    BIC = np.zeros([N_x, N_y])

    if args.maskfits:
        x, y = read_mask(args.maskfits)
    
    else:
        x, y = np.indices((N_x, N_y))
        x = x.flatten()
        y = y.flatten()

    start1 = time.time()
    
    xy = list(zip(x, y))
    results = []

    print("Starting")
    with futures.ProcessPoolExecutor(args.numProcessor) as executor:
        results = executor.map(
            partial(call_fitting2, wavelengths=wavelengths, noise_q=noise_q,
                      noise_u=noise_u, noise_i=noise_i),
            xy)
        
    
    # # test bedding
    # for _v in xy:
    #     results.append(
    #         call_fitting2(_v, wavelengths=wavelengths, noise_q=noise_q,
    #                   noise_u=noise_u, noise_i=noise_i))

    results = list(results)
    for _, an in enumerate(xy):        
        # # making new pixels with the new the poln angle and the
        p0_cube[an] = results[_]["p0"]
        PA_cube[an] = results[_]["pa"]
        RM_cube[an] = results[_]["rm"]
        dRM_cube[an] = results[_]["drm"]
        p0err_cube[an] = results[_]["p0_err"]
        PAerr_cube[an] = results[_]["pa_err"]
        RMerr_cube[an] = results[_]["rm_err"]
        dRMerr_cube[an] = results[_]["drm_err"]
        REDUCED_CHISQR[an] = results[_]["red_chi_sq"]
        CHISQR[an] = results[_]["chi_sq"]
        AIC[an] = results[_]["aic"]
        BIC[an] = results[_]["bic"]

    end1 = time.time()
    logging.debug('Total time for multiprocessing is %.6f  seconds. '%(end1-start1))
    
    # now save the fits
    p0_hdr = modify_fits_header(qhdr, ctype='p', unit='ratio')
    PA_hdr = modify_fits_header(qhdr, ctype='PA', unit='radians')
    RM_hdr = modify_fits_header(qhdr, ctype='RM', unit='rad/m/m')
    fit_hdr = modify_fits_header(qhdr, ctype='fit', unit='None')
    #dRM_hdr = modify_fits_header(qhdr, ctype='dRM', unit='rad^2/m^4')

    fits.writeto(args.prefix + '-p0.FITS', p0_cube, p0_hdr, overwrite=True)
    fits.writeto(args.prefix + '-PA.FITS', PA_cube, PA_hdr, overwrite=True)
    fits.writeto(args.prefix + '-RM.FITS', RM_cube, RM_hdr, overwrite=True)
    #fits.writeto(args.prefix + '-dRM.FITS', dRM_cube, dRM_hdr, overwrite=True)
    fits.writeto(args.prefix + '-dRM.FITS', dRM_cube, RM_hdr, overwrite=True)

    fits.writeto(args.prefix + '-p0err.FITS', p0err_cube, p0_hdr, overwrite=True)
    fits.writeto(args.prefix + '-PAerr.FITS', PAerr_cube, PA_hdr, overwrite=True)
    fits.writeto(args.prefix + '-RMerr.FITS', RMerr_cube, RM_hdr, overwrite=True)
    #fits.writeto(args.prefix + '-dRMerr.FITS', dRMerr_cube, dRM_hdr, overwrite=True)
    fits.writeto(args.prefix + '-dRMerr.FITS', dRMerr_cube, RM_hdr, overwrite=True)


    fits.writeto(args.prefix + '-REDCHI.FITS', REDUCED_CHISQR, fit_hdr, overwrite=True)
    fits.writeto(args.prefix + '-CHI.FITS', CHISQR, fit_hdr, overwrite=True)
    fits.writeto(args.prefix + '-AIC.FITS', AIC, fit_hdr, overwrite=True)
    fits.writeto(args.prefix + '-BIC.FITS', BIC, fit_hdr, overwrite=True)

