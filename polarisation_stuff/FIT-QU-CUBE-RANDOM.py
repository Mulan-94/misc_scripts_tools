#! /usr/bin/env python


import numpy
import pylab
from lmfit import minimize, Parameters, Minimizer
import time
import astropy.io.fits as pyfits
import sys
from multiprocessing import Pool
import argparse
import os
import logging



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


def check_shape(qfits, ufits, frequencies):
    """
    Checks the shape of the cubes and frequency file
    """
    qhdr =  pyfits.getheader(qfits)
    uhdr =  pyfits.getheader(ufits)
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
    xpix, ypix = numpy.where(maskdata > 0)
    return xpix, ypix



def realimag(array):
     return numpy.array([(x.real, x.imag) for x in array]).flatten()


def model(param, model=False, eps=False):

    """linear model """
    p0 = param['p0']
    RM = param['RM']
    PA = param['PA']
    dRM = param['dRM']
    #DRM = param['DRM']
    p = p0 * numpy.exp(2j * (PA  + RM * wavelengths) )  * numpy.exp(-2 * dRM**2 * wavelengths**2) #numpy.sinc(DRM * wavelengths)
    if model:
        return p
    if eps:
        residual = realimag(fpol) -  realimag(p)
        sigma = realimag(sigmaqu)
        return  numpy.sqrt( residual**2/ sigma**2 )
    else:
        diff =  realimag(fpol) -  realimag(p)
        return  diff



def call_model(p0=0.5, PA=1, PA_min=-numpy.pi/2.0, PA_max= numpy.pi/2.0, RM=10, 
              RM_min=-12500, RM_max=12500, dRM=50, dRM_min=0, dRM_max=2500):
     params = Parameters()
     params.add('p0', value=p0, min=0.0001, max=1.0)
     params.add('PA', value=0, min=PA_min, max=PA_max)
     params.add('RM', value=RM, min=RM_min, max=RM_max)
     params.add('dRM', value=dRM, min=dRM_min, max=dRM_max)
     #params.add('DRM', value=50, min=-3000, max=3000)

     kw = {}
     kw['ftol'] = 1e-30

     start = time.time()
     mini =  Minimizer(model, params)
     #fit_residual = mini.minimize(method='nedger')
     fit_residual = mini.minimize(method='leastsq', params=params, **kw)
     end = time.time()
     return fit_residual



def call_fitting(x, y):

    global fpol, wavelengths, sigmaqu, noiseq, noiseu, noisei
    
    Q = qdata[:, x, y]
    U = udata[:, x, y]
    I = idata[:, x, y]
    
    q, u = Q/I, U/I 
    fpol = q + 1j * u

    ind_nans =  ~numpy.isnan(fpol)
    fpol = fpol[ind_nans]
    q = q[ind_nans]
    u = u[ind_nans]
    Q = Q[ind_nans]
    U = U[ind_nans]
    I = I[ind_nans]
    wavelengths = wavelengths[ind_nans]
    noiseq = noiseq[ind_nans]
    noiseu = noiseu[ind_nans]
    noisei = noisei[ind_nans]

    sigmaq = abs(q) * ( (noiseq/Q)**2 + (noisei/I)**2 )**0.5
    sigmau = abs(u) * ( (noiseu/U)**2 + (noisei/I)**2 )**0.5
    sigmaqu = sigmaq + 1j * sigmau

    theta = 0.5 *  numpy.arctan2(fpol.imag[-1], fpol.real[-1])
    rm = rmdata[x, y]
    fpol0 = pmax[x, y]

    fit_residual =  call_model(p0=fpol0, PA=theta, RM=rm)
    p0_fit = fit_residual.params['p0'].value
    p0_err = fit_residual.params['p0'].stderr
    PA_fit = fit_residual.params['PA'].value
    PA_err = fit_residual.params['PA'].stderr
    RM_fit = fit_residual.params['RM'].value
    RM_err = fit_residual.params['RM'].stderr   
    dRM_fit = fit_residual.params['dRM'].value
    dRM_err = fit_residual.params['dRM'].stderr
    #DRM_fit = fit_residual.params['DRM'].value
    #DRM_err = fit_residual.params['DRM'].stderr


    aic = fit_residual.aic
    bic = fit_residual.bic
    redsqr = fit_residual.redchi
    chisqr = fit_residual.chisqr
    return p0_fit, p0_err, PA_fit, PA_err, RM_fit, RM_err, dRM_fit, dRM_err, chisqr, redsqr, aic, bic


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
        default=1, type=int)
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
        frequencies = numpy.loadtxt(args.freq)
    except ValueError:
        logging.debug(">>> Problem found with frequency file. It should be a text file")
        sys.exit(">>> Exiting. See log file %s" %LOG_FILENAME)

    errors = check_shape(args.qfits, args.ufits, frequencies)
    if len(errors) > 0:
        logging.debug(errors)
        sys.exit(">>> Exiting. See log file %s" %LOG_FILENAME)

    
    global idata, udata, qdata, wavelengths, rmdata, pmax, noiseq, noiseu, noisei

    wavelengths =  (299792458.0/frequencies)**2 
    qdata, qhdr = read_data(args.qfits) # run Q-cube 
    udata, uhdr = read_data(args.ufits) # run U-cube
    idata, ihdr = read_data(args.ifits) # run I-cube

    if args.rm_image:
       rmdata, hdr = read_data(args.rm_image, freq=False)
    if args.pmax_image:
       pmax, hdr = read_data(args.pmax_image, freq=False)


    noise = numpy.loadtxt(args.noise)
    print(noise.shape)
    noiseq = noise[:, 0]
    noiseu = noise[:, 1]
    noisei = noise[:, 2]

    N_wave, N_x, N_y = qdata.shape
    p0_cube = numpy.zeros([N_x, N_y])
    PA_cube = numpy.zeros([N_x, N_y])
    RM_cube = numpy.zeros([N_x, N_y])
    dRM_cube = numpy.zeros([N_x, N_y])
    #DRM_cube = numpy.zeros([N_x, N_y])
    p0err_cube = numpy.zeros([N_x, N_y])
    PAerr_cube = numpy.zeros([N_x, N_y])
    RMerr_cube = numpy.zeros([N_x, N_y])
    dRMerr_cube = numpy.zeros([N_x, N_y])
    #DRMerr_cube = numpy.zeros([N_x, N_y])
    
    REDUCED_CHISQR = numpy.zeros([N_x, N_y])
    CHISQR = numpy.zeros([N_x, N_y])
    AIC = numpy.zeros([N_x, N_y])
    BIC = numpy.zeros([N_x, N_y])

    if args.maskfits:
        x, y = read_mask(args.maskfits)
    
    else:
        x, y = numpy.indices((N_x, N_y))
        x = x.flatten()
        y = y.flatten()

    N = len(x)
    pool = Pool(args.numProcessor)
    start1 = time.time()
    for i, (xx, yy) in enumerate(zip(x, y)):  
       start_all = time.time()
       p0, p0err, PA, PAerr, RM, RMerr, dRM, dRMerr, chisqr, redsqr, aic, bic = pool.apply(call_fitting, args=(xx, yy))

       p0_cube[xx, yy] = p0
       p0err_cube[xx, yy] = p0err

       PA_cube[xx, yy] = PA
       PAerr_cube[xx, yy] = PAerr

       RM_cube[xx, yy] = RM
       RMerr_cube[xx, yy] = RMerr

       dRM_cube[xx, yy] = dRM
       dRMerr_cube[xx, yy] = dRMerr

       #DRM_cube[xx, yy] = DRM
       #DRMerr_cube[xx, yy] = DRMerr

       REDUCED_CHISQR[xx, yy] = redsqr
       CHISQR[xx, yy] = chisqr
       AIC[xx, yy] = aic
       BIC[xx, yy] = bic

       end_all = time.time()
       #print('Processing pixel %d/%d for %.6f.'%(i, N, end_all-start_all))
       logging.debug('Processing pixel %d/%d, %.6f seconds. '%(i, N, end_all-start_all))
    end1 = time.time()
    logging.debug('Total time for multiprocessing is %.6f  seconds. '%(end1-start1))
    
    # now save the fits
    p0_hdr = modify_fits_header(qhdr, ctype='p', unit='ratio')
    PA_hdr = modify_fits_header(qhdr, ctype='PA', unit='radians')
    RM_hdr = modify_fits_header(qhdr, ctype='RM', unit='rad/m/m')
    fit_hdr = modify_fits_header(qhdr, ctype='fit', unit='None')
    #dRM_hdr = modify_fits_header(qhdr, ctype='dRM', unit='rad^2/m^4')

    pyfits.writeto(args.prefix + '-p0.FITS', p0_cube, p0_hdr, overwrite=True)
    pyfits.writeto(args.prefix + '-PA.FITS', PA_cube, PA_hdr, overwrite=True)
    pyfits.writeto(args.prefix + '-RM.FITS', RM_cube, RM_hdr, overwrite=True)
    #pyfits.writeto(args.prefix + '-dRM.FITS', dRM_cube, dRM_hdr, overwrite=True)
    pyfits.writeto(args.prefix + '-dRM.FITS', dRM_cube, RM_hdr, overwrite=True)

    pyfits.writeto(args.prefix + '-p0err.FITS', p0err_cube, p0_hdr, overwrite=True)
    pyfits.writeto(args.prefix + '-PAerr.FITS', PAerr_cube, PA_hdr, overwrite=True)
    pyfits.writeto(args.prefix + '-RMerr.FITS', RMerr_cube, RM_hdr, overwrite=True)
    #pyfits.writeto(args.prefix + '-dRMerr.FITS', dRMerr_cube, dRM_hdr, overwrite=True)
    pyfits.writeto(args.prefix + '-dRMerr.FITS', dRMerr_cube, RM_hdr, overwrite=True)


    pyfits.writeto(args.prefix + '-REDCHI.FITS', REDUCED_CHISQR, fit_hdr, overwrite=True)
    pyfits.writeto(args.prefix + '-CHI.FITS', CHISQR, fit_hdr, overwrite=True)
    pyfits.writeto(args.prefix + '-AIC.FITS', AIC, fit_hdr, overwrite=True)
    pyfits.writeto(args.prefix + '-BIC.FITS', BIC, fit_hdr, overwrite=True)

