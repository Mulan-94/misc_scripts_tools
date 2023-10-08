#! /usr/bin/env python

import argparse
import logging
import os
import sys
import time
from multiprocessing import Pool, Array
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import product

import numpy as np
from astropy.io import fits
from lmfit import minimize, Parameters, Minimizer

PATH = set(sys.path)
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir))
if not PATH.issuperset(PROJECT_ROOT):
    sys.path.append(PROJECT_ROOT)

from utils.genutils import dicto, read_npz

# Make syncronised data store and reused data stores global for easier use
global SDS, RDS

SDS = None
RDS = None

def make_syncd_data_stores(xdim, ydim, syncd=True):
    """
    (x|y)dim: (int, int)
        Shape of the resulting image.
        Basically the image dimensions (x,y)
    """
    shape = xdim, ydim

    dtype = "d"
    outputs = {
            "p0_cube": dtype,
            "PA_cube": dtype,
            "RM_cube": dtype,
            "dRM_cube": dtype,
            # "DRM_cube": dtype,
            "p0err_cube": dtype,
            "PAerr_cube": dtype,
            "RMerr_cube": dtype,
            "dRMerr_cube": dtype,
            # "DRMerr_cube": dtype,
            "REDUCED_CHISQR": dtype,
            "CHISQR": dtype,
            "AIC": dtype,
            "BIC": dtype
        }

    if syncd:
        outs = {key: Array(dt, np.zeros(shape, dtype=dt))
                for key, dt in outputs.items()}
    else:
        outs = {key: np.zeros(shape, dtype=dt)
                for key, dt in outputs.items()}
        
    return outs


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

    Ensure that the size of the cube in frequncency matches the
    frequency file
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

    if len(errors) > 0:
        logging.debug(errors)
        sys.exit(">>> Exiting. See log file %s" %LOG_FILENAME)
    
    return


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


def linear_model(param, model=False, eps=False,
    sigmaqu=None, wavelengths=None, fpol=None):

    """
    linear RM model
    """
    p0 = param['p0']
    RM = param['RM']
    PA = param['PA']
    dRM = param['dRM']
    #DRM = param['DRM']
    p = p0 * np.exp(2j * (PA  + RM * wavelengths) )  * np.exp(-2 * dRM**2 * wavelengths**2) #np.sinc(DRM * wavelengths)
    if model:
        return p
    if eps:
        residual = realimag(fpol) -  realimag(p)
        sigma = realimag(sigmaqu)
        return  np.sqrt( residual**2/ sigma**2 )
    else:
        diff =  realimag(fpol) -  realimag(p)
        return  diff



def call_model(p0=0.5, PA=1, PA_min=-np.pi/2.0, PA_max= np.pi/2.0, RM=10, 
              RM_min=-12500, RM_max=12500, dRM=50, dRM_min=0, dRM_max=2500,
              model=None):
    """
    p0: float
        Initial fpol guess
    pa: float
        Plolarisation angle in radians
    pa_min: float
        Minimum allowed polarisatoin angle
    pa_max: float
        Max allowed polarisation angle
    RM: float
        Initial rm gues
    RM_(min|max): float
        Minimum and maximum RMs to look ofr
    """
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

    global SDS, RDS
    
    PIX = {
        "Q": RDS["qdata"][:, x, y],
        "U": RDS["udata"][:, x, y],
        "I": RDS["idata"][:, x, y],
        "noiseq": RDS["noiseq"],
        "noiseu": RDS["noiseu"],
        "noisei": RDS["noisei"],
        "wavelengths": RDS["wavelengths"]
    }
    
    PIX["q"] = PIX["Q"]/PIX["I"]
    PIX["u"] = PIX["U"]/PIX["I"]

    fpol = PIX["q"] + 1j * PIX["u"]
    fpol = np.ma.masked_invalid(fpol)

    PIX = {key: np.ma.masked_array(data=array, mask=fpol.mask).compressed() 
            for key, array in PIX.items()}

    PIX["fpol"] = fpol.compressed()

    PIX = dicto(PIX)

    # getting noise in fractional q and u
    # PIX["sigmaq"]= abs(PIX.q) * ( (PIX.noiseq/PIX.Q)**2 + (PIX.noisei/PIX.I)**2 )**0.5
    # PIX["sigmau"]= abs(PIX.u) * ( (PIX.noiseu/PIX.U)**2 + (PIX.noisei/PIX.I)**2 )**0.5
    sigmaq = abs(PIX.q) * ( (PIX.noiseq/PIX.Q)**2 + (PIX.noisei/PIX.I)**2 )**0.5
    sigmau = abs(PIX.u) * ( (PIX.noiseu/PIX.U)**2 + (PIX.noisei/PIX.I)**2 )**0.5
    PIX["sigmaqu"] = sigmaq + 1j * sigmau

    # polarisation angle
    theta = 0.5 *  np.arctan2(fpol.imag[-1], fpol.real[-1])
    # reading x an y pixels from the rm map and maximum fpols map
    rm = rmdata[x, y]
    fpol0 = pmax[x, y]

    # calling the model with the initial max fpol, intial RM guess and the polarisation angle
    # set_trace()

    fit_residual = call_model(p0=fpol0, PA=theta, RM=rm,
        RM_min=-1000, RM_max=1000, 
        model=partial(
            linear_model, sigmaqu=PIX.sigmaqu,
            wavelengths=PIX.wavelengths, fpol=PIX.fpol)
        )
    SDS["p0_fit"][x,y] = fit_residual.params['p0'].value
    SDS["p0_err"][x,y] = fit_residual.params['p0'].stderr
    SDS["PA_fit"][x,y] = fit_residual.params['PA'].value
    SDS["PA_err"][x,y] = fit_residual.params['PA'].stderr
    SDS["RM_fit"][x,y] = fit_residual.params['RM'].value
    SDS["RM_err"][x,y] = fit_residual.params['RM'].stderr   
    SDS["dRM_fit"][x,y] = fit_residual.params['dRM'].value
    SDS["dRM_err"][x,y] = fit_residual.params['dRM'].stderr
    #DRM_fit = fit_residual.params['DRM'].value
    #DRM_err = fit_residual.params['DRM'].stderr

    SDS["aic"][x,y] = fit_residual.aic
    SDS["bic"][x,y] = fit_residual.bic
    SDS["redsqr"][x,y] = fit_residual.redchi
    SDS["chisqr"][x,y] = fit_residual.chisqr
    return True

def parser():
    parser = argparse.ArgumentParser(description='Performs linear ' 
             'least squares fitting to Q and U image.')
    parser.add_argument('-q', '--qcube', dest='qfits',
         help='Stokes Q cube (fits)')
    parser.add_argument('-u', '--ucube', dest='ufits',
        help='Stokes U cube (fits)')
    parser.add_argument('-i', '--icube', dest='ifits',
        help='Stokes I cube (fits)')
    parser.add_argument("-ldf", "-los-data-file", dest="los_df", type=str,
        help="""A single data file containing one of the LoS' data. This will
        contain among other things: 
        (1) the per channel frequencies,
        (2) I, Q and U noise which are needed for this
        script to function.
        """)
    parser.add_argument('-ncore', '--numpr', dest='numProcessor',
        help='number of cores to use. Default 1.', 
        default=1, type=int)
    parser.add_argument('-mask', '--maskfits', dest='maskfits', 
        help="""A mask image (fits). This package comes with a tool called 
        cleanmask to allow the user to create a mask, for more info
        'cleanmask -h""", default=None)
    parser.add_argument('-rm', '--rm-image', dest='rm_image', 
        help='First guess RM image (fits)', default=None)
    parser.add_argument('-fp', '--fp-max-image', dest='pmax_image', 
        help='First guess FPOL peak image (fits)', default=None)
    parser.add_argument('-rmch', '--remove-channel', dest='remove_channel',
        help='list of channels to remove', 
        type=str, action='append', default=None)
    parser.add_argument('-o', '--prefix', dest='prefix',
        help='This is a prefix for output files.')
    return parser



if __name__=='__main__':

    args = parser().parse_args()

    if args.prefix is None:
        prefix = "qu-fit"
    else:
        prefix = prefix


    LOG_FILENAME = prefix + '.log'
    logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)
    
    ldf = dicto(read_npz(args.los_df))
    
    check_shape(args.qfits, args.ufits, ldf.freqs)

    qdata, qhdr = read_data(args.qfits) # run Q-cube 
    udata, uhdr = read_data(args.ufits) # run U-cube
    idata, ihdr = read_data(args.ifits) # run I-cube

    xdim, ydim =  qhdr["NAXIS1"], qhdr["NAXIS2"]

    if args.rm_image:
        rmdata, hdr = read_data(args.rm_image, freq=False)
    else:
        rmdata = None
    if args.pmax_image:
        pmax, hdr = read_data(args.pmax_image, freq=False)
    else:
        pmax = None


    RDS = {
        "idata": idata,
        "qdata": qdata,
        "udata": udata,
        "wavelengths": ldf.lambda_sq,
        "rmdata": rmdata,
        "pmax": pmax,
        "noiseq": ldf.Q_err,
        "noiseu": ldf.U_err,
        "noisei": ldf.I_err,
        "fpol": qdata + 1j * udata
    }

    

    if args.maskfits:
        xcoord, ycoord = read_mask(args.maskfits)
    
    else:
        xcoord = np.arange(xdim)
        ycoord = np.arange(ydim)


    start1 = time.time()
    
    SDS = make_syncd_data_stores(xdim, ydim, syncd=False)
    for i, (xx, yy) in enumerate(product(xcoord, ycoord)):
        call_fitting(xx, yy)
       
      
    end1 = time.time()
    logging.debug('Total time for multiprocessing is %.6f  seconds. '%(end1-start1))
    
    # now save the fits
    p0_hdr = modify_fits_header(qhdr, ctype='p', unit='ratio')
    PA_hdr = modify_fits_header(qhdr, ctype='PA', unit='radians')
    RM_hdr = modify_fits_header(qhdr, ctype='RM', unit='rad/m/m')
    fit_hdr = modify_fits_header(qhdr, ctype='fit', unit='None')
    #dRM_hdr = modify_fits_header(qhdr, ctype='dRM', unit='rad^2/m^4')

    fits.writeto(prefix + '-p0.FITS', p0_cube, p0_hdr, overwrite=True)
    fits.writeto(prefix + '-PA.FITS', PA_cube, PA_hdr, overwrite=True)
    fits.writeto(prefix + '-RM.FITS', RM_cube, RM_hdr, overwrite=True)
    #fits.writeto(prefix + '-dRM.FITS', dRM_cube, dRM_hdr, overwrite=True)
    fits.writeto(prefix + '-dRM.FITS', dRM_cube, RM_hdr, overwrite=True)

    fits.writeto(prefix + '-p0err.FITS', p0err_cube, p0_hdr, overwrite=True)
    fits.writeto(prefix + '-PAerr.FITS', PAerr_cube, PA_hdr, overwrite=True)
    fits.writeto(prefix + '-RMerr.FITS', RMerr_cube, RM_hdr, overwrite=True)
    #fits.writeto(prefix + '-dRMerr.FITS', dRMerr_cube, dRM_hdr, overwrite=True)
    fits.writeto(prefix + '-dRMerr.FITS', dRMerr_cube, RM_hdr, overwrite=True)


    fits.writeto(prefix + '-REDCHI.FITS', REDUCED_CHISQR, fit_hdr, overwrite=True)
    fits.writeto(prefix + '-CHI.FITS', CHISQR, fit_hdr, overwrite=True)
    fits.writeto(prefix + '-AIC.FITS', AIC, fit_hdr, overwrite=True)
    fits.writeto(prefix + '-BIC.FITS', BIC, fit_hdr, overwrite=True)

