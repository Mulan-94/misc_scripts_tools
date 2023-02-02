import os

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord, FK5
import astropy.units as u
from astropy.wcs import WCS
from collections import namedtuple
from copy import deepcopy
from itertools import product
from regions import Regions, PixCoord, CirclePixelRegion, RectanglePixelRegion
from scipy.ndimage import label, find_objects


from scrap.scraplog import snitch
from utils.mathutils import (are_all_nan, are_all_zeroes, is_infinite, rms,
    signal_to_noise, snr_is_above_threshold, are_all_masked)

from utils.rmmath import (lambda_sq, frac_polzn, linear_polzn, linear_polzn_error,
    polarised_snr, frac_polzn_error, polzn_angle, polzn_angle_error)


from ipdb import set_trace

def read_fits_image(name: str, mask=None, numpy=True):
    """
    Returns an image item that contains the follwoing:
    1. the image data (with mask applied if it has been provided)
    2. The mask data itself (if mask was provided)
    3. The image header information. Without the history.

    name:
        Input image name
    mask:
        Input mask name 
    numpy: bool
        Only active if a mask provided. This specifies that a numpy
        mask should be retuned with the image (i.e valid regions are ones).
        If set to False, an image mask will be returned instead. 
        (i.e valid regions are zeroes)
    
    Access these using the format: returned object.name|mask|header|freq
    """
    image = namedtuple("image", "")
    
    data = fits.getdata(name).squeeze()
    image.header = fits.getheader(name)
    image.freq = image.header["CRVAL3"]
    image.chan_width = image.header["CDELT3"]

    if "HISTORY" in image.header:
        del image.header["HISTORY"]

    try:
        # testing if this value is an integer to trigger error
        int(os.path.basename(name).split("-")[-2])
        image.stokes = os.path.basename(name).split("-")[-3]
    except ValueError:
        image.stokes = os.path.basename(name).split("-")[-2]

    if mask:
        mask_data = read_fits_mask(mask)
        if numpy:
            image.mask = mask_data.numpy
            data = np.ma.masked_array(data=data, mask=mask_data.numpy)
        else:
            image.mask = mask_data.image
            data *= mask_data.image
    image.data = data
    return image


def read_fits_mask(name: str):
    """
    Return a mask for masked arrays
    Note that a FITS mask has ones for pixels that SHOULD NOT be masked
    A numpy mask has ones for pixels that SHOULD be masked
    """
    masks = namedtuple("mask", ["numpy", "image"])
    data = fits.getdata(name).squeeze()
    mask = masks(data, ~data)
    return mask


##################################
# Regions stuff

def world_to_pixel_coords(ra, dec, wcs_ref):
    """
    Convert world coordinates to pixel coordinates.
    The assumed reference is FK5
    ra: float
        Right ascension in degrees
    dec: float
        Declination in degrees
    wcs_ref:
        Image to use for WCS information

    Returns
    -------
        x and y pixel information
    """
    if isinstance(wcs_ref, str):
        wcs = get_wcs(wcs_ref)
    else:
        wcs = wcs_ref
    world_coord = FK5(ra=ra*u.deg, dec=dec*u.deg)
    skies = SkyCoord(world_coord)
    x, y = skies.to_pixel(wcs)
    return int(x), int(y)


def get_wcs(name:str, pixels=False):
    """
    Return the image wcs object or the image dimensions

    name:
        Image from which to get the wcs stuff
    pixels: bool
        If specified, will return the image dimensions in pixels
    """
    wcs = WCS(fits.getheader(name))
    if wcs.naxis > 2:
        dropped = wcs.naxis - 2
        # remove extra and superflous axes. These become problematic
        for _ in range(dropped):
                wcs = wcs.dropaxis(-1)
  
    if pixels:
        # get the image dimensions
        return wcs.pixel_shape
    else:
        return wcs


def write_regions(name: str, regs: tuple, overwrite=True, reg_id=False):
    """
    reg_id: bool
        Whether to add the region id to the written file
    """
    header = [
        "# Region file format: DS9 CARTA 2.0.0\n",
        ('global color=#2EE6D6 dashlist=8 3 width=2 ' +
        'font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 ' +
        'edit=1 move=1 delete=1 include=1 source=1\n'),
        "FK5\n"
        ]

    if ".reg" not in name:
        name += ".reg"

    if not os.path.isfile(name) or (os.path.isfile(name) and overwrite):
        outs = []
        with open(name, "w") as fil:
            for i, p in enumerate(regs, 1):
                if reg_id:
                    sub = p.split("#")
                    outs.append(sub[0] + f" # {i},los text={{reg_{i}}} {sub[1].strip()}\n")
                else:
                    outs.append(p+"\n")
            fil.writelines(header+outs)

        snitch.warning(f"Regions file written to: {name}."+
            " Overwriting If this file existed before")
    else:
        snitch.warning(f"Refusing to overwrite {name}")

    # return name without the extension. Want to modify in future
    return os.path.splitext(name)[0]


def sorta(inp):
    val, ind = np.unique(inp, return_index=True)
    val = val[np.argsort(ind)]
    return val

def make_default_regions(bounds, size, wcs_ref, reg_file,
    pixels=False, overwrite=True):
    """
    Create some default regions given an image
    Algorithm
    1. Get the x and y ranges for which to generate regions in pixels
        - If given in WCS coordinates, convert to pixels first
        - If given in pixels, leave it this way
    2. Determine all the x coords if less than imdim
    3. Determine all the y coords as long as they're less than imdim
    4. For each y
        - Create all the corressponding x regions in pixels
    5. Convert regions from pixels to wcs format
    6. write them out

    Inputs
    ------
    bounds: tuple | str
        minx, maxx, miny, maxy
        (min|max)(x|y): float
            Minimum|maximum x and y coordinate
        If its a string, assume its a mask name and read the mask properly
    size: int
        The desired region size
    image:
        The image from where to get the WCS reference
    wcs_ref: WCS
        A wcs object to be used to converted pixel to world coordinates
        and back
    reg_file:
        How to call the file where the regions will be stored
    pixels:
        Whether the provided coordinates are in pixel form or world coordinates
        Automatically assumed to be world coordinates
    """

    def is_valid_coord(size, coord, max_dim):
        """We assume that the MINIMUM POSIBLE  DIMENSION in
        both x and y is 0
        """
        this_max = coord + (size)/2
        this_min = coord - (size)/2
        if this_max>max_dim or this_min<0:
            return False
        else:
            return True
    
    
    wcs = get_wcs(wcs_ref)

    snitch.info("Making default regions")    

    if isinstance(bounds, str):
        # reading this as an image because I want the valid areas
        mask = read_fits_image(bounds).data
       
        # label the different regions for tag
        label(mask, output=mask)

        mycoords, mxcoords = np.where(mask>0)
        mcords = list(zip(mycoords, mxcoords))
        minx, maxx = mxcoords.min(), mxcoords.max()
        miny, maxy = mycoords.min(), mycoords.max()
        maxx_dim, maxy_dim = maxx, maxy
    else:
        # get the maximum x and y image dimensions possible
        minx, maxx, miny, maxy = bounds
        maxx_dim, maxy_dim = get_wcs(wcs_ref, pixels=True)
        minx, miny = world_to_pixel_coords(minx, miny, wcs)
        maxx, maxy = world_to_pixel_coords(maxx, maxy, wcs)

    xcoords = [_ for _ in range(minx, maxx, size*2) 
                if is_valid_coord(size, _, maxx_dim)]
    ycoords = [_ for _ in range(miny, maxy, size*2) 
                if is_valid_coord(size, _, maxy_dim)]
    
    cords = product(ycoords, xcoords)

    world_coords = []

    for y, x in cords:
        if isinstance(bounds, str):
            if (y,x) not in mcords:
                continue
            tag = f"g{mask[y,x]}"
        else:
            tag = "g1"
            
        sky = CirclePixelRegion(PixCoord(x, y), radius=size).to_sky(wcs)

        world_coords.append(
            "circle({:.6f}, {:.6f}, {:.6f}\") # tag={{{}}}".format(
        sky.center.ra.deg, sky.center.dec.deg, sky.radius.arcsec, tag)
        )
    
    snitch.info(f"{len(world_coords)} regions found")
    # Write the regions out
    reg_file += f"-size-{size}-default"
    reg_file = write_regions(reg_file+".reg", world_coords, overwrite=overwrite,
        reg_id=True)

    return reg_file


def make_noise_region_file(name=None, reg_xy=None):
    """
    todo: autogenerate of source noise region
    Write out a region file containing the noise region
    reg_xy: str
        the x and y coordinates and the region size of the noise region
        Should be specified in the format: (x, y, size)
    """
    # using the pictor A default region coordinate HERE!

    header = [
        "# Region file format: DS9 CARTA 2.0.0\n",
        ('global color=#2EE6D6 dashlist=8 3 width=2 ' +
        'font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 ' +
        'edit=1 move=1 delete=1 include=1 source=1\n'),
        "FK5\n"
        ]

    if reg_xy is None:
        reg_xy = "80.041875, -45.713452, 13.0000"
        snitch.warning("The noise region in use is for Pictor A.")
        snitch.warning(f"That is in WCS: {reg_xy}")
        snitch.warning(f"If this is not the intended region coordinate, " +
            "please specify this value in the function 'make_noise_region_file'")
    
    if name is None:
        name = os.path.join(os.path.dirname(name), "noise-region.reg")
    else:
        name += ".reg"

    with open(name, "w") as fil:
        fil.writelines([f"{p}\n" for p in header+[f"circle({reg_xy}\")"]])
        snitch.info(f"Noise region written: {name}")
    return os.path.splitext(name)[0]



##################################
# Regions Stuff
##################################

def read_regions_as_pixels(reg_file, wcs=None):

    if ".reg" not in reg_file:
        reg_file += ".reg"

    # convert to pixel values otherwise we don't get to_mask method
    regs = Regions.read(reg_file, format="ds9").regions

    if wcs is None:
        try:
            getattr(regs[0], "to_mask")
        except:
            snitch.error("Please provide a wcs reference!")

    # if no wcs, we assume that these regions are already in pixel format
    if wcs:
        for _, reg in enumerate(regs):
            regs[_] = regs[_].to_pixel(wcs)
    return regs


def parse_valid_region_candidates(image, reg_file, noise_file, threshold, 
    noise=None, overwrite=True):
    """
    Only store regions which are above the specified threshold
    Algorithm
    1. Read the region file
    2. Read the off source noise region file
    - Get the image data
    - Get the image noise
    3. Calculate the RMS noise for the noise region
    4. For each region:
        - read the data for that region
        - Determine whether the signal meets the threshold
        - If it does add it to the valid list otherwise ignore region
    5. write out the regions that met the threshold to some file
    """
    wcs = get_wcs(image)
    regs = read_regions_as_pixels(reg_file, wcs=wcs)
    noise_reg, = read_regions_as_pixels(noise_file, wcs=wcs)
    image_data = read_fits_image(image).data
    
    # use the provided noise if available else generate the noise
    noise = image_noise(noise_reg, image_data) if noise is None else noise

    if is_infinite(noise):
        valids = None
        return

    valids = []
    for reg in regs:
        signal = region_flux_jybm(reg, image_data)
        if region_is_above_thresh(signal, noise, threshold=threshold):
            sky = reg.to_sky(wcs)
            valids.append(
            "circle({:.6f}, {:.6f}, {:.6f}\") # tag={{{}}}".format(
                sky.center.ra.deg, sky.center.dec.deg, sky.radius.arcsec,
                ','.join(reg.meta["tag"])))

    if len(valids) > 0:
        snitch.info(f"Found {len(valids)}/{len(regs)} valid region candidates")
        valid_name = os.path.splitext(reg_file)[0] + "-valid-candidates"
        valid_name = write_regions(valid_name, valids, overwrite=overwrite, reg_id=True)
    else:
        valid_name = None
        snitch.warning("No valid regions were found")
    
    # Return the file where the candidates  are stored
    return valid_name
    

def image_noise(noise_reg, data):
    noise_data = read_region_data(noise_reg, data)

    if are_all_masked(noise_data) or are_all_nan(noise_data) or are_all_zeroes(noise_data):
        noise= np.nan
    else:
        noise = rms(noise_data)
    return noise


def read_region_data(reg, data):
    """
    Returns a data array containing only data from specified region

    get the weighted cut
    # see: https://astropy-regions.readthedocs.io/en/stable/masks.html?highlight=cutout#making-image-cutouts-and-multiplying-the-region-mask

    Algorithm
    1. Convert region to mask
    2. Multiply the full image data with the mask
    3. return this data

    reg: :obj:`regions.Region`
        Region of interest
    data: :obj:`np.ndarray`
        Data array of a given fits file
    """
    reg_mask = reg.to_mask()
    weighted_data_cut = reg_mask.multiply(data)
    weighted_data_cut = np.ma.masked_equal(weighted_data_cut,0)
    return weighted_data_cut


def region_flux_jybm(reg, data):
    """
    Check if all values are 
    - nans
    - zeros
    - if the noise is infinite
    """
    intense_cut = read_region_data(reg, data)

    if (are_all_nan(intense_cut) \
        or are_all_zeroes(intense_cut)):
        # skip all the nonsence if all the data is Nan
        snitch.debug(f"Skipping region:{reg.meta['text']} because NaN/Zeroes/inf ")
        flux_jybm = None
    else:
        #Using the center pixel
        # cpixx, cpixy = np.ceil(np.array(intense_cut.shape)/2).astype(int)
        # flux_jybm = intense_cut[cpixx, cpixy]
        ban = intense_cut.compressed()
        flux_jybm = ban[ban.size//2]
    return flux_jybm


def region_is_above_thresh(signal=None, noise=None, snr=None, threshold=1):
    """
    1. Get the data for this region
    2. Calculate the signal value
    3. Compare signal value to noise value
    4. Return if it meets the threshold
    
    """
    if snr is None:
        snr = np.abs(signal_to_noise(signal, noise))
    return snr_is_above_threshold(snr, threshold)



##################################
# CASA Related
##################################


# from casatasks import imstat
def get_box_dims(reg):
    """
    *** For use in imsats ***
    --------------------------
    Create a valid region entry. 
    
    reg: :obj:`regions.Region`
        A region object
    """
    # blc: bottom left corner, trc: top right corner
    # blc_x, trc_x, blc_y, trc_y
    box = ",".join(
        [str(getattr(reg.bounding_box, x)) 
            for x in "ixmin iymin ixmax  iymax".split()])
    return box


def get_imstat_box_dims(reg):
    """
    *** For use in imsats ***
    --------------------------
    Get box dimensions for CASA's imstat

    blc: bottom left corner, trc: top right corner
    This function is to deal with the off by -1 issue with casa's
    imstats TRC argument. The problem was that while using imstats and doing
    a box selection, setting the TRC value to eg (100, 100) would lead to the 
    selection of pixels upto (101, 101). So for example, I want pixels btwn
    blc(0,90) and trc(2, 92), I would expect to get back stats for 4=(2x2) pixels
    however, I get back stats for 9 pixels because 9=(3x3)

    I therefore reduce the trc values by 1 and return the values in the form:
        blc_x, blc_y, trc_x, trc_y

    reg: :obj:`regions.Region`
        A region object
    """
    box = []
    for x in "ixmin iymin ixmax  iymax".split():
        val = getattr(reg.bounding_box, x)
        if "max" in x:
            val -= 1
        box.append(str(val))
    return ",".join(box)



##################################
# Beam stuff
##################################


def get_flux(header, flux_sum):
    """
    Calculate region flux in Jansky ie from Jy/beam -> Jy
    header:
        FITS image header
    flux_sum:
        Sum of all pixels in a given region
    """
    bmaj = np.abs(header["BMAJ"])
    bmin = np.abs(header["BMIN"])
    cdelt1 = header["CDELT1"]
    cdelt2 = header["CDELT2"]
    # from definitions of FWHM
    gfactor = 2.0 * np.sqrt(2.0 * np.log(2.0))
    beam_area = np.abs((2 * np.pi * (bmaj/cdelt1) * (bmin/cdelt2)) / gfactor**2)
    if flux_sum>0:
        flux = flux_sum / beam_area

    else:
        flux = None
    return flux