import argparse
import numpy as np

from astropy.io import fits
from astropy.coordinates import SkyCoord, FK5
import astropy.units as u

from astropy.wcs import WCS
from regions import (Regions, CircleSkyRegion, RectangleSkyRegion)

from ipdb import set_trace
# from scrappy

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

    world_coord = FK5(ra=ra, dec=dec)
    skies = SkyCoord(world_coord)
    x, y = skies.to_pixel(wcs)
    return int(x), int(y)

#######################################################

def cumulate_regions(fname, data, reg, buffer):
    cx, cy = world_to_pixel_coords(reg.center.ra, reg.center.dec, wcs_ref=fname)
    x_npix, y_npix = data.shape
    w,h = None, None

    if isinstance(reg, CircleSkyRegion):
        w, h = [int(reg.radius.value)]*2
    elif isinstance(reg, RectangleSkyRegion):
        w, h = int(reg.width.value), int(reg.height.value)
    
    if w is not None:
        minx, maxx = cx-w, cx+w
        miny, maxy = cy-h, cy+h

        xs = np.ma.masked_outside(np.arange(minx, maxx), 0, x_npix).compressed()
        ys = np.ma.masked_outside(np.arange(miny, maxy), 0, y_npix).compressed()
        mesh = tuple(np.meshgrid(xs, ys, sparse=True))
        buffer[mesh] = 1
    return buffer




def make_mask(fname, outname, above=None, below=None, regname=None):
    """
    Make simple mask

    fname: str
        Input image for reference
    outname: str
        Output image name for refrence
    above: float
        Values above which to mask
    below: float
        Values below which to mask

    Returns
    -------
    Mask: np.ndarray
        A numpy array containing the mask
    """
    data = fits.getdata(fname).squeeze()
    hdr = fits.getheader(fname)
    del hdr["HISTORY"]

    if above is not None:
        data = np.ma.masked_greater(data, above)
    if below is not None:
        data = np.ma.masked_less(data, below)
    
    mask = data.mask
    mask = mask.astype("uint8")

    if regname is not None:
      
        buff = np.zeros(data.shape)
        regs = Regions.read(regname, format="ds9")

        for reg in regs:
            buff = cumulate_regions(fname, data, reg, buff)



        # x_npix, y_npix = data.shape
        # w,h = None, None

        # reg, = Regions.read(regname, format="ds9")
        # cx, cy = world_to_pixel_coords(reg.center.ra, reg.center.dec, wcs_ref=fname)
        # if isinstance(reg, RectangleSkyRegion):
        #     w, h = [reg.radius.value]*2
        # elif isinstance(reg, CircleSkyRegion):
        #     w, h = reg.width.value, reg.height.value
        
        # if w is not None:
        #     buff = np.zeros(x_npix, y_npix)
        #     minx, maxx = cx-w, cx+w
        #     miny, maxy = cy-h, cy+h

        #     xs = np.ma.masked_outside(np.arange(minx, maxx), 0, x_npix).compressed()
        #     ys = np.ma.masked_outside(np.arange(miny, maxy), 0, y_npix).compressed()
        #     mesh = tuple(np.meshgrid(xs, ys, sparse=True))
        #     buff[mesh] = 1

        if buff.sum() == 0:
            print("Invalid region(s). We're ignoring this")
        else:
            mask = buff * mask
            

    outname += ".mask.fits" if ".fits" not in outname else ""

    print(f"Writing output mask into {outname}")
    fits.writeto(outname, mask, header=hdr, overwrite=True)

    return mask


def parser():
    parse = argparse.ArgumentParser()
    parse.add_argument("iname",
        help="Input image from which to generate mask")
    parse.add_argument("-o", "--outname", dest="oname", 
        default="newmax.mask.fits", metavar="", type=str,
        help="Where to dump the output")
    parse.add_argument("-above", "--above", dest="above", default=None,
        metavar="", type=float,
        help="Valuse above which to mask")
    parse.add_argument("-below", "--below", dest="below", default=None,
        metavar="", type=float,
        help="Values below which to mask")
    parse.add_argument("-rb", "--region-bound", dest="regname",default=None,
        metavar="", type=str,
        help="DS9 region file within which to make our mask")
    return parse


def main():
    opts = parser().parse_args()
    mask = make_mask(opts.iname, opts.oname, above=opts.above, below=opts.below,
        regname=opts.regname)
    print('------- Done ------')

if __name__ == "__main__":
    main()
    """
    python simple-mask.py ../../6-00-polarimetry/i-mfs.fits -o testing -above 4e-3 -rb pica_region-for-mask.reg
    """