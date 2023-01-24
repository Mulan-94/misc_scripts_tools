import argparse
import os
import numpy as np
import astropy.units as u
from itertools import zip_longest

from astropy.io import fits
from astropy.coordinates import SkyCoord, FK5
from astropy.wcs import WCS
from scipy.ndimage import rotate

from regions import (Regions, CircleSkyRegion, RectangleSkyRegion)
# CircleAnnulusSkyRegion, EllipseSkyRegion, EllipseAnnulusSkyRegion
# LineSkyRegion, PolygonSkyRegion, PointSkyRegion, 
# RectangleAnnulusSkyRegion, RectangleAnnulusSkyRegion


import matplotlib.pyplot as plt

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
    return int(np.round(x,0)), int(np.round(y,0))

#######################################################


def merotate2(img, angle, pivot):
    """
    From:
    https://stackoverflow.com/questions/25458442/rotate-a-2d-image-around-specified-origin-in-python
    """
    padx = [img.shape[1] - pivot[0], pivot[0]]
    pady = [img.shape[0] - pivot[1], pivot[1]]
    paded = np.pad(img, [pady, padx], 'constant')
    rotated = rotate(paded, angle, reshape=False, output="uint8")
    return rotated[pady[0]:-pady[1], padx[0]:-padx[1]]


def cumulate_regions(fname, data, reg):
    buffer = np.zeros(data.shape)

    # todo: Add support for other types of regions
    if not hasattr(reg, "center"):
        print("This region has no center, skipping")
        return buffer

    cx, cy = world_to_pixel_coords(reg.center.ra, reg.center.dec, wcs_ref=fname)
    x_npix, y_npix = data.shape
    w,h = None, None
    
    if isinstance(reg, CircleSkyRegion):
        w, h = [int(reg.radius.value)]*2
    elif isinstance(reg, RectangleSkyRegion):
        w, h = int(reg.width.value)//2, int(reg.height.value)//2
    
    if w is not None:
        minx, maxx = cx-w, cx+w
        miny, maxy = cy-h, cy+h

        xs = np.ma.masked_outside(np.arange(minx, maxx), 0, x_npix).compressed()
        ys = np.ma.masked_outside(np.arange(miny, maxy), 0, y_npix).compressed()

        # notice how we pass y before x, returns x before y
        # Leave it this way!!!!
        mesh = tuple(np.meshgrid(ys, xs))
        buffer[mesh] = 1
        
        if hasattr(reg, "angle"):
            pivot = cx, cy
            buffer = merotate2(buffer, -reg.angle.value, pivot=pivot)        
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
      
        finale = np.zeros(data.shape)
        regs = Regions.read(regname, format="ds9")

        for reg in regs:
            finale += cumulate_regions(fname, data, reg)


        if finale.sum() == 0:
            print("Invalid region(s). We're ignoring this")
        else:
            mask = finale * mask
        plt.imshow(finale + mask, origin="lower")
        ylim, xlim = np.where(mask+finale == 1)
        plt.xlim(np.min(xlim), np.max(xlim))
        plt.ylim(np.min(ylim), np.max(ylim))
        plt.savefig(outname+"-overlay.png")

    outname += ".mask.fits" if ".fits" not in outname else ""

    print(f"Writing output mask into {outname}")
    fits.writeto(outname, mask, header=hdr, overwrite=True)

    return


def parser():
    parse = argparse.ArgumentParser()
    parse.add_argument("iname",
        help="Input image from which to generate mask")
    parse.add_argument("-o", "--outname", dest="oname", 
        default=[], metavar="", type=str, nargs="+",
        help="Where to dump the output(s). We can iterate over multiple reg files. See '-rb' ")
    parse.add_argument("-above", "--above", dest="above", default=None,
        metavar="", type=float,
        help="Valuse above which to mask")
    parse.add_argument("-below", "--below", dest="below", default=None,
        metavar="", type=float,
        help="Values below which to mask")
    parse.add_argument("-rb", "--region-bound", dest="regname", default=[],
        metavar="", type=str, nargs="+",
        help="DS9 region file(s) within which to make our mask")
    return parse


def main():
    opts = parser().parse_args()

    for i,( oname, regname) in enumerate(zip_longest(opts.oname, opts.regname)):
        if oname is None:
            oname = f"output-mask-{i}.fits"
        make_mask(opts.iname, oname, above=opts.above, below=opts.below,
            regname=regname)
    print('------- Done ------')

if __name__ == "__main__":
    main()
    """
    python simple-mask.py ../../6-00-polarimetry/i-mfs.fits -o testing -above 4e-3 -rb pica_region-for-mask.reg

    python simple-mask.py ../6-00-polarimetry/i-mfs.fits -o east-lobe.fits -above 4e-3 -rb important_regions/lobes/e-lobe.reg

    python simple-mask.py ../6-00-polarimetry/i-mfs.fits -o west-lobe.fits -above 4e-3 -rb important_regions/lobes/w-lobe.reg

    """