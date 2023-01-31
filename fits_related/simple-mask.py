import argparse
import os
import numpy as np
import astropy.units as u
from itertools import zip_longest

from astropy.io import fits
from astropy.coordinates import SkyCoord, FK5
from astropy.wcs import WCS
from scipy.ndimage import rotate

from regions import (Regions, LineSkyRegion, RectangleSkyRegion)

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

def gen_reg_mask(reg, wcs, shape):
    """
    reg: Region
        Region being worked on
    wcs:
        Wcs info for transformation
    shape: tuple
        Shape of the resulting image
    """
    reg = reg.to_pixel(wcs)
    mask = reg.to_mask()
    mask = mask.to_image(shape)
    return mask.astype("uint8")



def cumulate_regions(fname, data, reg, fill_val=1, default_arr="zeros"):
    """
    fname: str
        Name of the input image. To be used for wcs reference
    data: np.ndarray
        Numpy array containining image data
    reg: :obj:`Regions`
        Region being currently processed
    fill_value: int
        What to the chosen valid region area with. Default is 1. Meaning that
        within the bounds of this region, the value there will be 1.
    default_arr: str
        What the buffering array will be composed of. Intention is to be:
        'zeros' of 'ones' and nothing else! 
    """
    buffer = getattr(np, default_arr)(data.shape, dtype="uint8")

    if isinstance(reg, LineSkyRegion):
        print("We don't do lines here :), skipping")
        return buffer

    if isinstance(reg, RectangleSkyRegion):
        # using a custom masker for rectangular regions because it gives
        # better results!!! Do NOT change!
        cx, cy = world_to_pixel_coords(reg.center.ra, reg.center.dec,
            wcs_ref=fname)
        x_npix, y_npix = data.shape
        
        w, h = np.ceil(np.array((reg.width.value, reg.height.value))//2).astype(int)
        
        minx, maxx = cx-w, cx+w+1
        miny, maxy = cy-h, cy+h+1

        xs = np.ma.masked_outside(np.arange(minx, maxx), 0, x_npix).compressed()
        ys = np.ma.masked_outside(np.arange(miny, maxy), 0, y_npix).compressed()

        # notice how we pass y before x, returns x before y
        # Leave it this way!!!!
        mesh = tuple(np.meshgrid(ys, xs))
        buffer[mesh] = fill_val
        
        if hasattr(reg, "angle"):
            pivot = cx, cy
            buffer = merotate2(buffer, -reg.angle.value, pivot=pivot)
    else:
        wcs = get_wcs(fname)
        buffer = gen_reg_mask(reg, wcs, data.shape)

    return buffer

def make_mask(fname, outname, above=None, below=None, regname=None, ex_regname=None):
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

    # cater for empty masks

    if above is not None:
        data = np.ma.masked_greater(data, above)
    if below is not None:
        data = np.ma.masked_less(data, below)
    if None not in [above, below]:
        data = np.ma.masked_array(data=data, mask=np.ma.make_mask(data))

    
    mask = data.mask
    mask = mask.astype("uint8")

    if regname is not None:
      
        finale = np.zeros(data.shape, dtype="uint8")
        regs = Regions.read(regname, format="ds9")

        for reg in regs:
            # finale = finale+ cumulate_regions(fname, data, reg)
            finale = np.bitwise_or(finale, cumulate_regions(fname, data, reg))

        if finale.sum() == 0:
            print("Invalid region(s). We're ignoring this")
        else:
            mask = finale * mask
        # plt.imshow(finale + mask, origin="lower")
        # ylim, xlim = np.where(mask+finale >= 1)
        # plt.xlim(np.min(xlim), np.max(xlim))
        # plt.ylim(np.min(ylim), np.max(ylim))
        # plt.savefig(outname+"-overlay.png")

    if ex_regname is not None:
        finale = np.zeros(data.shape, dtype="uint8")
        regs = Regions.read(ex_regname, format="ds9")

        for reg in regs:
            finale = np.bitwise_or(finale, cumulate_regions(fname, data, reg))

        # invert this mask to contain 1s where
        finale = np.logical_not(finale).astype("uint8")

        if finale.sum() == 0:
            print("Invalid region(s). We're ignoring this")
        else:
            mask = finale * mask

    plt.imshow(mask, origin="lower")
    ylim, xlim = np.where(mask >= 1)
    plt.xlim(np.min(xlim), np.max(xlim))
    plt.ylim(np.min(ylim), np.max(ylim))
    plt.savefig(outname+"-overlay-final.png")

    outname += ".mask.fits" if ".fits" not in outname else ""

    print(f"Writing output mask into {outname}")
    # del hdr["BUNIT"]
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
        help="Include everything above this value. ie. > above. Aka the lower limit")
    parse.add_argument("-below", "--below", dest="below", default=None,
        metavar="", type=float,
        help="Include everything below this value. i.e < below. Aka the upper limit")
    parse.add_argument("-rb", "--region-bound", dest="regname", default=[],
        metavar="", type=str, nargs="+",
        help="DS9 region file(s) within which to make our mask")
    parse.add_argument("-er", "--exclude-region", dest="ex_regname", default=None,
        metavar="", type=str,
        help="DS9 region file(s) containing the regions that should be excluded")
    return parse


def main():
    opts = parser().parse_args()

    for i,( oname, regname) in enumerate(zip_longest(opts.oname, opts.regname)):
        if oname is None:
            oname = f"output-mask-{i}.fits"
        make_mask(opts.iname, oname, above=opts.above, below=opts.below,
            regname=regname, ex_regname=opts.ex_regname)
    print('------- Done ------')

if __name__ == "__main__":
    main()
    """
    python simple-mask.py ../../6-00-polarimetry/i-mfs.fits -o testing -above 4e-3 -rb pica_region-for-mask.reg

    python simple-mask.py ../6-00-polarimetry/i-mfs.fits -o east-lobe.fits -above 4e-3 -rb important_regions/lobes/e-lobe.reg

    python simple-mask.py ../6-00-polarimetry/i-mfs.fits -o west-lobe.fits -above 4e-3 -rb important_regions/lobes/w-lobe.reg

    # with region exclusions
    python simple-mask.py ../6-00-polarimetry/i-mfs.fits -o test.fits -above 4e-3 -rb important_regions/lobes/2023-lobes.reg -er important_regions/hotspots/core.reg

    """