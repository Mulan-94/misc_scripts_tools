import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from glob import glob

from ipdb import set_trace


def read_image_cube(imname, mask=False):
    """
    imname: str
        Name of the input image, or image cube
    mask: bool
        Whether the image input is a mask or not
    
    Returns
    -------
    A dictionary containing header, data, and wcs as
    hdr, data, wcs keys respectively
    """
    with fits.open(imname) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data.squeeze()
        wcs = WCS(hdr).celestial

    if mask:
        data = data.astype("bool")
        data = np.logical_not(data)
        data = data.astype("uint8")
        """
        Read mask and then invert it to comply with numpy mask
        For image mask, the mask is 1 where the image is to be seen
        In numpy, when mask is 1, it means flag out 
        """
    return dict(hdr=hdr, data=data, wcs=wcs)

def get_masked_data(img_name, mask_name):
    """
    Return masked image data
    Input
    -----
    img_name: str
        Input image name
    mask_name: str
        Corresponding mask for that image

    Returns
    -------
    image: np.ndarray
        A masked image data array
    """
    image = read_image_cube(img_name, mask=False)["data"]
    mask = read_image_cube(mask_name, mask=True)["data"]
    image = np.ma.masked_array(data=image, mask=mask, fill_value=np.nan)
    return image


def rotate(inp, theta):
    """
    Input
    -----
    inp: data
        Input data coordinates. i.e coords to be rotated
    theta
        Angle in Degrees

    Output
    ------
    Rotated coordinates by theta
    """
    def rotation_matrix(theta):
        """
        See https://scipython.com/book/chapter-6-numpy/examples/creating-a-rotation-matrix-in-numpy/
        as a reference
        """
        rotmat = [(np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))]
        return np.array(rotmat)
    ads = np.deg2rad(theta)
    res = np.multiply(rotation_matrix(theta), inp)
    return res


def extend_fits_axes(name, new_axes=4):
    """
    Extenend the number of axes ina fits image
    See https://stackoverflow.com/questions/55003367/re-order-cards-in-a-fits-header
    name: str
        Name of the input image
    new_axes: int
        By how much to extend image axes
    """
    with fits.open(name) as hdul:
        print(f"Extending naxis by: {new_axes - hdul[0].data.ndim} dims")
        new_axes = tuple([None for _ in range(new_axes - hdul[0].data.ndim)])
        if new_axes:
            hdul[0].data = hdul[0].data[new_axes]
            hdul[0].header["NAXIS"] = hdul[0].data.ndim
            hdul[0].header.insert("NAXIS2", ("NAXIS3", 1), after=True)
            hdul[0].header.insert("NAXIS3", ("NAXIS4", 1), after=True)
        try:
            hdul[0].verify()
        except VerifyError:
            print("attempting to Fixing verification errors")
            hdul[0].verify('fix')
        hdul.writeto(name, overwrite=True)
        print("Extension done")
