import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from glob import glob



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
        """
        Read mask and then invert it to comply with numpy mask
        For image mask, the mask is 1 where the image is to be seen
        In numpy, when mask is 1, it means flag out 
        """
        data =  ~np.asarray(data, dtype=bool)
    return dict(hdr=hdr, data=data, wcs=wcs)


def make_mask(imname, xy_dims=None):
    """
    Simple function to make a mask. Set all NaNs to 0 and pixels with values to 1
 
   imname: str
        Name of the input image

     xy_dims: tuple
        A tuple containing the new data sizes that the input dat should be resized to
        This is with the intent to make an image bigger that it already is by padding
        it. They should be in the order x,y.
        Better make it an even number and hopefully the input image is of an even number

    Returns
    -------
    Will automatically be inputimage name + ".mask.fits"
    xy_dims: tuple
        A tuple containing the new data sizes that the input dat should be resized to
        This is with the intent to make an image bigger that it already is by padding
        it. They should be in the order x,y.
        Better make it an even number and hopefully the input image is of an even number
    """
    with fits.open(imname) as hdul:
        hdu = hdul[0]
        data = hdu.data
        data[np.where(~np.isnan(data))] = 1
        data[np.where(np.isnan(data))] = 0
        data = data.astype("int8")
        if xy_dims is not None:
            xy_dims = tuple(xy_dims)
            x_pad, y_pad = (np.array(xy_dims) - np.array(data.shape[-2:]))//2
            data = np.pad(data[0,0], ((x_pad, x_pad), (y_pad, y_pad)))
            # add extra padding if the shapes are not exact same
            if not np.all(np.array(xy_dims) == np.array(data.shape)):
                if data.shape[0] != xy_dims[0]:
                    x_pad = xy_dims[0] - data.shape[0]
                else:
                    x_pad = 0
                if data.shape[1] != xy_dims[1]:
                    y_pad = xy_dims[1] - data.shape[1]
                else:
                    y_pad = 0
                data = np.pad(data, ((0,x_pad), (0, y_pad)))
            data = data.reshape(1,1, *data.shape)
        hdu.data = data
        hdul.writeto(imname+".mask.fits", overwrite=True)
    print(f"Done with {imname}")


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

    image = np.ma.masked_array(data=image, mask=mask)
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
