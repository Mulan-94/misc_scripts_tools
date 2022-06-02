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
    """

    def compute_in_out_slice(N, N0, R, R0):
        """
        Gotten from https://github.com/ratt-ru/owlcat/blob/2be3c537073ba6ff86df474e204465c8a405cb19/Owlcat/FitsTool.py#L460-L478
        
        given an input axis of size N, and an output axis of size N0, 
        and reference pixels of I and I0 respectively, computes two slice
        objects such that
                A0[slice_out] = A1[slice_in]
        would do the correct assignment (with I mapping to I0, and the 
        overlapping regions transferred)

        N  : size of x-axis of input image
        N0 : size of x-axis of output image
        R  : Half of length of x-axis of input image
        R0  : Half of length of x-axis of output image
        
        For example
        N  : 169
        N0 : 572
        R  : 169//2 = 84
        R0 : 572//2 = 286
        """
        i, j = 0, N     # input slice
        i0 = R0 - R
        j0 = i0 + N     # output slice

        if i0 < 0:
            i = -i0
            i0 = 0
        if j0 > N0:
            j = N - (j0 - N0)
            j0 = N0
        if i >= j or i0 >= j0:
            return None, None
        return slice(i0, j0), slice(i, j)


    with fits.open(imname) as hdul:
        hdu = hdul[0]
        # delete header history
        if "HISTORY" in hdu.header:
            del hdu.header["HISTORY"]
        
        data = hdu.data
        # change zeroes to Nans
        data[np.where(data==0)] = np.nan

        # where not nan is valid else not valid
        data[np.where(~np.isnan(data))] = 1
        data[np.where(np.isnan(data))] = 0
        data = data.astype("int8")

        if xy_dims is not None:
            xy_dims = tuple(xy_dims)

            nx = data.shape[-1]
            ny = data.shape[-2]
            rx, ry = nx//2, ny//2
            rx0, ry0 = xy_dims[0]//2, xy_dims[1]//2
            ndata = np.zeros(xy_dims)
            xout, xin = compute_in_out_slice(nx, xy_dims[0], rx, rx0)
            yout, yin = compute_in_out_slice(ny, xy_dims[1], ry, ry0)

            ndata[..., yout, xout] = data[..., yin, xin]

            wcs = WCS(hdu.header) #, mode="pyfits")
            pixcoord = np.zeros((1, hdu.header["NAXIS"]), float)
            pixcoord[0, 0] = nx//2
            pixcoord[0, 1] = ny//2
            world = wcs.wcs_pix2world(pixcoord, 0)  # get WCS of center pixel
            hdu.header["CRVAL1"] = world[0,0]
            hdu.header["CRVAL2"] = world[0,1]
            hdu.header["CRPIX1"] = rx0 + 1
            hdu.header["CRPIX2"] = ry0 + 1

            hdu.data = ndata
        else:
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
