import numpy as np
from astropy.io import fits

"""
Generate RM maps from Brentjens RM-syntehsis

After running brentjens rmsynthesis:

rmsynthesis inputs/q-image-cube.fits inputs/u-image-cube.fits frequencies.txt --dphi 1 --low '-200' --high 200 -m 100 -o costa-rm

"""

rmfile = "rmsf.txt"
imname = "p-rmcube-dirty.fits"
# mask_name = "../ficube-out/tnp_mask.fits"
mask_name = "../../../../masks/true_mask.fits"

# should be the same as the number of output planes of this cube
phi_range = np.loadtxt(rmfile)[:, 0]

hdr = fits.getheader(imname)
del_list = []
for key in hdr.keys():
    if "BMIN" in key or "BMAJ" in key or "BPA" in key or "4" in key:
        del_list.append(key)
for _ in del_list:
    del hdr[_]

del hdr["POL"]
hdr["BUNIT"] = "RAD/M^2"


mask = np.squeeze(fits.getdata(mask_name))
mask = mask.astype("float")
mask =  np.ma.masked_less(mask, 1)
mask = mask.filled(np.nan)

data = np.squeeze(fits.getdata(imname))
data *= mask

arg_max = np.nargmax(data, axis=0)
rm_vals = arg_max - phi_range.max()

fits.writeto("rm-map.fits", rm_vals, header=hdr, overwrite=True)