# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

i_image = "i-mfs.fits"
q_image = "q-mfs.fits"
u_image = "u-mfs.fits"


# polarization angle starts from 0 North, apply this offset
ANGLE_OFFSET = 90


def polangle(q, u):
    return 0.5 * np.ma.arctan2(u, q)


def read_image(imname):
    data = fits.getdata(imname).squeeze()
    return data


idata = read_image(i_image)
idata = np.ma.masked_where(idata<0.01, read_image(i_image))
mask_data = idata.mask
qdata = np.ma.masked_array(data=read_image(q_image), mask=mask_data)
udata = np.ma.masked_array(data=read_image(u_image), mask=mask_data)
angle = np.rad2deg(polangle(qdata, udata))



wiggle = 20
skip = 20
scales = 0.05


ydim, xdim = np.where(mask_data == False)
slicex = slice(None, angle.shape[0], skip)
slicey = slice(None, angle.shape[-1], skip)
col, row = np.mgrid[slicex, slicey]
xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle


pangle = angle[::skip, ::skip]
u = v = scales * np.ones_like(pangle)


specs = dict(pivot='tail', headlength=0, width=0.0012, scale=5, headwidth=1)

plt.imshow(
    idata, origin="lower",
    cmap="coolwarm", aspect="equal", vmin=0.04, vmax=1)
plt.quiver(row, col, u, v, angles=pangle+ANGLE_OFFSET, **specs)
# plt.scatter(row.flatten(), col.flatten(), marker="+", s=40, alpha=0.5, c="orange")


plt.xlim(*xlims)
plt.ylim(*ylims)
plt.savefig("VEX2-b-90.svg", dpi=400)
