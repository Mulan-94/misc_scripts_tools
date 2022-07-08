# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from matplotlib.colors import LogNorm


from ipdb import set_trace

KEYS = ["data", "hdr", "wcs"]
values_from_keys = lambda ind, keys: tuple(map(ind.get, keys))


def read_data(image):              
    with fits.open(image) as hdu:
        hdr = hdu[0].header
        data = hdu[0].data.squeeze()
        wcs = WCS(hdr).celestial

    return dict(data=data, hdr=hdr, wcs=wcs)
    # return data, hdr, wcs

magfile = 'with_ricks_mask/with-ricks-data-p0-peak-rm.fits'
anglefile = 'with_ricks_mask/with-ricks-data-PA-pangle-at-peak-rm.fits'
fpolfile = 'with_ricks_mask/with-ricks-data-FPOL-at-center-freq.fits'

rotate = np.pi/2.0
step = 5
scale = 5 #15
cutoff = 0.005

# load total intensity
# image = get_pkg_data_filename('I-MFS.fits')
data = read_data('I-MFS.fits')["data"]
wcs = read_data('I-MFS.fits')["wcs"]
mask = read_data('with_ricks_mask/ricks_data.mask.fits')["data"]

mask = ~np.asarray(mask, bool)

data = np.ma.masked_array(data=data, mask=mask)



lstep = 0.5
# np.linspace(0.01, 0.1, 10, endpoint=True)
levels = 2**np.arange(0, data.max()+10, lstep)*0.01
levels = np.ma.masked_greater(levels, data.max()).compressed()



subplot_xargs = {'projection': wcs}


f = read_data(magfile)["data"]
p = read_data(anglefile)["data"]
# p += np.pi/2.0


plt.close("all")

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=subplot_xargs) 
img = ax.imshow(data, cmap='seismic', origin='lower', interpolation='nearest',
                     levels=levels) 
# cbar =  fig.colorbar(img, orientation='horizontal',fraction=0.1)
#cbar.set_label('Total Intensity [Jy/beam]')
# cbar.set_label('Fractional Polarization')
#cbar.set_label('$\sigma$ [rad m$^{-2}$]')
# cbar.set_label('RM [rad m$^{-2}$]')

skip = 10
slicex = slice(None, mask.shape[0], skip)
slicey = slice(None, mask.shape[-1], skip)

mask = mask[slicex, slicey]
p = p[slicex, slicey]

# meshgrid indexing type ij returns MN, while xy returns NM. Default is xy
# mgrid uses slices, while meshgrid uses proper arrays
col, row = np.mgrid[slicex, slicey]

p = np.ma.masked_array(data=p, mask=mask)

scales = np.random.randint(0, 5, p.shape)
u = np.cos(p)*scales
v = np.sin(p)*scales

# ax.contourf(row, col, Z)
ax.quiver(row, col, u, v, angles="xy", headlength=1, headwidth=1, pivot='middle', scale=50);plt.show()

ax.get_transform('fk5')
ra, dec = ax.coords
ra.set_major_formatter('hh:mm:ss')
dec.set_major_formatter('dd:mm:ss')
ax.set_xlabel('J2000 Right Ascension')
ax.set_ylabel('J2000 Declination')

plt.axis([2200, 2770, 2050, 2250])

plt.axis('off')
ax.tick_params(axis='both', which='both', length=0)

plt.savefig('fig-12-1.png')
