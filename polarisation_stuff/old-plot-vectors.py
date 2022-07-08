# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm



from ipdb import set_trace

# import matplotlib
# matplotlib.rcParams.update({'font.size': 18, 'font.family':'serif'})# 

def read_data(image):              
    with fits.open(image) as hdu:
        hdr = hdu[0].header
        data = hdu[0].data.squeeze()
        wcs = WCS(hdr).celestial

    # return dict(data=data, hdr=hdr, wcs=wcs)
    return data, hdr, wcs

"""
def plot_vectors(pa_image, p0_image, contour_image=None, rotate=np.pi/2.0, step=15,
	   scale=100, p0_cutoff=0.01, levels=[ 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]):

    paimage = get_pkg_data_filename(pa_image)
    padata, hdr, wcs = read_data(paimage)

    p0image = get_pkg_data_filename(p0_image)
    p0data, hdr, wcs = read_data(p0image)

    subplot_xargs = {'projection':wcs}
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=subplot_xargs)
            
    if rotate:
        padata += rotate

    ax.get_transform('fk5')
    ra, dec = ax.coords
    ra.set_major_formatter('hh:mm:ss')
    dec.set_major_formatter('dd:mm:ss')
    plt.xlabel('J2000 Right Ascension')
    plt.ylabel('J2000 Declination')
    plt.axis([500, 1891, 1200, 2325])
    if contour_image:
        contour_img = get_pkg_data_filename(contour_image)
        contourdata, chdr, cwcs = read_data(contour_img)
        ax.contour(contourdata, levels, colors='black', origin='lower',
             linewidths=0.5)

    
    (ys, xs) = np.where(p0data > 0)     
    #print(ys, xs)
    print(len(ys))
    ys = [ys[i] for i in range(0, len(ys), step)]
    xs = [xs[i] for i in range(0, len(xs), step)]
    #(ys, xs) = p0data.shape
    #linelist=[]
    for y in ys:
        for x in xs:
            #if p0data[y,x] > p0_cutoff:
            r = 1 * scale * 0.5  # scale fractional polarization
            #a = p[y, x] * np.pi/180 # convert to radians
            a =  padata[y, x]
            x1 = x + r*np.sin(a) # x 5+ f * sin(a), x- f * sin(a)
            y1 = y - r*np.cos(a) # 
            x2 = x - r*np.sin(a)
            y2 = y + r*np.cos(a)
                 #x_world, y_world = fig.pixel2world([x1,x2],[y1,y2])
                 #line = np.array([x_world,y_world])
                 #line = np.array([[x1, x2], [y1,y2]])
                 #linelist.append(line)
            plt.plot([x1, x2], [y1,y2], 'k-', lw=1)
    #plt.tight_layout()
    #plt.savefig('PNGS/CYG-FIT-PA-WEST.PDF')
    plt.show()

"""

  

magfile = 'turbo-p0-peak-rm.fits'
anglefile = 'turbo-PA-pangle-at-peak-rm.fits'

rotate = np.pi/2.0
step = 5
scale = 5 #15
cutoff = 0.005

# load total intensity
image = get_pkg_data_filename('clean-avg-taper15-postmask/38-avg50-taper15-postmask-MFS-I-image.fits')
data, hdr, wcs = read_data(image)


#maskdata2 = fits.open('CYG-S-I-MASK.FITS')[0].data[0, 0,...] # 
#maskdata2[maskdata2==0] = float(np.nan)
#data = data * maskdata2

#maskdata = fits.open('CYG-0.75-SHI-I-MASK.FITS')[0].data[0, 0,...] # 
#maskdata[maskdata==0] = float(np.nan)

levels = np.array([0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
#levels = np.array([0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
subplot_xargs = {'projection':wcs}


# fp = fits.open(magfile)
# pa = fits.open(anglefile)
# f= fp[0].data#[0,0]
# p= pa[0].data#[0,0]
# p += np.pi/2.0


f, _, _ = read_data(magfile)
p, _, _ = read_data(anglefile)
p += np.pi/2.0

#p = p * maskdata
#f[f >=1] = float(np.nan)


levels_f = np.array([ 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9])

from matplotlib.colors import LogNorm

plt.close("all")

fig, ax = plt.subplots(figsize=(16, 8), subplot_kw=subplot_xargs) 
img = ax.imshow(f, cmap='jet', origin='lower', interpolation='nearest',  #gist_ncar
                     norm=LogNorm(vmin=0.001, vmax=0.8)) #vmin=0, vmax=800) #
cbar =  fig.colorbar(img, orientation='horizontal',fraction=0.1)
#cbar.set_label('Total Intensity [Jy/beam]')
cbar.set_label('Fractional Polarization')
#cbar.set_label('$\sigma$ [rad m$^{-2}$]')
#cbar.set_label('RM [rad m$^{-2}$]')

ax.get_transform('fk5')
ra, dec = ax.coords
ra.set_major_formatter('hh:mm:ss')
dec.set_major_formatter('dd:mm:ss')
plt.xlabel('J2000 Right Ascension')
plt.ylabel('J2000 Declination')
#plt.axis([500, 1700, 1350, 2200]) # E-lobe 31 July
#plt.axis([2400, 3800, 1950, 2950]) # W-lobe 31 July
plt.axis([2200, 2770, 2050, 2250])

plt.axis('off')
#plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off')
ax.tick_params(axis='both', which='both', length=0)


(ys, xs) = f.shape
linelist=[]

set_trace()
for y in range(0, ys, step):
    for x in range(0, xs, step):
        if f[y,x] > cutoff:
            r = 1 * scale * 0.5  # scale fractional polarization
            #r = f[y, x] * scale
            #a = p[y, x] * np.pi/180 # convert to radians
            a =  p[y, x]
            x1 = x + r*np.sin(a) # x + f * sin(a), x- f * sin(a)
            y1 = y - r*np.cos(a) # 
            x2 = x - r*np.sin(a)
            y2 = y + r*np.cos(a)
            #x_world, y_world = fig.pixel2world([x1,x2],[y1,y2])
            #line = np.array([x_world,y_world])
            #line = np.array([[x1, x2], [y1,y2]])
            #linelist.append(line)
            ax.plot([x1, x2], [y1,y2], '-', color='k', lw=1)
            set_trace()
#plt.tight_layout()
set_trace()


# plt.savefig('fig-12-1.PDF')
plt.savefig('fig-12-1.png')
#plt.show()
    #fig.show_lines(linelist,layer='vectors',color=color)

#show_vectors(p0, pa, step=20, scale=30, rotate=90, cutoff=0.01, color='white')



