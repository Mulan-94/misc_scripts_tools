import numpy as np

################################################################
# Calculating the maximum channel width before bw depolariszion
################################################################


simple_lsq = lambda x: (3e8/x)**2

def max_chan_width(max_rm, max_rot, obs_freq):
    """
    Get the maximum allowable channel width before bandwidth depolzn
    takes over. Sebokolodi 2020 Eq 4

    max_rm: float
        Maximum allowable RM
    max_rot: float
        Maximum allowable rotation per channel in degrees
    """
    max_rot = np.deg2rad(max_rot)
    wave_sq = simple_lsq(obs_freq)
    max_cw = (max_rot * obs_freq) /(2 * wave_sq * max_rm)
    print(f"{max_cw/1e6:.2f} MHz")
    return max_cw


# this seems to give appropriate values defining the MHz
mhz = 2**19.9

#for our observation minimum and maximum freqs available
min_freq, max_freq = 861_224_609.375, 1_706_461_914.0625

# (max_rm)     : RMs of Pictor A are around 45 but we set the range to 60
# (max_rot)    : Allowing a maximum PA rotation of 10 degrees
# (obs_freq)   : Minimum freq of observation
# (max_cwidth) : the maximum channel width before bandwidth depolarization


# max_cwidth = max_chan_width(max_rm=60, max_rot=10, obs_freq=min_freq)

# using a max_rot of 6 to get 128 (=2**7) channels. 
# nchans here will be 137 but we move it to the closest power of 2
max_cwidth = max_chan_width(max_rm=60, max_rot=6, obs_freq=min_freq)


#number of frequency channels to genenerate
n_chans = np.arange(min_freq, max_freq, max_cwidth).size

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def mkdir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)
    return dir

mkpath = os.path.join

vd = mkdir("versuz")
for reg in range(1, 101):
    plt.close("all")
    ima = mpimg.imread(f"sim-data/two/qu-fits-f-dual/qufit-reg_{reg}-MODEL.png")
    imb = mpimg.imread(f"sim-data/two/qu-fits-nf-dual/qufit-reg_{reg}-MODEL.png")
    fig, ax = plt.subplots(figsize=(16, 9), ncols=2, nrows=1)
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[0].imshow(ima)
    ax[1].imshow(imb)
    fig.tight_layout()
    fig.savefig(mkpath(vd, f"reg_{reg}.png"))
    print(f"Reg {reg} Done")



# plot RM spacing distribution histogram

def plot_rm_spacing_single(bins=15):
    rm1, rm2 = [], []
    f_rm1, f_rm2 = [], []
    nf_rm1, nf_rm2 = [], []
    for reg in range(1, 101):
        data = dict(np.load(f"sim-data/two/los-data-nf/reg_{reg}.npz"))
        rm1.append(data["in_rm"])
        rm2.append(data["in_rm2"])


        f_fit_data = np.loadtxt(f"sim-data/two/qu-fits-f-dual/qufit-reg_{reg}-fitparameters.txt")
        nf_fit_data = np.loadtxt(f"sim-data/two/qu-fits-nf-dual/qufit-reg_{reg}-fitparameters.txt")
        # get p2 [2] and p6 [6]
        f_rm1.append(f_fit_data[2])
        f_rm2.append(f_fit_data[6])

        nf_rm1.append(nf_fit_data[2])
        nf_rm2.append(nf_fit_data[6])

    
    rm1 = np.array(rm1)
    rm2 = np.array(rm2)

    f_rm1 = np.array(f_rm1)
    f_rm2 = np.array(f_rm2)
    nf_rm1 = np.array(nf_rm1)
    nf_rm2 = np.array(nf_rm2)

    #space = np.abs(rm1 - rm2)
    space = rm1 - rm2

    plt.close("all")
    fig, ax = plt.subplots(figsize=(6,6), ncols=1, nrows=1)
    ax.hist(space, bins, histtype="step", color="orangered")
    ax.set_xlabel("RM spacing between two components [rad/m$^2$]")
    ax.yaxis.set_visible(False)

    fig.savefig(mkpath(
        "versuz", f"rm_spacing-a1.png"))

    return rm1, rm2,f_rm1,f_rm2,nf_rm1,nf_rm2



def plot_rm_spacing(bins=10):
    rm1, rm2 = [], []
    f_rm1, f_rm2 = [], []
    nf_rm1, nf_rm2 = [], []
    for reg in range(1, 101):
        data = dict(np.load(f"sim-data/two/los-data-nf/reg_{reg}.npz"))
        rm1.append(data["in_rm"])
        rm2.append(data["in_rm2"])


        f_fit_data = np.loadtxt(f"sim-data/two/qu-fits-f-dual/qufit-reg_{reg}-fitparameters.txt")
        nf_fit_data = np.loadtxt(f"sim-data/two/qu-fits-nf-dual/qufit-reg_{reg}-fitparameters.txt")
        # get p2 [2] and p6 [6]
        f_rm1.append(f_fit_data[2])
        f_rm2.append(f_fit_data[6])

        nf_rm1.append(nf_fit_data[2])
        nf_rm2.append(nf_fit_data[6])

    
    rm1 = np.array(rm1)
    rm2 = np.array(rm2)

    f_rm1 = np.array(f_rm1)
    f_rm2 = np.array(f_rm2)
    nf_rm1 = np.array(nf_rm1)
    nf_rm2 = np.array(nf_rm2)

    #space = np.abs(rm1 - rm2)
    space = rm1 - rm2

    plt.close("all")
    fig, ax = plt.subplots(figsize=(8,8), ncols=1, nrows=3)
    ax[0].hist(space, bins, histtype="step", color="orangered")
    ax[0].set_xlabel("RM spacing between two components")
    ax[0].yaxis.set_visible(False)

    nlos = np.arange(1, 101)
    ax[1].plot(nlos, rm1, "k")
    ax[1].plot(nlos, f_rm1, "r")
    ax[1].plot(nlos, nf_rm1, "b")


    ax[2].plot(nlos, rm2, "k")
    ax[2].plot(nlos, f_rm2, "r")
    ax[2].plot(nlos, nf_rm2, "b")
    fig.tight_layout()
    fig.savefig(mkpath(vd, f"rm_spacing2.png"))

    return rm1, rm2,f_rm1,f_rm2,nf_rm1,nf_rm2

rm1, rm2,f_rm1,f_rm2,nf_rm1,nf_rm2 = plot_rm_spacing(bins=10)


checks = list(zip(
    zip(np.round(nf_rm1).astype(int), np.round(nf_rm2).astype(int)),
    zip(np.round(f_rm1).astype(int), np.round(f_rm2).astype(int)),
    zip(rm1, rm2)))

checks = list(zip(
    zip(np.round(nf_rm1).astype(int), np.round(f_rm1).astype(int), rm1),
    zip(np.round(nf_rm2).astype(int)), np.round(f_rm2).astype(int), rm2),
    )


# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from casacore.tables import table
from astropy import units as u
from astropy.coordinates import SkyCoord

def phase_center_from_ms(ms_name):
    """
    Get phase center from MS return it as RA and DEC string
    ms_name:  str
        Name of the MS
    """
    # 1. Read the PHASE_DIR column in the FIELD subtable, the values are in radians
    with table(f"{ms_name}::FIELD", ack=False) as field_sub:
        ra_rad, dec_rad = field_sub.getcol("PHASE_DIR").squeeze()

    # 2. convert to sky coordinatess
    sk = SkyCoord(ra_rad*u.rad, dec_rad*u.rad)

    # format ra to hms
    ra_hms = sk.ra.to_string(unit=u.hourangle, sep='hms', pad=True,
        alwayssign=True, precision=1)

    # format dec to dms
    dec_dms = sk.dec.to_string(unit=u.degree, sep='dms', pad=True,
        alwayssign=True, precision=1)
    
    return ra_hms, dec_dms



#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
import astropy.units as u
import astopy.constants as const

def linear_angular(z, d_linear=None, theta=None):

    """
    Convert angular distance to linear scale and vice versa

    inputs:
    z : float
        Redshift
    d_linear: Tuple[float, str]
        Tuple consisting of linear scale of the galaxy and its unit (mpc | kpc)
    theta: Tuple[float, str]
        Consisting of angular scale of the galaxy and its unit (rad, deg, arcsec, arcmin)

    --------------------------------------------
    Where separated by '|', it means that use a single item from either
    side of the bracket, and use only that side for the rest of the choices
    also ie. rad -> Mpc, arcmin -> kpc

    Basic ingredients:
    theta: [rad | arcmin]
        Angular size of the galaxy
    d_linear: [Mpc | kpc]
        Actual linear size of the galaxy
    d_to_gal: [Mpc | kpc]
        Distance from the earth to the galaxy
    z: [dimensionless]
        Reshift of the galaxy
    H0: [km/s]
        Hubble's constant
    v: [km/s]
        Recession velocity
    C: [km/s]
        Speed of light in a vacuum

    This information can be derived using the following basic formulas
    1. theta = d_linear / d_to_gal
    2. v = H0 * d_to_gal
    3. z = v/C
    """

    # in km/s
    C = const.c.value / 1000 

    # in km/s
    H0 = 70

    if d_linear is not None:
        # convert whatever to mpc
        d_linear = (d_linear[0] * getattr(u, d_linear[1])).to("Mpc").value
        # calculate theta in radians
        theta = (d_linear * H0)/ (C * z)
        # conver radians to arcsec
        theta = (theta*u.rad).to("arcsec").value
        res = dict(val=theta, unit="arcsec")
    else:
        # convert theta to radians
        theta = (theta[0]* getattr(u, theta[1])).to("rad").value
        # calculae linear size in mpc
        d_linear = (theta * C * z) / H0
        # convert mpc to kpc
        d_linear = (d_linear*u.Mpc).to("kpc").value
        res = dict(val=d_linear, unit="kpc")
    
    print("{val:.4f} {unit}".format(**res))
    return

#--------------------------------------------------------------------------
# revisiting channel selections

scan4 = "3:5", "9:26", "41:56", "59:61", "72:74", 


scan_full = "2:5", "9:26", "42:56", "60:61", "73:74"


import numpy as np


def unpack_select_chans(lst, oname="sel.txt"):
    imgs = np.array([f"{_}".zfill(4) for _ in range(80)])
    sel = []
    for _ in lst:
        if ":" in _:
            start, stop = _.split(":")
            start = int(start)
            stop = int(stop) + 1
            sel.extend(imgs[start:stop])
        else:
            sel.append(imgs[int(_)])

    with open(oname, "w") as fil:
        for _ in sel:
            fil.writelines(f"{_}\n")
    return sel


def jy_per_beam_to_jy_per_arcsec2(imn):
    """
    Convert jy/beam to jy/arcsecond squared
    """
    #imn = "i-mfs.fits"
    import astropy.units as u
    from radio_beam import Beam
    from astropy.io import fits

    Beam.from_fits_header(imn)
    beam = Beam.from_fits_header(imn)
    print(beam.sr.to(u.arcsec**2))

    """
    to get values as jy/arcsec
    Divide x / beam_in_arcsec
    """
    return beam.sr.to(u.arcsec**2).value





#-------------------------------------------------------------------------------
import numpy as np


def create_selected_files(fname, oname, sel):
    with open(fname, "r") as fil:
        dat = fil.readlines()

    regs = dat[:3]
    for _ in sel:
        regs.append(dat[_+2])

    with open(oname, "w") as fil:
        fil.writelines(regs)

    return


# from masked regions
# regions showing double peaks
doubles = {81,107,141,209,484,493,494,548,549,676,742,743,820,900,1039,1041,1362,1373,1430,1487,1489,1556,1563,1569,1632,1635,1705,1838}


# regions showing weird looking multiple peaks
weird_multi = {
    55,179,180,228,232,233,234,296,297,355,420,485,486,547,558,610,611,612,623,625,626,680,691,747,748,820,885,886,887,904,905,1043,1044,1045,1046,1047,1432,1433,1501,1502,1503,1566,1567,1625,1626,1637,1638,1639,1707,1708,1709,1710,1841,1845
}

# regions showing smaller peaks on the side
side_peaks = [
    3,183,235,236,332,353,354,356,357,358,361,362,363,364,365,366,367,368,459,556,614,617,618,619,620,621,622,684,685,686,687,688,1378,1570,1571,1572,1711,1712,1714,1842,1843,1844,1892,1897,1898,1899,1900,1901,
]


# # regions showing noise
noisy = [
    45,432,497,560,584,590,627,648,656,693,694,765,829,831,841,895,896,964,1094,1393,1395,1912,1955,1962,1963,1964,1965,2006,2007,2120,2182,71,72,154,162,207,496,2034
]

create_selected_files(
    "products/masked-scrap-outputs-s3/regions/regions-valid.reg",
    "investigations/doubles.reg", doubles)

create_selected_files(
    "products/masked-scrap-outputs-s3/regions/regions-valid.reg",
    "investigations/weird-doubles.reg", weird_multi)

create_selected_files(
    "products/masked-scrap-outputs-s3/regions/regions-valid.reg",
    "investigations/side-peaks.reg", side_peaks)

create_selected_files(
    "products/masked-scrap-outputs-s3/regions/regions-valid.reg",
    "investigations/noisy.reg", noisy)



# 2,3,4,5,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,60,61,73,74


################################################################################
#I am calculating some lobe statistics
#  spectral index
################################################################################

import numpy as np
from astropy.io import fits

def calculate_stats(image, mask):
    data = np.squeeze(fits.getdata(image))
    mask_data = np.squeeze(fits.getdata(mask)).astype(float)
    mask_data = np.ma.masked_less(mask_data, 1)
    mask_data = np.ma.filled(mask_data, np.nan)


    masked = mask_data * data
    
    mean = np.nanmean(masked)
    std = np.nanstd(masked)
    print(f"mean: {mean:10.2}")
    print(f"std :   {std:10.2}")
    return


lobes = os.path.join(os.environ["mask_dir"], "lobes.fits")
elobe = os.path.join(os.environ["mask_dir"], "east-lobe.fits")
wlobe = os.path.join(os.environ["mask_dir"], "west-lobe.fits")

image = "products/spi-fitting/spi-map.alpha.fits"
image_err = "products/spi-fitting/spi-map.alpha_err.fits"





###############-----------------------------------------------------------------
# REad the HISTORY of an MS

from casacore.tables import table
def read_ms_history(ms_name):
    print(f"REading MS:   {ms_name}")
    with table(f"{ms_name}::HISTORY") as tb:
        tb.getcol("MESSAGE")
        with open("ms-history.txt", "w") as fil:
            history = [f"{_}\n" if "taskname" not in _ else f"\n\n" + "#"*80 + f"\n\n{_}" for _ in tb.getcol("MESSAGE")]
          
            fil.writelines(history)
    print("History written")
    return



###############-----------------------------------------------------------------

def radio_power(z, freq, spi, flux):
    """
    Calculate radio power of a galaxy

    # obtained from eq 13 of Boxelaar, Weeren, Botteon of halo-FDCA
    # https://doi.org/10.1016/j.ascom.2021.100464
    # https://ned.ipac.caltech.edu/Library/Distances/distintro.html#:~:text=Since%20the%20luminosity%20distance%20equals,or%2059%20Mpc%20(18%25).


    Parameters
    ----------
    z: float
        Redshift of the galaxy
    spi: float
        Spectral index
    freq: float
        Frequency in MHz which flux is at
    flux: float
        Flux in Jansky

    Returns
    -------
    power: float
        Radio power in W/Hz units
    """


    from astropy.cosmology import FlatLambdaCDM
    from math import pi
    import astropy.units as u


    # set up the cosomological params: Hubble constant and matter 0.3
    hubble_const = 70
    cosmo = FlatLambdaCDM(H0=hubble_const, Om0=0.3)

    linear_dist = (3e8* z)/hubble_const   # distance in kpc

    # similar to cosmo.luminosity_distance(z)
    luminosity_dist = cosmo.luminosity_distance(z)



    power = (((4 * pi * (luminosity_dist**2)) / ((1 + z)**(1 + spi))) *\
                     flux_density).to(u.W/u.Hz)

    # this one below is from Boxelaar
    # power2 = (4*pi*luminosity_dist**2. *((1.+z)**((-1.*spi) - 1.))* flux_density*((freq/freq)**spi)).to(u.W/u.Hz)

    print(power)
    return power.value


# for example for Pictor A using robertson (1973) values
radio_power(0.035, 408, -0.75, 135)



###############-----------------------------------------------------------------