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