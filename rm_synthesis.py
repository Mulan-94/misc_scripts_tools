import os
from glob import glob

import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
# matplotlib.rcParams.update({'font.size':18, 'font.family':'DejaVu Sans'})
from scipy import signal

from ipdb import set_trace

plt.style.use("seaborn")


def Faraday2Lambda(lam2, phi_range, pol_lam2):

    """
    Computes Faraday Spectra using RM-Synthesis 
    as defined by Brentjens and de Bruyn (2005)

    """

    N = len(lam2)

    l20 = lam2.mean()
    fdata = np.zeros([len(phi_range)], dtype=complex)
    

    for k, phi in enumerate(phi_range):
        try:
            fdata[k] = pow(N, -1) * np.sum(pol_lam2 * 
                    np.exp(-2j * (lam2-l20) * phi))   
        except ZeroDivisionError:
            continue
    return fdata



def RMClean(lam2, phi_range, fspectrum, 
          niter=500, gain=0.1):

    fwhm = (3.8/ abs(lam2[0]-lam2[-1]))
    sigma = (fwhm/2.35482)
    Gauss = np.exp(-0.5 * (phi_range/sigma)**2) 

    # I am padding here to avoid edge effects.
    
    pad = abs(phi_range[-1]) * 2
    dpad = abs(phi_range[0]-phi_range[1])
    phi_pad = np.arange(-pad, pad, dpad)
    dshift = int(pad/(2.0 * dpad))

    rmsf_fixed = Faraday2Lambda(lam2, phi_pad, 1) 
    components = np.zeros([len(phi_range)], dtype=complex)

    for n in range(niter):
        temp = np.zeros([len(phi_range)], dtype=complex)
        f_amp = np.absolute(fspectrum)
        ind = np.where(f_amp == f_amp.max())[0]
        f_comp = fspectrum[ind[0]] * gain
        temp[ind[0]] = f_comp
        components += temp         
    
        dirac = np.zeros(len(phi_range))
        dirac[ind[0]] = 1
        rmsf = signal.convolve(rmsf_fixed, dirac, mode='same')
        rmsf = rmsf[dshift:-dshift+1]
 
        fspectrum -= f_comp * rmsf

    Fres = fspectrum
    fclean = signal.convolve(components, Gauss, mode='same') + Fres

    return fclean, components


def plot_data(lam, plam, phi, fphi):
    
    
    fig, (ax, ay) = plt.subplots(1, 2, figsize=(12, 5))
    ax.plot(lam, abs(plam), 'b.')
    ax.set_xlabel('Wavelength [m$^2$]')
    ax.set_ylabel('Fractional Polarization')
          
    ay.plot(phi, abs(fphi), 'b')
    ay.set_xlabel('Faraday Depth [rad m$^{-2}$]')
    ay.set_ylabel('Faraday Spectrum')
    plt.tight_layout()
    plt.show()



def main(lam, pdata, phi_max=5000, phi_step=10,
         niter=1000, gain=0.1, plot=False):


    phi_range =  np.arange(-phi_max, phi_max+phi_step, phi_step)
    # this ensures that the middle value is zero. 

    fdirty = Faraday2Lambda(lam, phi_range, pdata)
    
    
    #fclean, fcomp = RMClean(lam, phi_range, fdirty, 
    #                niter=niter, gain=gain)

    #if plot:
    #    plot_data(x, pdata, phi_range, fclean)


    return phi_range, fdirty



qfiles = sorted(glob("Q-regions-mpc-20-toto/*.npz"))
ifiles = sorted(glob("I-regions-mpc-20-toto/*.npz"))
ufiles = sorted(glob("U-regions-mpc-20-toto/*.npz"))


def read_npz(fname):
    with np.load(fname, allow_pickle=True) as dat:
        freqs = dat["freqs"]* 1e9
        flux = dat["flux_jybm"]
    return {"flux": flux, "freqs": freqs}


for files in zip(ifiles, qfiles, ufiles):
    datas = {}

    for fil in files:
        datas[fil[0]] = read_npz(fil)

    reg_num = fil.split("/")[-1].split("_")[1]
    file_core = "-".join(fil.split("/")[0].split("_")[0].split("-")[1:-1])

    freq = datas["I"]["freqs"]
    Q = datas["Q"]["flux"]
    U = datas["U"]["flux"]
    I = datas["I"]["flux"]

    P = Q + 1j * U

    c = 3e8
    lam2 = (c/freq)**2

    ind_nan = ~np.isnan(np.absolute(P))
    P = P[ind_nan]

    if P.size==0:
        print(f"Skipping region {reg_num}")
        continue
    
    lam2 = lam2[ind_nan]

    faraday_depth, RM_spectrum = main(lam2, P, phi_max = 200, phi_step=10)

    odir = "plots-rmsynth-"+file_core
    if not os.path.isdir(odir):
        os.makedirs(odir)

    plt.close("all")
    fig, ax = plt.subplots(figsize=(16, 9), ncols=2)

    ax[0].plot(lam2, np.absolute(P), 'g.', label='|P|')
    #ax[0].plot(lam2, P.real, 'b.', label='Q')
    #ax[0].plot(lam2, P.imag, 'r.', label='U')
    ax[0].set_xlabel('$\lambda^2$ [m$^{-2}$]')
    ax[0].set_ylabel('Polarisation Intensity [Jy beam$^{-1}$')
    ax[0].legend(loc='best')

    ax[1].plot(faraday_depth, np.absolute(RM_spectrum), 'g', label='amp')
    #ax[1].plot(faraday_depth, RM_spectrum.real, 'b', label='real')
    #ax[1].plot(faraday_depth, RM_spectrum.imag, 'r', label='imag')
    ax[1].set_xlabel('Faraday depth [rad m$^{-2}$]')
    ax[1].set_ylabel('Polarisation Intensity')
    ax[1].legend(loc='best')

    fig.tight_layout()
    fig.savefig(os.path.join(odir, f"reg_{reg_num}.png"))






