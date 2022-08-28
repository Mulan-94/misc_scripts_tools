import argparse
import os
import sys
from glob import glob

# Lexy: jUST being lazy, sort this out when I sort it out
sys.path.append(os.getcwd())

import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
# matplotlib.rcParams.update({'font.size':18, 'font.family':'DejaVu Sans'})
from scipy import signal

# setting this to qu_po because of its location in the misc_scripts_and_tools
# see https://stackoverflow.com/questions/61532337/python-modulenotfounderror-no-module-named
from scrap import IOUtils

from ipdb import set_trace

matplotlib.use('Agg') 
plt.style.use("seaborn")


def lambda_to_faraday(lambda_sq, phi_range, pol_lam2):
    """
    Computes Faraday Spectra using RM-Synthesis 
    as defined by Brentjens and de Bruyn (2005) eqn. 36
    from polarised surface brightes per lambda

    lambda_sq
        Lambda squared ranges
    phi_range
        Range of faraday depths to consider
    pol_lam2
        Observed complex polarised surface brightness for each lambda squared

    Returns
    -------
    Polarised spectra per depth over a range of depths
    """

    N = len(lambda_sq)

    # get the initial lambda square value from the mean
    init_lambda_sq = lambda_sq.mean()
    fdata = np.zeros([len(phi_range)], dtype=complex)
    

    # for each phi, calculate the depth
    # we're getting the rm spectrum per depth
    for k, phi in enumerate(phi_range):
        try:
            fdata[k] = pow(N, -1) * np.sum(
                pol_lam2 * np.exp(-2j * (lambda_sq-init_lambda_sq) * phi)
                )
        except ZeroDivisionError:
            continue
    return fdata


def rm_clean(lam2, phi_range, fspectrum, niter=500, gain=0.1):
    """
    Clean out the dirty Faraday dispersion measure
    """

    fwhm = (3.8/ abs(lam2[0]-lam2[-1]))

    # FWHM of a gaussian is sigma * ((8ln2)**0.5)
    sigma = (fwhm/2.35482)

    # see wikipedia for general fomr of a gaussian. Mean is 0, amp is 1
    # https://en.wikipedia.org/wiki/Gaussian_function
    Gauss = np.exp(-0.5 * (phi_range/sigma)**2) 

    # I am padding here to avoid edge effects.
    # aka increasing the range of phi on both sides
    pad = abs(phi_range[-1]) * 2
    dpad = abs(phi_range[0]-phi_range[1])
    phi_pad = np.arange(-pad, pad, dpad)
    dshift = int(pad/(2.0 * dpad))

    rmsf_orig = lambda_to_faraday(lam2, phi_range, 1) 
    rmsf_fixed = lambda_to_faraday(lam2, phi_pad, 1) 
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
    return fclean, components, rmsf_orig



def lexy_rm_clean(lambdas, phi_range, dirty_spectrum, n_iterations=500, loop_gain=0.1, threshold=None):

    minmax = lambda x: (x.min(), x.max())
    residuals = dirty_spectrum.copy() 
    clean_components = np.zeros_like(dirty_spectrum)

    for n in range(n_iterations):
        # if the noise in the residuals has reached the threshold stop the loop
        # print("Iteration ", n)

        if threshold is not None:
            if residuals.std() <= threshold:
                break
        
        # Find location of peak in dirty spectrum/
        # peak_loc = np.where(residuals==np.max(residuals))
        peak_loc = np.where(np.abs(residuals)==np.max(np.abs(residuals)))
        
        # scale real and imagiary by loop gain
        fraction = residuals[peak_loc] * loop_gain
        clean_components[peak_loc] += fraction

        #shift and scale
        step = np.abs(phi_range[0]-phi_range[1])
        padding = phi_range.size//2
        edge = phi_range.max() + (padding * step) + step
        padded_phi_range = np.arange(-edge, edge, step)
        padded_rmsf = lambda_to_faraday(lam2, padded_phi_range, 1)
        delta = np.zeros_like(phi_range)
        delta[peak_loc] = 1

        ## shifting
        rmsf = signal.convolve(padded_rmsf, delta, mode="same")[padding+1:-padding]

        ## scaling
        nrmsf = rmsf*fraction
        
        # if n%100 == 0:
        #     fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(16,9))
        #     ax[0, 0].plot(phi_range, np.abs(dirty_spectrum), "b--", label="Dirty")
        #     ax[0, 0].legend()
            
        #     ax[0, 1].plot(phi_range, np.abs(residuals), "y-", label="residuals")
        #     ax[0, 1].axvline(phi_range[peak_loc], color="orangered", linestyle="--")
        #     ax[0, 1].axhline(np.abs(residuals[peak_loc]), color="orangered", linestyle="--")
        #     ax[0, 1].legend()

        #     ax[1, 0].plot(phi_range, np.abs(clean_components), "k-", label="cleans")
        #     ax[1, 0].legend()
            
        #     ax[1, 1].plot(phi_range, np.abs(rmsf), label="RMSF")

        #     az = ax[1, 1].twinx()
        #     az.plot(phi_range, np.abs(nrmsf), "r:", label="Scaled rmsf")
        #     az.set_ylabel("Scaled", color="red")
        #     az.tick_params(axis="y", labelcolor="red")
        #     ax[1, 1].legend()


        #     # ax[2, 0].plot(phi_range, np.abs(restored), label="Restored + residuals")
        #     # ax[2, 0].legend()

        #     fig.suptitle(f"Iteration {n}")
        #     fig.tight_layout()
        #     plt.show()
        
        # subtract scaled and shifted RMSF from whatever is there
        residuals -= nrmsf


    # should be a gaussian with FWHM same as the RMSF main lobe
    fwhm = get_rmsf_fwhm(None, None, lambdas=minmax(lambdas))
    
    # FWHM of a gaussian is sigma * ((8ln2)^0.5)
    sigma = fwhm / np.sqrt(8 * np.log(2))
    restoring_beam = np.exp(-0.5 * (phi_range/sigma)**2) 

    restored = signal.convolve(clean_components, restoring_beam, mode="same")
    restored += residuals

    # # if n%500 == 0:
    fig, ax = plt.subplots(nrows=3,ncols=2, figsize=(16,9))
    ax[0, 0].plot(phi_range, np.abs(dirty_spectrum), "b--", label="Dirty")
    ax[0, 0].legend()
    
    ax[0, 1].plot(phi_range, np.abs(residuals), "y-", label="residuals")
    ax[0, 1].axvline(phi_range[peak_loc], color="orangered", linestyle="--")
    ax[0, 1].axhline(np.abs(residuals[peak_loc]), color="orangered", linestyle="--")
    ax[0, 1].legend()

    ax[1, 0].plot(phi_range, np.abs(clean_components), "k-", label="cleans")
    ax[1, 0].legend()
    
    ax[1, 1].plot(phi_range, np.abs(rmsf), label="RMSF")

    az = ax[1, 1].twinx()
    az.plot(phi_range, np.abs(nrmsf), "r:", label="Scaled rmsf")
    az.set_ylabel("Scaled", color="red")
    az.tick_params(axis="y", labelcolor="red")
    ax[1, 1].legend()


    ax[2, 0].plot(phi_range, np.abs(restored), label="Restored + residuals")
    ax[2, 0].legend()

    fig.suptitle(f"Iteration {n}")
    fig.tight_layout()
    plt.show()
    
    return restored


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


def rm_synthesis(lambda_sq, lpol, phi_max=600, phi_step=10, niter=1000, gain=0.1,
    clean=False):
    """
    lambda_sq:
        Lambda squared
    lpol
        Linear polzn akd QU power
    phi_max
        Maximum depth
    niter:
        Number of iteration for clean
    gain:
        Gain factor for clean
    plot:
        To plot or not?
    
    Algorithm
    1. Get the dirty faraday spectra
    2. Perform RM clean
    """

    phi_range =  np.arange(-phi_max, phi_max+phi_step, phi_step)
    # this ensures that the middle value is zero. 

    fdirty = lambda_to_faraday(lambda_sq, phi_range, lpol)

    outs = { "depths": phi_range, "fdirty": fdirty.copy()}
    
    if clean:
        
        fclean, fcomp, rmsf = rm_clean(lambda_sq, phi_range, fdirty.copy(), 
                    niter=niter, gain=gain)
        # fclean = lexy_rm_clean(lambda_sq, phi_range, fdirty, n_iterations=500, loop_gain=0.1, threshold=None)
        outs.update({"fclean": fclean, "fcomp": None, "rmsf": rmsf })

    return outs


def read_npz(fname):
    with np.load(fname, allow_pickle=True) as dat:
        datas = {k: v for k, v in dat.items()}
    return datas


wavelength = lambda x: 3e8/x

def max_fdepth(central_freq, chanwidth):
    """
    See Eqn 63 Brentjens

    The maximum Faraday depth to which one has more than 50% sensitivity

    central_freq:
        Central frequency of channel in freq band
    chanwidth
        A single channel width in frequency
    """
    
    delta_lamsq = wavelength(central_freq+chanwidth)**2 -  wavelength(central_freq-chanwidth)**2
    phi_max = (3**0.5) / delta_lamsq
    return np.abs(phi_max)


def max_visible_depth_scale(min_freq):
    """
    The scale in φ space to which sensitivity has dropped to 50%

    min_freq
        Smallest frequency available in the band
    """
    return np.pi / wavelength(min_freq)**2


def get_rmsf_fwhm(start_band, bandwidth, lambdas=None):
    """
    See Eqn 61 Brentjens
    Approximate FWHM of the main peak of the RMSF
    start_band
        Initial frequency of the observation band
    bandwidth:
        Total frequency bandwidth of the observation
    """
    if lambdas:
        lambdas = tuple(lambdas)
        del_lamsq = lambdas[-1] - lambdas[0]
    else:
        del_lamsq = wavelength(start_band+bandwidth)**2 - wavelength(start_band)**2
    fwhm = (2*(3**0.5)) / del_lamsq
    # fwhm = 3.8 / del_lamsq
    return fwhm


def arg_parser():
    parse = argparse.ArgumentParser()
    parse.add_argument("-id", "--input-dir", dest="data_dirs", type=str,
        nargs="+",
        help="Directory containing the various data files")
    parse.add_argument("-od", "--output-dir", dest="output_dir", type=str,
        default="plots_rmsynth",
        help="Where to dump the output plots if available")

    parse.add_argument("-md", "--max-fdepth", dest="max_fdepth", type=int,
        default=500,
        help="Maximum Faraday depth. Default is 500")
    parse.add_argument("--depth-step", dest="depth_step", type=int,
        default=10,
        help="Faraday depth step. Default is 10")
    parse.add_argument("-iters", "--iters", dest="niters", type=int,
        default=1000,
        help="Number of RM clean iterations. Default is 1000")
    return parse


if __name__ == "__main__":
    opts = arg_parser().parse_args()

    for _i, data_dir in enumerate(opts.data_dirs):
        if not os.path.isdir(f"{opts.output_dir}-{_i}"):
            os.makedirs(f"{opts.output_dir}-{_i}")

        output_dir = f"{opts.output_dir}-{_i}"

        # get the various lines of sight
        data_files =  sorted(glob(f"{data_dir}/*.npz"))

        for data_file in data_files:
            reg_num = os.path.splitext(os.path.basename(data_file))[0].split("_")[-1]

            print(f"File: {data_file}")
            datas = read_npz(data_file)
            freq, stokes_q, stokes_u, stokes_i = (
                datas["freqs"], datas["Q"], datas["U"], datas["I"])

            linear_pol = stokes_q + 1j *stokes_u

            light_speed = 3e8
            lam2 = (light_speed/freq)**2

            # get the linear polzn values that are not nAN
            ind_nan = ~np.isnan(np.absolute(linear_pol))
            linear_pol = linear_pol[ind_nan]
            lam2 = lam2[ind_nan]

            if linear_pol.size==0:
                print(f"Skipping region {reg_num}")
                continue
            
            stokes_i = stokes_i[ind_nan]

            rm_products = rm_synthesis(lam2, linear_pol, phi_max=opts.max_fdepth,
                phi_step=opts.depth_step, niter=opts.niters, clean=True)
            
            del rm_products["fcomp"]
            out_dir = IOUtils.make_out_dir(os.path.dirname(data_file) + f"-depths-{opts.max_fdepth}")
            outfile = os.path.join(out_dir, f"reg_{reg_num}.npz")
            print(f"Saving data to:         {outfile}")
            np.savez(outfile, **rm_products)

            #plotting everything
            plt.close("all")
            fig, ax = plt.subplots(figsize=(16, 9), ncols=2)

            # ax[0].plot(lam2, np.absolute(linear_pol), 'o', label='| P |')
            # ax[0].set_ylabel('Polarisation Intensity [Jy/bm]')
            # ax[0].legend(loc='best')
            
            ax[0].plot(lam2, np.absolute(linear_pol)/stokes_i, 'o')
            ax[0].set_xlabel('$\lambda^2$ [m$^{-2}$]')
            ax[0].set_ylabel('Fractional Polarisation')
            

            ax[1].plot(rm_products['depths'], np.absolute(rm_products['fdirty']), 'r--', label='Dirty Amp')

            if "fclean" in rm_products:
                ax[1].plot(rm_products['depths'], np.absolute(rm_products['fclean']), 'k', label='Clean Amp')

            # if "fcomp" in rm_products:
            #     ax[1].plot(rm_products['depths'], np.absolute(rm_products['fcomp']), 'orangered', label='Clean Amp')

            

            ax[1].set_xlabel('Faraday depth [rad m$^{-2}$]')
            ax[1].set_ylabel('Farady Spectrum')
            ax[1].legend(loc='best')

            fig.tight_layout()
            oname = os.path.join(output_dir, f"reg_{reg_num}.png")
            print(f"Saving Plot at:          {oname}")
            fig.savefig(oname)






"""
SpwID  Name   #Chans   Frame   Ch0(MHz)  ChanWid(kHz)  TotBW(kHz) CtrFreq(MHz)  Corrs          
0      none      81   TOPO     861.120     10449.219    846386.7   1279.0889   XX  XY  YX  YY


46:  1341784179.6875
47:  1352233398.4375


max_fdepth(1341784179.6875, cwid)
max_fdepth(1352233398.4375, cwid)
max_visible_depth_scale(861.120e6)
rmsf_fwhm(861.120e6, 846386.7e3)
"""