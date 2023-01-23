"""
Some references
see https://stackoverflow.com/questions/61532337/python-modulenotfounderror-no-module-named
"""
import argparse
import os
import sys
from glob import glob

# Lexy: jUST being lazy, sort this out when I sort it out
sys.path.append(os.getcwd())

import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from scipy import signal
from astropy.constants import c as light_speed

from scrap import IOUtils

from ipdb import set_trace


# matplotlib.rcParams.update({'font.size':18, 'font.family':'DejaVu Sans'})
# matplotlib.use('Agg') 
plt.style.use("seaborn")


FIGSIZE = (16,9)
get_wavel = lambda x: 3e8/x
lw = 1.2

def lambda_to_faraday(lambda_sq, phi_range, lpol):
    """
    Computes Faraday Spectra using RM-Synthesis 
    as defined by Brentjens and de Bruyn (2005) eqn. 36
    from polarised surface brightes per lambda

    lambda_sq
        Lambda squared ranges
    phi_range
        Range of faraday depths to consider
    lpol
        Observed complex polarised surface brightness for each lambda squared

    Returns
    -------
    Polarised spectra per depth over a range of depths
    """
    N = len(lambda_sq)

    # get the initial lambda square value from the mean
    init_lambda_sq = np.nanmean(lambda_sq)
    fdata = np.zeros([len(phi_range)], dtype=complex)
    

    # for each phi, calculate the depth
    # we're getting the rm spectrum per depth
    for k, phi in enumerate(phi_range):
        try:
            fdata[k] = pow(N, -1) * np.nansum(
                lpol * np.exp(-2j * (lambda_sq-init_lambda_sq) * phi)
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

        rmtf = signal.convolve(rmsf_fixed, dirac, mode='same')
        rmtf = rmtf[dshift:-dshift+1]

        fspectrum -= f_comp * rmtf

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
        rmtf = signal.convolve(padded_rmsf, delta, mode="same")[padding+1:-padding]

        ## scaling
        nrmsf = rmtf*fraction

        # subtract scaled and shifted rmtf from whatever is there
        residuals -= nrmsf


    # should be a gaussian with FWHM same as the rmtf main lobe
    fwhm = get_rmsf_fwhm(None, None, lambdas=minmax(lambdas))
    
    # FWHM of a gaussian is sigma * ((8ln2)^0.5)
    sigma = fwhm / np.sqrt(8 * np.log(2))
    restoring_beam = np.exp(-0.5 * (phi_range/sigma)**2) 

    restored = signal.convolve(clean_components, restoring_beam, mode="same")
    restored += residuals

    # # if n%500 == 0:
    fig, ax = plt.subplots(nrows=3,ncols=2, figsize=FIGSIZE)
    ax[0, 0].plot(phi_range, np.abs(dirty_spectrum), "b--", label="Dirty")
    ax[0, 0].legend()
    
    ax[0, 1].plot(phi_range, np.abs(residuals), "y-", label="residuals")
    ax[0, 1].axvline(phi_range[peak_loc], color="orangered", linestyle="--")
    ax[0, 1].axhline(np.abs(residuals[peak_loc]), color="orangered", linestyle="--")
    ax[0, 1].legend()

    ax[1, 0].plot(phi_range, np.abs(clean_components), "k-", label="cleans")
    ax[1, 0].legend()
    
    ax[1, 1].plot(phi_range, np.abs(rmtf), label="rmtf")

    az = ax[1, 1].twinx()
    az.plot(phi_range, np.abs(nrmsf), "r:", label="Scaled rmtf")
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
    fig, (ax, ay) = plt.subplots(1, 2, figsize=FIGSIZE)
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
        
        fclean, fcomp, rmtf = rm_clean(lambda_sq, phi_range, fdirty.copy(), 
                    niter=niter, gain=gain)
        # fclean = lexy_rm_clean(lambda_sq, phi_range, fdirty, n_iterations=500, loop_gain=0.1, threshold=None)
        outs.update({"fclean": fclean, "fcomp": None, "rmtf": rmtf })

    return outs


def read_npz(fname):
    with np.load(fname, allow_pickle=True) as dat:
        datas = {k: v for k, v in dat.items()}
    return datas



def _max_fdepth(central_freq, chanwidth):
    """
    See Eqn 63 Brentjens

    The maximum Faraday depth to which one has more than 50% sensitivity


    chanwidth
        A single channel width in frequency
    """
    
    delta_lamsq = get_wavel(chanwidth)**2
    phi_max = (3**0.5) / delta_lamsq
    return np.abs(phi_max)



def max_visible_depth_scale(min_freq):
    """
    The scale in φ space to which sensitivity has dropped to 50%

    min_freq
        Smallest frequency available in the band
    """
    return np.pi / get_wavel(min_freq)**2


def fwhm_resolution(min_freq, max_freq):
    """
    Corrected by schintzeler 2008 to 3.8/difference in wavelength squared
    For originial equation See Eqn 61 Brentjens
    min_freq
        Initial frequency of the observation band
    max_freq:
        Where the band ends
    """
    # these values will reverse in wavelength because freq and wave are
    # inversley proportional
    max_wav = c/min_freq
    min_wav = c/max_freq
    bw_wave_sq = max_wav**2 - min_wav**2
    phi_max = 3.8/bw_wave_sq
    return phi_max.value

def arg_parser():
    from textwrap import fill, dedent
    class BespokeFormatter(argparse.RawDescriptionHelpFormatter):
        def _fill_text(self, text, width, indent):
            wrap_width = 80
            return "_"*wrap_width + "\n\n Description\n\n" +\
                "\n".join([fill(dedent(_), width=wrap_width,
                                initial_indent=" ", subsequent_indent="  ",
                                tabsize=2)
                            for _ in text.splitlines()]) + \
                "\n" + "_"*wrap_width

    parse = argparse.ArgumentParser(
        formatter_class=BespokeFormatter,
        description="""This script takes in I, Q and U data and does the prcoess
        of RM-SYNthesis and RM-CLEAN. It gives these outputs.

        Pickle files containing:
        .\t1. The dirty FDF (Faraday Dispersion Funtion / Faraday spectrum)
        .\t2. The cleaned FDF
        .\t3. Faraday depths used
        
        These are the keys:
        .\tdepths
        .\tfdirty
        .\tfclean
        .\trmtf

        Plots of:
        .\t4. The dirty and clean FDF and position angle vs wavelength sq and its
        linear squares fit
        .\t5. The DATAs RMSF
        """)
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
        output_dir = f"{opts.output_dir}-{os.path.basename(data_dir)}"
    
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    
        # get the various lines of sight
        data_files =  sorted(glob(f"{data_dir}/*.npz"), key=os.path.getctime)

        for _i, data_file in enumerate(data_files):
            reg_num = os.path.splitext(os.path.basename(data_file))[0].split("_")[-1]
        
            print(f"File: {data_file}")
            datas = read_npz(data_file)

            mask = datas["mask"]
            for key, value in datas.items():
                if key not in ["mask"]:
                    datas[key] = np.ma.masked_array(data=value, mask=mask).compressed()


            freq, stokes_i, linear_pol ,fpol, fpol_err, pangle, pangle_err = (
                datas["freqs"], datas["I"], datas["lpol"],
                datas["fpol"], datas["fpol_err"], datas["pangle"],
                datas["pangle_err"])

            lam2 = (light_speed.value/freq)**2

            if  len(linear_pol)==0:
                print(f"Skipping region {reg_num}")
                continue
                        
            rm_products = rm_synthesis(
                lam2, linear_pol.data, phi_max=opts.max_fdepth,
                phi_step=opts.depth_step, niter=opts.niters, clean=True)
            
            del rm_products["fcomp"]
            out_dir = IOUtils.make_out_dir(
                os.path.dirname(data_file) + f"-depths-{opts.max_fdepth}")
            outfile = os.path.join(out_dir, f"reg_{reg_num}.npz")
            print(f"Saving data to:         {outfile}")
            np.savez(outfile, **rm_products)

            
            #plotting everything

            plt.close("all")
            fig, ax = plt.subplots(figsize=FIGSIZE, ncols=3)
            ax[0].errorbar(lam2, fpol, yerr=fpol_err, 
                            fmt='o', ecolor="red")
            ax[0].set_xlabel('$\lambda^2$ [m$^{-2}$]')
            ax[0].set_ylabel('Fractional Polarisation')

            
            u_pangle = np.unwrap(pangle, period=np.pi, discont=np.pi/2)
            
            # ax[1].plot(lam2, pangle, "r+", label="original")
            ax[1].errorbar(lam2, u_pangle, yerr=pangle_err,
                            fmt='o', ecolor="red", label="unwrapped angle")
            # linear fitting
            res = np.ma.polyfit(lam2, u_pangle, deg=1)
            reg_line = np.poly1d(res)(lam2)
  
            ax[1].plot(lam2, reg_line, "g--", label=f"linear fit, slope: {res[0]:.3f}", lw=lw)
            ax[1].set_xlabel('$\lambda^2$ [m$^{-2}$]')
            ax[1].set_ylabel('Polarisation Angle')
            ax[1].legend()

            fd_peak = np.where(np.abs(rm_products["fclean"]) == np.abs(rm_products['fclean']).max())
            rm_val = rm_products["depths"][fd_peak][0]

            ax[2].plot(rm_products['depths'], np.abs(rm_products['fdirty']),
                        'r--', label='Dirty Amp')
            if "fclean" in rm_products:
                ax[2].plot(rm_products['depths'], np.abs(rm_products['fclean']),
                            'k', label=f'Clean Amp, RM {rm_val:.2f}')
                # ax[2].axvline(rm_val, label=f"{rm_val:.3f}")
            ax[2].set_xlabel('Faraday depth [rad m$^{-2}$]')
            ax[2].set_ylabel('Farady Spectrum')
            ax[2].legend(loc='best')

            fig.tight_layout()
            oname = os.path.join(output_dir, f"reg_{reg_num}.png")
            print(f"Saving Plot at:          {oname}")
            fig.savefig(oname)
        

        # RMTF
        plt.close("all")
        fig, ax = plt.subplots(figsize=FIGSIZE, ncols=1, squeeze=True)
        
        ax.plot(rm_products["depths"], np.abs(rm_products["rmtf"]), "k-",
            lw=lw, label="Amp")
        ax.plot(rm_products["depths"], rm_products["rmtf"].real,
            color="orangered", ls="--", lw=lw, label="Real")
        ax.plot(rm_products["depths"], rm_products["rmtf"].imag, "g:",
            lw=lw, label="Imag")
        ax.set_xlabel(r"Faraday depth $\phi$")
        ax.set_ylabel("RMTF")
        ax.legend()
        fig.tight_layout()
        print(f"Saving RMTF to: " + os.path.join(output_dir, "rmtf.png"))
        fig.savefig(os.path.join(output_dir, "rmtf.png"))






"""
SpwID  Name   #Chans   Frame   Ch0(MHz)  ChanWid(kHz)  TotBW(kHz) CtrFreq(MHz)  Corrs          
0      none      81   TOPO     861.120     10449.219    846386.7   1279.0889   XX  XY  YX  YY

Spectral Windows:  (1 unique spectral windows and 1 unique polarization setups)
  SpwID  Name   #Chans   Frame   Ch0(MHz)  ChanWid(kHz)  TotBW(kHz) CtrFreq(MHz)  Corrs          
  0      none    4096   TOPO     856.000       208.984    856000.0   1283.8955   XX  XY  YX  YY
Sources: 1



max_visible_depth_scale(861.120e6


# get the resolution of the RMSF
start = 856e6
end = 856e6 + 856000e3
fwhm_resolution(start, end)
we get 41 rad/m2


# Get the maximum rm depth
We can get the max rm allowable for our data from brentjens code
1. Get the list of input frequencies in ascending order
2. Convert to lambda square
3. sort min to max
4. get the minimum channel width min(waves[:-1] - waves[1:])
5. divide 2root3 / number 4
See equation 61

I found this value to be around 3964

"""