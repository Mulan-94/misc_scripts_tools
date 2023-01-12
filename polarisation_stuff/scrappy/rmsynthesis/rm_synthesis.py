"""
Edited from Lerato's script
Some references
see https://stackoverflow.com/questions/61532337/python-modulenotfounderror-no-module-named
"""
import argparse
import os
import sys
from glob import glob
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from concurrent.futures import ProcessPoolExecutor
from functools import partial

PATH = set(sys.path)
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir))
if not PATH.issuperset(PROJECT_ROOT):
    sys.path.append(PROJECT_ROOT)

from utils.genutils import fullpath, make_out_dir, dicto
from utils.logger import logging, LOG_FORMATTER, setup_streamhandler
from utils.rmmath import lambda_sq

from ipdb import set_trace

snitch = logging.getLogger(__name__)
snitch.addHandler(setup_streamhandler())
snitch.setLevel("INFO")

# import matplotlib
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
    init_lambda_sq = lambda_sq.mean()
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
        ind = np.argmax(f_amp)
        f_comp = fspectrum[ind] * gain
        temp[ind] = f_comp
        components += temp         
    
        dirac = np.zeros(len(phi_range))
        dirac[ind] = 1

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
        # snitch.info("Iteration ", n)

        if threshold is not None:
            if residuals.std() <= threshold:
                break
        
        # Find location of peak in dirty spectrum/
        # peak_loc = np.where(residuals==np.max(residuals))
        peak_loc = np.argmax(np.abs(residuals))
        
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

    outs = { "depths": phi_range, "fdirty": deepcopy(fdirty)}
    
    if clean:
        
        fclean, fcomp, rmtf = rm_clean(lambda_sq, phi_range, fdirty.copy(), 
                    niter=niter, gain=gain)
        # fclean = lexy_rm_clean(lambda_sq, phi_range, fdirty, n_iterations=500, loop_gain=0.1, threshold=None)
        outs.update({"fclean": fclean, "rmtf": rmtf })

    outs = dicto(outs)

    return outs


def read_npz(filename):
    with np.load(filename, allow_pickle=True) as dat:
        datas = dict(dat)
    return datas


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
        description="""This script takes in I, Q and U data and does the process
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
        help="Directory containing the various LoS data files")
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


def read_los_data(filename, compress=True):
    snitch.info(f"Reading in file: {filename}")
    losdata = read_npz(filename)

    filename = os.path.basename(filename)
    reg_num = os.path.splitext(filename)[0].split("_")[-1]

    mask = losdata.pop("mask")
    if compress:
        for key, value in losdata.items():
            # get only data that is not masked
            losdata[key] = np.ma.masked_array(
                data=value, mask=mask).compressed()

    losdata["reg_num"] = reg_num
    losdata = dicto(losdata)

    return losdata


def write_out_rmsynthesis_data(data, idir, depth, odir=None):
    """
    We will dump this data in the same base directory as the input ddata
    """

    if odir is None:
        odir = os.path.abspath(idir)
    # odir += f"-depths-{depth}"
    odir = make_out_dir(odir)

    ofile = fullpath(odir, f"reg_{data.reg_num}")
    np.savez(ofile, **data)
    snitch.info(f"Saved data to: {ofile}")
    
    return ofile


def plot_los_rmdata(los, los_rm, losdata_fname):
    odir = os.path.dirname(losdata_fname) 
    odir = make_out_dir(odir+"-plots")
    ofile = fullpath(odir, f"reg_{los.reg_num}.png")

    plt.close("all")
    fig, ax = plt.subplots(figsize=FIGSIZE, ncols=3)

    ax[0].errorbar(los.lambda_sq, los.fpol, yerr=los.fpol_err, 
                    fmt='o', ecolor="red")
    ax[0].set_xlabel('$\lambda^2$ [m$^{-2}$]')
    ax[0].set_ylabel('Fractional Polarisation')

    
    los.pangle = np.unwrap(los.pangle, period=np.pi, discont=np.pi/2)
    # linear fitting
    res = np.ma.polyfit(los.lambda_sq, los.pangle, deg=1)
    reg_line = np.poly1d(res)(los.lambda_sq)
    
    # ax[1].plot(los.lambda_sq, los.pangle, "r+", label="original")
    ax[1].errorbar(los.lambda_sq, los.pangle, yerr=los.pangle_err,
                    fmt='o', ecolor="red", label="unwrapped angle")
    ax[1].plot(los.lambda_sq, reg_line, "g--",
        label=f"linear fit, slope: {res[0]:.3f}", lw=lw)
    ax[1].set_xlabel('$\lambda^2$ [m$^{-2}$]')
    ax[1].set_ylabel('Polarisation Angle')
    ax[1].legend()


    fclean = np.abs(los_rm.fclean)
    rm_val = los_rm.depths[np.argmax(fclean)]

    ax[2].plot(los_rm.depths, np.abs(los_rm.fdirty),
                'r--', label='Dirty Amp')
    if "fclean" in los_rm:
        ax[2].plot(los_rm.depths, fclean, 'k',
            label=f'Clean Amp, RM {rm_val:.2f}')
        # ax[2].axvline(rm_val, label=f"{rm_val:.3f}")
    ax[2].set_xlabel('Faraday depth [rad m$^{-2}$]')
    ax[2].set_ylabel('Farady Spectrum')
    ax[2].legend(loc='best')

    if "snr" in los:
        snr_idx = np.argmax(los.snr)
        fig.suptitle(
            # f"(||P|| : P$_{{err}}$) SNR$_{{max}}$ = {np.max(los.snr):.2f} " +
            f"(I$_{{los}}$ : I$_{{global\_rms}}$) SNR$_{{max}}$ = {np.max(los.snr):.2f} " +
            f"@ chan = {los.freqs[snr_idx]/1e9:.2f} GHz " +
            f"and $\lambda^{{2}}$ = {los.lambda_sq[snr_idx]:.2f}")
    fig.tight_layout()
    fig.savefig(ofile)
    
    snitch.info(f"Saved Plot at: {ofile}")
    return ofile


def plot_rmtf(los_rm, rmplot_name):
    plt.close("all")
    fig, ax = plt.subplots(figsize=FIGSIZE, ncols=1, squeeze=True)
    
    ax.plot(los_rm.depths, np.abs(los_rm.rmtf), "k-",
        lw=lw, label="Amp")
    ax.plot(los_rm.depths, los_rm.rmtf.real,
        color="orangered", ls="--", lw=lw, label="Real")
    ax.plot(los_rm.depths, los_rm.rmtf.imag, "g:",
        lw=lw, label="Imag")
    ax.set_xlabel(r"Faraday depth $\phi$")
    ax.set_ylabel("RMTF")
    ax.legend()
    fig.tight_layout()
    
    ofile = fullpath(os.path.dirname(rmplot_name), "rmtf.png")
    fig.savefig(ofile)
    snitch.info(f"Saved RMTF plot to: {ofile}")

def rm_and_plot(data_dir, opts=None):
    los = read_los_data(data_dir)
    if los.lambda_sq is None:
        los.lambda_sq = lambda_sq(los.freqs, los.chan_width)
    snitch.info("starting RM")
    rmout = rm_synthesis(
        los.lambda_sq, los.lpol, phi_max=opts.max_fdepth,
        phi_step=opts.depth_step, niter=opts.niters, clean=True)
    
    rmout["reg_num"] = los.reg_num
    
    losdata_fname = write_out_rmsynthesis_data(rmout, idir=data_dir,
        depth=opts.max_fdepth, odir=opts.output_dir)
    
    #plotting everything
    rmplot_name = plot_los_rmdata(los, rmout, losdata_fname)
    return rmplot_name
        


def main():
    opts = arg_parser().parse_args()

    for data_dir in opts.data_dirs:

        # get the various lines of sight in order, doesnt really matter though
        data_files =  sorted(glob(f"{data_dir}/*.npz"), key=os.path.getctime)

        if len(data_files) == 0:
            snitch.info("No los data files found")
            sys.exit()

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(
                partial(rm_and_plot, opts=opts),
                data_files
            ))       

        ##################################################
        # Debug things, do not delete!!
        ##################################################
        
        # for data_file in data_files:
        #     rm_and_plot(data_file, opts)
        
        ##################################################

        # RMTF
        los = read_los_data(data_files[0], compress=False)
        phi_range = np.arange(-opts.max_fdepth, opts.max_fdepth+opts.depth_step,
                            opts.depth_step)
        rmsf_orig = lambda_to_faraday(los.lambda_sq, phi_range, 1)
        rmsf = dicto({"depths": phi_range, "rmtf": rmsf_orig})
        plot_rmtf(rmsf, results[0])
    return
    
if __name__ == "__main__":
    main()