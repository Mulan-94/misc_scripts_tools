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
import matplotlib.gridspec as gridspec
import numpy as np

from scipy import signal
from concurrent import futures
from functools import partial

from utils.genutils import fullpath, make_out_dir, dicto
from utils.logger import logging, LOG_FORMATTER, setup_streamhandler
from utils.rmmath import lambda_sq

snitch = logging.getLogger(__name__)
snitch.addHandler(setup_streamhandler())
snitch.setLevel("INFO")

FIGSIZE = (10,10)
get_wavel = lambda x: 3e8/x
lw = 1.2



def lambda_to_faraday(lambda_sq, phi_range, lpol, derotate=True):
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
    if derotate:
        init_lambda_sq = lambda_sq.mean()
    else:
        init_lambda_sq = 0
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


def rm_clean(lam2, phi_range, fspectrum, niter=500, gain=0.1, derotate=True):
    """
    Clean out the dirty Faraday dispersion measure
    """
    if derotate:
        fwhm = (3.8/ abs(lam2[0]-lam2[-1]))
    else:
        # use the peak of the real component for the gaussian as the CLEAN beam
        # according to rudnick and cotton 2023
        fwhm = 2 / (lam2[-1]+lam2[0])

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

    rmsf_orig = lambda_to_faraday(lam2, phi_range, 1, derotate=derotate) 
    rmsf_fixed = lambda_to_faraday(lam2, phi_pad, 1, derotate=derotate) 
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


def rm_synthesis(lambda_sq, lpol, phi_max=600, phi_step=10, niter=1000, gain=0.1,
    clean=False, derotate=True):
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
    
    fdirty = lambda_to_faraday(lambda_sq, phi_range, lpol, derotate=derotate)

    outs = { "depths": phi_range, "fdirty": deepcopy(fdirty)}
    
    if clean:
        
        fclean, fcomp, rmtf = rm_clean(lambda_sq, phi_range, fdirty.copy(), 
                    niter=niter, gain=gain, derotate=derotate)
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
    parse.add_argument("-np", "--no-plot", dest="plot", action="store_false",
        help="plots for this data? Default is to plot")
    parse.add_argument("-nd", "--no-derotate", dest="no_derotate", action="store_false",
        help="Use this switch to NOT derotate the RMTF by the mean lambda squared.")
    parse.add_argument("-debug", "--debug", dest="debug", action="store_true",
        help="Enable debug mode, will run in serial mode")
    return parse


def read_los_data(filename, compress=True):
    snitch.info(f"Reading in file: {filename}")
    losdata = read_npz(filename)

    filename = os.path.basename(filename)
    reg_num = os.path.splitext(filename)[0].split("_")[-1]

    mask = losdata.pop("mask")
    if compress:
        for key, value in losdata.items():
            if key.lower() == "tag" or value.size==1:
                continue
            # get only data that is not masked
            losdata[key] = np.ma.masked_array(
                data=value, mask=mask).compressed()

    if "lpol" not in losdata:
        losdata["lpol"] = losdata["Q"]/losdata["I"] + 1j*(losdata["U"]/losdata["I"])
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

    ofile = fullpath(odir, f"reg_{data.reg_num}")
    np.savez(ofile, **data)
    snitch.info(f"Saved data to: {os.path.relpath(ofile, '.')}")
    
    return ofile


def plot_los_rmdata(los, los_rm, losdata_fname):
    odir = os.path.dirname(losdata_fname) 
    odir = make_out_dir(odir+"-plots")
    ofile = fullpath(odir, f"reg_{los.reg_num}")
    if los.tag is not None:
        ofile += f"-{los.tag}.png"
    else:
        ofile += ".png"

    plt.close("all")
    # fig, ax = plt.subplots(figsize=FIGSIZE, ncols=3)
    fig, ax = plt.subplot_mosaic([['top left', 'top right'],
                                  ['bot', 'bot']],
        figsize=FIGSIZE, gridspec_kw={"wspace":0})

    ax["top left"].errorbar(los.lambda_sq, los.fpol, yerr=los.fpol_err, 
                    fmt='o', ecolor="red")
    ax["top left"].set_xlabel('$\lambda^2$ [m$^{-2}$]')
    ax["top left"].set_ylabel('Fractional Polarisation')
    ax["top left"].minorticks_on()
    

    
    los.pangle = np.unwrap(los.pangle, period=np.pi, discont=np.pi/2)
    # linear fitting
    res = np.ma.polyfit(los.lambda_sq, los.pangle, deg=1)
    reg_line = np.poly1d(res)(los.lambda_sq)
    
    # ax[1].plot(los.lambda_sq, los.pangle, "r+", label="original")
    ax["top right"].errorbar(los.lambda_sq, los.pangle, yerr=los.pangle_err,
                    fmt='o', ecolor="red", label="unwrapped angle")
    ax["top right"].plot(los.lambda_sq, reg_line, "g--",
        label=f"linear fit, slope: {res[0]:.3f}", lw=lw)
    ax["top right"].set_xlabel('$\lambda^2$ [m$^{-2}$]')
    ax["top right"].set_ylabel('Polarisation Angle')
    ax["top right"].legend()
    ax["top right"].minorticks_on()
    ax["top right"].tick_params(axis='y', which="both", labelright=True, 
        labelleft=False, left=False, right=True)
    ax["top right"].yaxis.set_label_position("right")

    fclean = np.abs(los_rm.fclean)
    rm_val = los_rm.depths[np.argmax(fclean)]

    ax["bot"].plot(los_rm.depths, np.abs(los_rm.fdirty),
                'r--', label='Dirty Amp')
    if "fclean" in los_rm:
        ax["bot"].plot(los_rm.depths, fclean, 'k',
            label=f'Clean Amp, RM {rm_val:.2f}')
        # ax["bot"].axvline(rm_val, label=f"{rm_val:.3f}")
    ax["bot"].set_xlabel('Faraday depth [rad m$^{-2}$]')
    ax["bot"].set_ylabel('Farady Spectrum')
    ax["bot"].legend(loc='best')
    ax["bot"].minorticks_on()

    if "snr" in los:
        snr_idx = np.argmax(los.snr)
        fig.suptitle(
            # f"(||P|| : P$_{{err}}$) SNR$_{{max}}$ = {np.max(los.snr):.2f} " +
            f"(I$_{{los}}$ : I$_{{global\_rms}}$) SNR$_{{max}}$ = {np.max(los.snr):.2f} " +
            f"@ chan = {los.freqs[snr_idx]/1e9:.2f} GHz " +
            f"and $\lambda^{{2}}$ = {los.lambda_sq[snr_idx]:.2f}")
    fig.tight_layout()
    fig.savefig(ofile)
    
    snitch.info(f"Saved Plot at: {os.path.relpath(ofile, '.')}")
    return ofile



def plot_rmtf(los_rm, rmplot_name):
    plt.close("all")
    fig, ax = plt.subplots(figsize=FIGSIZE, ncols=1, squeeze=True)
    
    ax.plot(los_rm.depths, np.abs(los_rm.rmtf), "k-",
        lw=4, label="Amp")
    ax.plot(los_rm.depths, los_rm.rmtf.real,
        color="orangered", ls="--", lw=4, label="Real")
    ax.plot(los_rm.depths, los_rm.rmtf.imag, ":", color="blue",
        lw=4, label="Imag")
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel(r"Faraday depth $\phi$", fontsize=20)
    ax.set_ylabel("RMTF", fontsize=20)
    ax.legend(fontsize=20)
    fig.tight_layout()
    
    ofile = fullpath(os.path.dirname(rmplot_name), "rmtf.pdf")
    fig.savefig(ofile, dpi=300)
    snitch.info(f"Saved RMTF plot to: {ofile}")

def rm_and_plot(data_dir, opts=None, plot=True, odir=None):
    los = read_los_data(data_dir)
    if los.lambda_sq is None:
        los.lambda_sq = lambda_sq(los.freqs, los.chan_width)
    snitch.info("starting RM")
    rmout = rm_synthesis(
        los.lambda_sq, los.lpol, phi_max=opts.max_fdepth,
        phi_step=opts.depth_step, niter=opts.niters, clean=True, derotate=opts.no_derotate)
    
    rmout["reg_num"] = los.reg_num
    rmout["tag"] = los.tag or ""

    losdata_fname = write_out_rmsynthesis_data(rmout, idir=data_dir,
        depth=opts.max_fdepth, odir=odir)
    
    #plotting everything
    if plot:
        rmplot_name = plot_los_rmdata(los, rmout, losdata_fname)
    else:
        rmplot_name = False
    return rmplot_name
        


def main():
    opts = arg_parser().parse_args()

    for data_dir in opts.data_dirs:

        # get the various lines of sight in order, doesnt really matter though
        data_files =  sorted(glob(f"{data_dir}/*.npz"), key=os.path.getctime)

        if len(data_files) == 0:
            snitch.info("No los data files found")
            sys.exit()

        plot = opts.plot

        odir = make_out_dir(opts.output_dir)

        if opts.debug:
            ##################################################
            # Debug things, do not delete!!
            ##################################################
            for data_file in data_files:
                rm_and_plot(data_file, opts, plot=plot, odir=odir)
            
        else:
            with futures.ProcessPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(
                    partial(rm_and_plot, opts=opts, plot=plot, odir=odir),
                    data_files
                ))       

   
        # RMTF
        los = read_los_data(data_files[0], compress=False)
        phi_range = np.arange(-opts.max_fdepth, opts.max_fdepth+opts.depth_step,
                            opts.depth_step)
        rmsf_orig = lambda_to_faraday(los.lambda_sq, phi_range, 1, derotate=opts.no_derotate)
        rmsf = dicto({"depths": phi_range, "rmtf": rmsf_orig})

        if opts.plot:
            plot_rmtf(rmsf, results[0])
    return


def console():
    """A console run entry point for setup.cfg"""
    main()
    snitch.info("Bye :D !")

if __name__ == "__main__":
    console()