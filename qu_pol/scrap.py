"""
References
==========
https://www.extragalactic.info/~mjh/radio-flux.html
https://github.com/mhardcastle/radioflux/blob/master/radioflux/radioflux.py#L95-L109
https://science.nrao.edu/facilities/vla/proposing/TBconv
https://www.eaobservatory.org/jcmt/faq/how-can-i-convert-from-mjybeam-to-mjy/
"""

import argparse
import logging
import os
import regions
import matplotlib.pyplot as plt
import numpy as np
import warnings

# from casatasks import imstat
from astropy.io import fits
from glob import glob
from concurrent import futures
from functools import partial
from time import perf_counter
from itertools import product, chain
from ipdb import set_trace

plt.style.use("bmh")

light_speed = 3e8
marker_size = 10

# ignore overflow errors, assume these to be mostly flagged data
warnings.simplefilter("ignore")

l_handler = logging.FileHandler("xcrapping.log", mode="w")
l_handler.setLevel(logging.DEBUG)

s_handler = logging.StreamHandler()
s_handler.setLevel(logging.INFO)


logging.basicConfig(level=logging.DEBUG,
    datefmt='%H:%M:%S %d.%m.%Y',
    format="%(asctime)s - %(levelname)s - %(message)s", 
    handlers=[l_handler, s_handler])

def make_out_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    return os.path.relpath(dir_name)


def rms(in_data):
    """Root mean square"""
    return np.sqrt(np.square(in_data).mean())


def ssq(in_data):
    """Sum of squares"""
    return np.nansum(np.square(in_data))


def sqrt_ssq(*args):
    """Square root of sum of squares"""
    squares = [np.square(x) for x in args]
    squares = np.sqrt(np.sum(squares, axis=0))
    return squares

def are_all_nan(inp):
    return np.all(np.isnan(inp))

def are_all_zeroes(inp):
    return np.all(inp==0)

def is_infinite(inp):
    return np.isinf(inp)


def linear_polzn(stokes_q, stokes_u):
    return sqrt_ssq(stokes_q, stokes_u)


def fractional_polzn(stokes_i, stokes_q, stokes_u):
    linear_pol = linear_polzn(stokes_q, stokes_u)
    frac_pol = linear_pol / stokes_i
    return frac_pol


def timer(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        logging.info(f"'{func.__name__}' run in: {perf_counter()-start:.2f} sec")
        return result
    return wrapper


# ==========================================================================

def get_useful_data(fname):
    data = {}
    with fits.open(fname) as hdul:
        # data["bmaj"] = hdul[0].header["BMAJ"]
        # data["bmin"] = hdul[0].header["BMIN"]
        data["freqs"] = hdul[0].header["CRVAL3"]
        # data["waves"] = lambda_sq(data["freqs"])
        data["fnames"] = fname
        data["data"] = hdul[0].data.squeeze()
        data["freqs"] /= 1e9
    return data


def extract_stats2(fname, reg, noise_reg, sig_factor=10):
    # get image data
    im_data = get_useful_data(fname)
    data = im_data.pop("data")

    # Get image noise from standard deviation of a sourceless region
    noise_cut = get_data_cut(noise_reg, data)

    noise = np.nanstd(noise_cut)

    # global noise sigma
    # glob_noise = np.nanstd(data)
    
    # Get mean flux over the pixels
    intense_cut = get_data_cut(reg, data)


    if (are_all_nan(intense_cut) or are_all_zeroes(intense_cut) or
        is_infinite(noise)):
        # skip all the nonsence if all the data is Nan
        logging.debug(f"Skipping region:{reg.meta['label']} {fname} because NaN/Zeroes/inf ")
        im_data["flux_jybm"] = im_data["flux_jy"] = None
        return im_data

    # flux per beam sum
    flux_jybm = np.nansum(get_data_cut(reg, data))

    # flux_jybm = np.nanmean(intense_cut)

    # flux in jansky
    # flux_jy = get_flux(hdu_header, flux_jybm)
    flux_jy = None
    
    if (flux_jybm > sig_factor * noise) and flux_jybm < 0.5e3:
        im_data["flux_jybm"] = flux_jybm
        im_data["flux_jy"] = flux_jy
    else:
        im_data["flux_jybm"] = im_data["flux_jy"] = None

    return im_data

# ==========================================================================

def get_flux(header, flux_sum):
    bmaj = np.abs(header["BMAJ"])
    bmin = np.abs(header["BMIN"])
    cdelt1 = header["CDELT1"]
    cdelt2 = header["CDELT2"]
    # from definitions of FWHM
    gfactor = 2.0 * np.sqrt(2.0 * np.log(2.0))
    beam_area = np.abs((2 * np.pi * (bmaj/cdelt1) * (bmin/cdelt2)) / gfactor**2)
    if flux_sum>0:
        flux = flux_sum / beam_area

    else:
        flux = None
    return flux


def get_box_dims(reg):
    # blc: bottom left corner, trc: top right corner
    # blc_x, trc_x, blc_y, trc_y
    box = ",".join(
        [str(getattr(reg.bounding_box, x)) 
            for x in "ixmin iymin ixmax  iymax".split()])
    return box


def get_imstat_box_dims(reg):
    """
    blc: bottom left corner, trc: top right corner
    This function is to deal with the off by -1 issue with casa's
    imstats TRC argument. The problem was that while using imstats and doing
    a box selection, setting the TRC value to eg (100, 100) would lead to the 
    selection of pixels upto (101, 101). So for example, I want pixels btwn
    blc(0,90) and trc(2, 92), I would expect to get back stats for 4=(2x2) pixels
    however, I get back stats for 9 pixels because 9=(3x3)

    I therefore reduce the trc values by 1 and return the values in the form:
        blc_x, blc_y, trc_x, trc_y
    """
    box = []
    for x in "ixmin iymin ixmax  iymax".split():
        val = getattr(reg.bounding_box, x)
        if "max" in x:
            val -= 1
        box.append(str(val))
    return ",".join(box)


def get_data_cut(reg, data):
    reg_mask = reg.to_mask()
    data_cut = reg_mask.cutout(data)
    return data_cut

@timer
def get_image_stats2(stokes, file_core, images, regs, noise_reg):
    out_dir = make_out_dir(f"{stokes}-{file_core}")
    fluxes, waves = [], []
    logging.info("starting get_image_stats")
    for reg in regs:
        logging.info(f"Region: {reg.meta['label']}")
        with futures.ProcessPoolExecutor(max_workers=16) as executor:
            results = executor.map(
                partial(extract_stats2, reg=reg, noise_reg=noise_reg, sig_factor=10), images
                )
        
        # some testbed
        # results = []
        # for im in images:
        #     if "37b-QU-for-RM-1024-1279-0254-Q-image.fits" not in im:
        #         continue
        #     results.append(extract_stats2(im, reg=reg, noise_reg=noise_reg, sig_factor=10))


        # Q and U for each freq
        outs = {"flux_jy": [],"flux_jybm": [],"waves": [],"freqs": [],"fnames": []}
        outfile = os.path.join(out_dir, f"{reg.meta['label']}_{stokes}")


        for res in results:           
            outs["flux_jy"].append(res["flux_jy"])
            outs["flux_jybm"].append(res["flux_jybm"])
            # outs["waves"].append(res["waves"])
            outs["freqs"].append(res["freqs"])
            outs["fnames"].append(res["fnames"])

        np.savez(outfile, **outs)

    logging.info(f"Stokes {stokes} done")
    logging.info("---------------")
    return


def generate_regions(reg_fname, factor=50, max_w=572, max_h=572):
    """
    Create a DS9 region file containing a bunch of regions
    factor: int
        In my case, the size of the region ie length / widh t
        reg_fname:
            File name for the resulting region files
        max_*:
            Maximum pixel image height or width. 
            So that regions don't go beyound image dims
    """
    # left to right
    width_range = 74, 569

    # bottom to top
    height_range = 190, 450
    header = [
        "# Region file format: DS9 CARTA 2.0.0",
        ('global color=green dashlist=8 3 width=1 ' +
        'font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 ' +
        'edit=1 move=1 delete=1 include=1 source=1'),
        "physical"
    ]

    pts = []
    count = 0
    for height in range(*height_range, factor):
        for width in range(*width_range, factor):
            # pts.append("circle({}, {}, {}) # color=#2EE6D6 width=2".format(width, height+factor, factor/2))
            width = max_h if width > max_w else width
            height = max_h if height > max_h else height
            pts.append(
                "box({}, {}, {}, {}) # color=#2EE6D6 width=2 text={{reg_{}}}".format(
                    width, height, factor, factor, count))
            count += 1

    if ".reg" not in reg_fname:
        reg_fname += f"-{factor}.reg"

    if not os.path.isfile(reg_fname):
        with open(reg_fname, "w") as fil:
            pts = [p+"\n" for p in header+pts]
            fil.writelines(pts)

        logging.info(f"Regions file written to: {reg_fname}")
    else:
        logging.info(f"File already available at: {reg_fname}")
    return reg_fname


def lambda_sq(freq_ghz):
    #speed of light in a vacuum
    global light_speed
    freq_ghz = freq_ghz * 1e9
    # frequency to wavelength
    wave = light_speed/freq_ghz
    return np.square(wave)


def freq_sqrt(lamsq):
    #speed of light in a vacuum
    global light_speed

    lamsq = np.sqrt(lamsq)
    # frequency to wavelength
    freq_ghz = (light_speed/lamsq)/1e9
    return freq_ghz


def format_lsq(inp, func):
    """
    Converting and formating output
    Funxtions expdcted lambda_sq, and freq_sqrt
    """
    inp = func(inp)
    return [float(f"{_:.2f}") for _ in inp]


def active_legends(fig):
    legs = { _.get_legend_handles_labels()[-1][0] for _ in fig.axes
             if len(_.get_legend_handles_labels()[-1])>0 }
    return list(legs)


def create_figure(grid_size, fsize=(20, 10), sharex=True, sharey=False):
    fig, sp = plt.subplots(*grid_size, sharex=sharex, sharey=sharey,
        # gridspec_kw={"wspace": 0, "hspace": 1}, 
        figsize=fsize, dpi=200)
    # plt.figure(figsize=fsize)
    return fig, sp



def plot_spectra(file_core, outfile, xscale="linear"):
    """
    file_core: str
        core of the folders where the data are contained
    outfile: str
        prefix name of the output file
    """
    # r: red, b: blue, k: black
    colours = {
        "Q": "r2", "U": "b1", "I": "ko", 
        "poln_power": "g+", "frac_poln":"mx"}
    fight = lambda x: int(os.path.basename(x).split("_")[1])

    q_files = sorted(glob(f"./Q-{file_core}/*.npz"), key=fight)
    u_files = sorted(glob(f"./U-{file_core}/*.npz"), key=fight)
    i_files = sorted(glob(f"./I-{file_core}/*.npz"), key=fight)

    qui_files = list(zip(q_files, u_files, i_files))
    n_qf = len(q_files)
    logging.info(f"Found {n_qf} QUI files")

    # rationale is a 4:3 aspect ratio, max is this value x 3 = 12:9
    # toget respsective sizes, add length+width and mult by ratio
    rows = 9 if n_qf > 108 else int(np.ceil(3/7*(np.sqrt(n_qf)*2)))
    cols = 12 if n_qf > 108 else int(np.ceil(4/7*(np.sqrt(n_qf)*2)))
    grid_size_sq = rows*cols

    logging.info("Starting plots")
    plt.close("all")
    for i, files in enumerate(qui_files):
        if i % grid_size_sq == 0:
            fig, sp = create_figure((rows, cols), fsize=(50, 30), sharey=False)
            rc = product(range(rows), range(cols))

        row, col = next(rc)
        polns = {}
    
        for stokes in files:
            reg_name = os.path.splitext(os.path.basename(stokes))[0].split("_")
            c_stoke = reg_name[-1]
            specs = {k:v for k,v in zip(["c","marker"], colours[c_stoke])}
            specs.update(dict(s=marker_size*4.1, label=c_stoke, alpha=0.4))

            # logging.info(f"Reg {reg_name[1]}, Stokes {stokes}")
            
            with np.load(stokes, allow_pickle=True) as data:
                # these frequencies are already in GHZ
                # flip so that waves increaase
                freqs = np.flip(data["freqs"])
                polns[c_stoke] = np.flip(data["flux_jybm"].astype(float))
            
            waves = lambda_sq(freqs)
    
            # sp[row, col].scatter(waves, polns[c_stoke], **specs)
        
        
        sp[row, col].set_title(f"Reg {reg_name[1]}", y=1.0, pad=-20, size=9)
        sp[row, col].set_xscale(xscale)
        sp[row, col].set_yscale(xscale)
        sp[row, col].xaxis.set_tick_params(labelbottom=True)

        del specs["label"]
        # # for power plots
        # polns["poln_power"] = linear_polzn(polns["Q"], polns["U"])
        # specs.update({k:v for k,v in zip(["c","marker"], colours["poln_power"])})
        # sp[row, col].scatter(waves, polns["poln_power"], label="poln_power", **specs)
        
        # for fractional polarization
        polns["frac_poln"] = fractional_polzn(polns["I"], polns["Q"], polns["U"])
        specs.update({k:v for k,v in zip(["c","marker"], colours["frac_poln"])})
        
        sp[row, col].scatter(waves, polns["frac_poln"], label="frac_poln", **specs)

        # adding in the extra x-axis for wavelength
        new_ticklocs = np.linspace((1.2*waves.min()), (0.9*waves.max()), 8)
        ax2 = sp[row, col].twiny()
        ax2.set_xlim(sp[row, col].get_xlim())
        ax2.set_xticks(new_ticklocs)
        ax2.set_xticklabels(format_lsq(new_ticklocs, freq_sqrt))
        ax2.tick_params(axis="x",direction="in", pad=-15)

        if row/rows == 0:
            plt.setp(ax2, xlabel="Freq GHz")

        if np.prod((i+1)%grid_size_sq==0 or (n_qf<grid_size_sq and i==n_qf-1)):
            # Remove empties
            empties = [i for i, _ in enumerate(sp.flatten()) if (not _.lines) and (not _.collections)]
            for _ in empties:
                fig.delaxes(sp.flatten()[_])
            
            logging.info(f"Starting the saving process: Group {int(i/grid_size_sq)}")
            fig.tight_layout(h_pad=3)
            legs = active_legends(fig)
            fig.legend(legs, bbox_to_anchor=(1, 1.01), markerscale=3, ncol=len(legs))

            # fig.suptitle("Q and U vs $\lambda^2$")
            oname = f"{outfile}-{int(i/grid_size_sq)}-{xscale}"
            
            plt.setp(sp[:,0], ylabel="Frac Pol")
            plt.setp(sp[-1,:], xlabel="Wavelength m$^2$")
    
            fig.savefig(oname, bbox_inches='tight')
            plt.close("all")
            logging.info(f"Plotting done for {oname}")

def read_sorted_filnames(fname):
    with open(fname, "r") as post:
        items = post.readlines()
        items = [_.replace("\n", "") for _ in items]
    return items

def parser():
    parsing = argparse.ArgumentParser()
    parsing.add_argument("-rs", "--region-size", dest="reg_size", nargs="+",
        type=int, default=[], metavar="", 
        help="Create regions of this pixel size and perform analyses on them")
    parsing.add_argument("-ap", "--auto-plot", dest="auto_plot", action="store_true",
        help="Plot all the specified region pixels")
    parsing.add_argument("-p", "--plot", dest="plot", nargs="*", type=int, 
        metavar="",
        help="Make plots for these region sizes manually. These will be linearly scaled")
    parsing.add_argument("-ps", "--plot-scales", dest="plot_scales", metavar="",
        default=["linear"], nargs="*", choices=["linear", "log"],
        help="Scales for the plots. Can be a space separated list of different scales.")
    parsing.add_argument("-t", "--testing", dest="testing", metavar="", type=str,
        help="Testing prefix. Will be Prepended with '-'", default=None)
    return parsing


if __name__ == "__main__":
    opts = parser().parse_args()

    # for factor in [70, 50, 10, 7, 5, 3]:
    if opts.testing is None:
        testing = ""
    else:
        testing = "-" + opts.testing

    if opts.reg_size:
        sortkey = lambda x: int(os.path.basename(x).split("-")[0])

        """
        Making sorted files using the following procedure
        # Create a list of all the Q, U folder contents only
        1. ls -rt channelised/*-*[0-9]/*[0-9][0-9][0-9][0-9]-*image* > post.txt

        # OPen the file with sublime and
        1b. Delete all MFS files listed
        2. Find all (using search bar ctrl +H) with Q-image.fits

        3. select all and modify to the fname of I files
        4. Change the containing dir for the i files because they end in '-I'. Done using regex
            find: 
                channelised/(\d*)-(\d*)/37b-QU-for-RM-(\d*)-(\d*)-I-(\d{4})-image.fits
            replace: 
                channelised/$1-$2-I/37b-QU-for-RM-$3-$4-I-$5-image.fits
        THEY WILL  BE IN POST.TXT
        """

        for factor in opts.reg_size:
            start = perf_counter()
            file_core = f"regions-mpc-{factor}{testing}"

            reg_file = generate_regions(f"regions/beacons", factor=factor)
            regs = regions.Regions.read(reg_file, format="ds9")

            noise_reg = regs.pop(-1)
            for stokes in "I Q U".split():
                
                if stokes != "I":
                    images = sorted(glob(f"./channelised/*-*"), key=sortkey)
                    sstring = f"/*[0-9][0-9][0-9][0-9]-{stokes}-*image*"
                else:
                    images = sorted(glob(f"./channelised/*-*-I"), key=sortkey)
                    sstring = f"/*{stokes}-[0-9][0-9][0-9][0-9]-*image*"
                
                images = list(chain.from_iterable([sorted(glob(im+sstring)) for im in images]))
                # images = read_sorted_filnames("post.txt")
                
                logging.info(f"Working on Stokes {stokes}")
                logging.info(f"With {len(regs)} regions")
                logging.info(f"And {len(images)} images")
                
                bn = get_image_stats2(stokes, file_core, images, regs, noise_reg)

            if opts.auto_plot:
                logging.info("Autoplotting is enabled")
                plot_dir = make_out_dir(f"plots-QU-{file_core}")
                pout = os.path.join(plot_dir,  f"QU-{file_core}")
                plot_spectra(file_core, f"{plot_dir}/QU-regions-mpc{testing}{factor}")

            logging.info(f"Finished factor {factor} in {perf_counter() - start} seconds")
            logging.info("======================================")

    if opts.plot:
        logging.info(f"Plotting is enabled for regions {opts.plot}")
        for factor in opts.plot:
            for scale in opts.plot_scales:
                file_core = f"regions-mpc-{factor}{testing}"
                plot_dir = make_out_dir(f"plots-QU-{file_core}")
                pout = os.path.join(plot_dir,  f"QU-{file_core}")
                plot_spectra(file_core, f"{plot_dir}/QU-regions-mpc{testing}{factor}", xscale=scale)