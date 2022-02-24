"""
References
==========
https://www.extragalactic.info/~mjh/radio-flux.html
https://github.com/mhardcastle/radioflux/blob/master/radioflux/radioflux.py#L95-L109
https://science.nrao.edu/facilities/vla/proposing/TBconv
https://www.eaobservatory.org/jcmt/faq/how-can-i-convert-from-mjybeam-to-mjy/
"""
import logging
import os
import regions
import matplotlib.pyplot as plt
import numpy as np

# from casatasks import imstat
from astropy.io import fits
from glob import glob
from concurrent import futures
from functools import partial
from time import perf_counter
from itertools import product, chain
from ipdb import set_trace

light_speed = 3e8
marker_size = 10

logging.basicConfig(level=logging.INFO, 
    format="%(levelname)s %(message)s", 
    filename="test_option.log", filemode="w")


def timer(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        print(f"'{func.__name__}' run in: {perf_counter()-start:.2f} sec")
        return result
    return wrapper


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


def flux_sum_from_flux(fname, flux):
    with fits.open(fname) as hdul:
        header = hdul[0]
    bmaj = np.abs(header["BMAJ"])
    bmin = np.abs(header["BMIN"])
    cdelt1 = header["CDELT1"]
    cdelt2 = header["CDELT2"]
    # from definitions of FWHM
    gfactor = 2.0 * np.sqrt(2.0 * np.log(2.0))
    beam_area = np.abs((2 * np.pi * (bmaj/cdelt1) * (bmin/cdelt2)) / gfactor**2)

    return flux * beam_area


def get_box_dims(reg):
    # blc: bottom left corner, trc: top right corner
    # blc_x, trc_x, blc_y, trc_y
    # print(reg.bounding_box)
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


def rms(in_data):
    """Root mean square"""
    return np.sqrt(np.square(in_data).mean())

def ssq(in_data):
    """Sum of squares"""
    return np.nansum(np.square(in_data))

def sqrt_ssq(*args):
    """Square root of sum of squares"""
    squares = [np.square(x) for x in args]
    squares = np.sqrt(np.nansum(squares, axis=0))
    return squares

def extract_stats(fname, reg, noise_reg, sig_factor=10):
    # """
    # fname: image name
    # reg: region of interest
    # noise_reg: the noise region
    # sig_factor: z-score?
    # """
    # stats = imstat(imagename=fname,box=get_box_dims(reg))
   
    # if "flux" not in stats:
    #     return None, None
    
    # noise_stats = imstat(imagename=fname, box=get_box_dims(noise_reg))
    # noise_std = noise_stats["sigma"]
    # flux = stats["flux"][0]
    # snr = flux/noise_std

    # glob_sigma = imstat(imagename=fname)["sigma"]        

    # if snr > sig_factor * glob_sigma:
    #     freq = float(stats["blcf"].split(",")[-2].strip()[:-2])
    #     waves = (light_speed/freq)**2
    #     return flux, waves
    # else:
        # return None, None
    pass


def get_data_cut(reg, data):
    reg_mask = reg.to_mask()
    data_cut = reg_mask.cutout(data)
    return data_cut


def extract_stats2(fname, reg, noise_reg, sig_factor=10):
    with fits.open(fname) as hdul:
        hdu_data = hdul[0].data.squeeze().astype(np.float64)
        hdu_header = hdul[0].header

    data_cut = get_data_cut(reg, hdu_data)
    data_flux = np.nansum(data_cut)
    data_flux_bm = get_flux(hdu_header, data_flux)
    waves = None
    freqs = None
    if data_flux is not None and data_flux !=0 :
        glob_sigma = np.nanstd(hdu_data)
        noise_sigma = np.nanstd(get_data_cut(noise_reg, hdu_data))
        snr = data_flux/noise_sigma
        
        if snr > sig_factor * glob_sigma:
            freqs = hdu_header["CRVAL3"]
            waves = (light_speed/freqs)**2
        else:
            data_flux = None
            data_flux_bm = None
   
    logging.info(f"flux: {data_flux}, waves: {waves} ,freqs: {freqs}, {fname}")
    return data_flux, data_flux_bm, waves, freqs, fname

@timer
def get_image_stats2(stokes, file_core, images, regs, noise_reg):
    out_dir = make_out_dir(f"{stokes}-{file_core}")
    fluxes, waves = [], []
    logging.info("starting get_image_stats")
    for reg in regs:
        logging.info(f"Region: {reg.meta['label']}")
        with futures.ProcessPoolExecutor(max_workers=70) as executor:
            results = executor.map(
                partial(extract_stats2, reg=reg, noise_reg=noise_reg, sig_factor=10), images
                )
        
        # Q and U for each freq
        results = list(results)
        fluxes = [res[0] for res in results]
        fluxes_bm = [res[1] for res in results]
        waves = [res[2] for res in results]
        freqs = [res[3] for res in results]
        fnames = [res[4] for res in results]

        np.savez(os.path.join(out_dir, f"{reg.meta['label']}_{stokes}"), 
            flux=fluxes, fluxes_bm=fluxes_bm, waves=waves, freqs=freqs, fnames=fnames)

    logging.info(f"Stokes {stokes} done")
    logging.info("---------------")
    return


def make_out_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    return os.path.relpath(dir_name)


def create_figure(grid_size, fsize=(20, 10)):
    fig, sp = plt.subplots(*grid_size, sharex=False, sharey=False,
        gridspec_kw={"wspace": 0, "hspace": 0}, figsize=fsize, dpi=200)
    return fig, sp



@timer
def plot_spectra(file_core, outfile, xscale="linear"):
    """
    file_core: str
        core of the folders where the data are contained
    outfile: str
        prefix name of the output file
    """
    colours = {"Q": "r", "U": "b"}
    fight = lambda x: int(os.path.basename(x).split("_")[1])

    q_files = sorted(glob(f"./Q-{file_core}/*.npz"), key=fight)
    u_files = sorted(glob(f"./U-{file_core}/*.npz"), key=fight)

    qu_files = list(zip(q_files, u_files))
    n_qf = len(q_files)

    # rationale is a 4:3 aspect ratio, max is this value x 3 = 12:9
    # toget respsective sizes, add length+width and mult by ratio
    rows = 9 if n_qf > 108 else int(np.ceil(3/7*(np.sqrt(n_qf)*2)))
    cols = 12 if n_qf > 108 else int(np.ceil(4/7*(np.sqrt(n_qf)*2)))
    grid_size_sq = rows*cols

    print("Starting plots")
    for i, files in enumerate(qu_files):
        if i % grid_size_sq == 0:
            fig, sp = create_figure((rows, cols), fsize=(50, 30))
            rc = product(range(rows), range(cols))

        row, col = next(rc)
        powers = {}
        for stokes in files:
            reg_name = os.path.splitext(os.path.basename(stokes))[0].split("_")
            c_stoke = reg_name[-1]
            with np.load(stokes, allow_pickle=True) as data:
                powers[c_stoke] = data["flux"].astype(float)
                waves = data["waves"].astype(float)

            sp[row, col].plot(waves, powers[c_stoke], f"{colours[c_stoke]}o", markersize=marker_size, label=c_stoke, alpha=0.4)
            sp[row, col].set_title(f"Reg {reg_name[1]}", y=1.0, pad=-14, size=9)
            sp[row, col].set_xscale(xscale)
            sp[row, col].set_yscale(xscale)

        # for power plots
        power = sqrt_ssq(powers["Q"], powers["U"])
        sp[row, col].plot(waves, power, f"g+", markersize=marker_size, label="power", alpha=0.5)
        
        if np.prod((row*col)+1==grid_size_sq or (n_qf<grid_size_sq and i==n_qf-1)):
            fig.tight_layout()
            fig.legend(["Q", "U", "power"], bbox_to_anchor=(1, 1.01), markerscale=3, ncol=3)
            # fig.suptitle("Q and U vs Lambda**2")
            fig.savefig(f"{outfile}-{int(i/grid_size_sq)}", bbox_inches='tight')
            plt.close("all")
            print(f"Plotting done for {outfile}-{int(i/grid_size_sq)}")


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
        'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
        "physical"
    ]

    pts = []
    count = 0
    for height in range(*height_range, factor):
        for width in range(*width_range, factor):
            # pts.append("circle({}, {}, {}) # color=#2EE6D6 width=2".format(width, height+factor, factor/2))
            width = max_h if width > max_w else width
            height = max_h if height > max_h else height
            pts.append("box({}, {}, {}, {}) # color=#2EE6D6 width=2 text={{reg_{}}}".format(width, height, factor, factor, count))
            count += 1

    if ".reg" not in reg_fname:
        reg_fname += f"-{factor}.reg"

    with open(reg_fname, "w") as fil:
        pts = [p+"\n" for p in header+pts]
        fil.writelines(pts)

    logging.info(f"Regions file written to: {reg_fname}")
    return reg_fname

"""
def get_image_stats(stokes, file_core, images, regs, noise_reg):
    out_dir = make_out_dir(f"{stokes}-{file_core}")
    fluxes, waves = [], []
    for reg in regs:
        print(f"Region: {reg.meta['label']}")

        with futures.ProcessPoolExecutor(max_workers=70) as executor:
            results = executor.map(
                partial(extract_stats, reg=reg, noise_reg=noise_reg), images
                )


        # Q and U for each freq
        results = list(results)
        fluxes = [res[0] for res in results if res[0] is not None]
        waves = [res[1] for res in results if res[1] is not None]

        np.savez(os.path.join(out_dir, f"{reg.meta['label']}_{stokes}"), 
            flux=fluxes, waves=waves)

    print(f"Stokes {stokes} done")
    print("---------------")
    return True
"""

# [3, 5, 7, 9, 11]
# [3, 5, 7, 10, 50]:
# for factor in [50, 10, 7, 5, 3]:


sortkey = lambda x: int(os.path.basename(x).split("-")[0])

for factor in [70, 50, 10, 7, 5, 3]:

    start = perf_counter()

    file_core = f"regions-mpc-{factor}"

    reg_file = generate_regions(f"regions/beacons", factor=factor)
    regs = regions.Regions.read(reg_file, format="ds9")

    noise_reg = regs.pop(-1)

    for stokes in "Q U".split():
        images = sorted(glob(f"./channelised/*-*"), key=sortkey)
        sstring = f"/*[0-9][0-9][0-9][0-9]-{stokes}-*image*"
        images = list(chain.from_iterable([sorted(glob(im+sstring)) for im in images]))
        
          
        print(f"Working on Stokes {stokes}")
        print(f"With {len(regs)} regions")
        print(f"And {len(images)} images")
        
        bn = get_image_stats2(stokes, file_core, images, regs, noise_reg)

    plot_dir = make_out_dir(f"plots-{stokes}-{file_core}")
    pout = os.path.join(plot_dir,  f"QU-{file_core}")

    plot_spectra(file_core, pout+"linear", xscale="linear")
    plot_spectra(file_core, pout+"log", xscale="log")

    print(f"Finished factor {factor} in {perf_counter() - start} seconds")
    print("======================================")
