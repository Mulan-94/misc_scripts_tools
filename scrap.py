import os
import regions
import matplotlib.pyplot as plt
import numpy as np

from casatasks import imstat
from astropy.io import fits
from glob import glob
from concurrent import futures
from functools import partial
from time import perf_counter
from itertools import product


from ipdb import set_trace


def get_box_dims(reg):
    # blc: bottom left corner, trc: top right corner
    # blc_x, trc_x, blc_y, trc_y
    # print(reg.bounding_box)
    box = ",".join(
        [str(getattr(reg.bounding_box, x)) 
            for x in "ixmin iymin ixmax  iymax".split()])
    return box


def extract_stats(fname, reg, noise_reg, sig_factor=10):    
    stats = imstat(imagename=fname,box=get_box_dims(reg))
   
    if "flux" not in stats:
        return None, None
    
    noise_stats = imstat(imagename=fname,box=get_box_dims(noise_reg))
    noise_std = noise_stats["sigma"]
    flux = stats["flux"][0]
    snr = flux/noise_std

    glob_sigma = imstat(imagename=fname)["sigma"]        

    if snr > sig_factor * glob_sigma:
        with fits.open(fname) as hdul:
            freq = hdul[0].header["CRVAL3"]
        # reg.meta["label"]
        waves = (1/freq**2)
        return flux, waves
    else:
        return None, None


def make_out_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    return os.path.relpath(dir_name)


def create_figure(grid_size, fsize=(20, 10)):
    fig, sp = plt.subplots(*grid_size, sharex=True, sharey=False,
        gridspec_kw={"wspace": 0, "hspace": 0}, figsize=fsize, dpi=200)
    # plt.figure(figsize=fsize)
    return fig, sp


def plot_spectra(file_core, outfile):
    colours = {"Q": "r", "U": "b"}
    fight = lambda x: int(os.path.basename(x).split("_")[1])

    q_files = sorted(glob(f"./Q-{file_core}/*.npz"), key=fight)
    u_files = sorted(glob(f"./U-{file_core}/*.npz"), key=fight)
    qu_files = list(zip(q_files, u_files))
    n_qf = len(q_files)
    rows = int(np.ceil(np.sqrt(n_qf))) if n_qf < 100 else 7
    cols = rows + 2
    grid_size_sq = rows*cols

    for i, files in enumerate(qu_files):

        if i % grid_size_sq == 0:
            fig, sp = create_figure((rows, cols), fsize=(50, 30))
            rc = product(range(rows), range(cols))

        row, col = next(rc)

        # print(f"row: {row}, col: {col}")
        for stokes in files:
            reg_name = os.path.splitext(os.path.basename(stokes))[0].split("_")
            c_stoke = reg_name[-1]
            with np.load(stokes) as data:
                flux = data["flux"]
                waves = data["waves"]
            
            sp[row, col].plot(waves, flux, f"{colours[c_stoke]}o", markersize=3, label=c_stoke)
            sp[row, col].set_title(f"Reg {reg_name[1]}", y=1.0, pad=-14, size=9)
            sp[row, col].set_xscale("linear")
        
        if np.prod((row+1, col+1)) == grid_size_sq:
            fig.tight_layout()
            fig.legend(["Q", "U"], bbox_to_anchor=(1, 1.01), markerscale=3, ncol=2)
            fig.savefig(f"{outfile}-{int(i/grid_size_sq)}")
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

    print(f"Regions file written to: {reg_fname}")
    return reg_fname

def get_image_stats(stokes, file_core, images, regs):
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

        # for fname in images:
        #     res = extract_stats(fname, reg, noise_reg, sig_factor=10)
        #     if None not in res:
        #         fluxes.append(res[0])
        #         waves.append(res[1])

        np.savez(os.path.join(out_dir, f"{reg.meta['label']}_{stokes}"), 
            flux=fluxes, waves=waves)

    print(f"Stokes {stokes} done")
    print("---------------")
    return


# [3, 5, 7, 9, 11]
# [3, 5, 7]:
for factor in [3, 5, 7, 10, 12, 14, 16]:

    start = perf_counter()

    file_core = f"regions-mpc-{factor}"

    reg_file = generate_regions(f"regions/beacons", factor=factor)
    # regs = regions.Regions.read(reg_file, format="ds9")

    # noise_reg = regs.pop(-1)

    # for stokes in "Q U".split():
    #     images = sorted(glob(f"./channelised/*/*00*{stokes}*image*"))
        
    #     print(f"Working on Stokes {stokes}")
    #     print(f"With {len(regs)} regions")
        
    #     get_image_stats(stokes, file_core, images, regs)

    # plot_spectra(file_core, f"QU-{file_core}")

    print(f"Finished factor {factor} in {perf_counter() - start} seconds")
    print("======================================")
