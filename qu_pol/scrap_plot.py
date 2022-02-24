#### For plotting
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from ipdb import set_trace
from itertools import product
from astropy.io import fits


def beam_area(fname):
    with fits.open(fname) as hdul:
        header = hdul[0]
    bmaj = np.abs(header["BMAJ"])
    bmin = np.abs(header["BMIN"])
    cdelt1 = header["CDELT1"]
    cdelt2 = header["CDELT2"]
    # from definitions of FWHM
    gfactor = 2.0 * np.sqrt(2.0 * np.log(2.0))
    area = np.abs((2 * np.pi * (bmaj/cdelt1) * (bmin/cdelt2)) / gfactor**2)

    return area


def sqrt_ssq(*args):
    """Square root of sum of squares"""
    squares = [np.square(x) for x in args]
    squares = np.sqrt(np.sum(squares, axis=0))
    return squares

def create_figure(grid_size, fsize=(20, 10)):
    fig, sp = plt.subplots(*grid_size, sharex=True, sharey=False,
        gridspec_kw={"wspace": 0, "hspace": 0}, figsize=fsize, dpi=200)
    # plt.figure(figsize=fsize)
    return fig, sp

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

            
def plot_power(file_core, outfile, xscale="linear"):
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
    rows = int(np.ceil(np.sqrt(n_qf))) if n_qf < 100 else 7
    cols = rows + 2
    grid_size_sq = rows*cols

    print("Starting plots")
    for i, files in enumerate(qu_files):

        if i % grid_size_sq == 0:
            fig, sp = create_figure((rows, cols), fsize=(50, 30))
            rc = product(range(rows), range(cols))

        row, col = next(rc)

        # print(f"row: {row}, col: {col}")
        powers = {}
        # print(files)
        for stokes in files:
            reg_name = os.path.splitext(os.path.basename(stokes))[0].split("_")
            c_stoke = reg_name[-1]
            with np.load(stokes, allow_pickle=True) as data:
                flux = data["flux"]
                waves = data["waves"]
            powers[c_stoke] = flux.astype(float)
        power = sqrt_ssq(powers["Q"], powers["U"])
        sp[row, col].plot(waves, power, f"go", markersize=3, label="power")
        sp[row, col].set_title(f"Reg {reg_name[1]}", y=1.0, pad=-14, size=9)
        sp[row, col].set_xscale(xscale)
        
        if np.prod((row+1, col+1)) == grid_size_sq or (n_qf<grid_size_sq and i==n_qf-1):
            fig.tight_layout()
            fig.legend(["power"], bbox_to_anchor=(1, 1.01), markerscale=3, ncol=3)
            fig.savefig(f"{outfile}-power-{int(i/grid_size_sq)}", bbox_inches='tight')
            plt.close("all")
            print(f"Plotting done for {outfile}-power-{int(i/grid_size_sq)}")



if __name__ == "__main__":
    for i in [50]:
        plot_spectra(f"regions-mpc-test-{i}", f"QU-regions-mpc-{i}")
        # plot_spectra(f"regions-mpc-{i}", f"QU-regions-mpc-{i}-logx", xscale="log")
    
    # plot_spectra("regions-mpc-50-optest", "QU-regions-mpc-50-optest")
    # plot_power("regions-mpc-50", "QU-regions-mpc-50")