#### For plotting
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from ipdb import set_trace
from itertools import product
from astropy.io import fits

plt.style.use("bmh")

l_handler = logging.FileHandler("xcrapping.log", mode="a")
l_handler.setLevel(logging.DEBUG)
s_handler = logging.StreamHandler()
s_handler.setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG,
    datefmt='%H:%M:%S %d.%m.%Y',
    format="%(asctime)s - %(levelname)s - %(message)s", 
    handlers=[l_handler, s_handler])


marker_size = 6

def sqrt_ssq(*args):
    """Square root of sum of squares"""
    squares = [np.square(x) for x in args]
    squares = np.sqrt(np.sum(squares, axis=0))
    return squares


def linear_polzn(stokes_q, stokes_u):
    return sqrt_ssq(stokes_q, stokes_u)


def fractional_polzn(stokes_i, stokes_q, stokes_u):
    linear_pol = linear_polzn(stokes_q, stokes_u)
    frac_pol = linear_pol / stokes_i
    return frac_pol


def create_figure(grid_size, fsize=(20, 10), sharex=True, sharey=False):
    fig, sp = plt.subplots(*grid_size, sharex=sharex, sharey=sharey,
        gridspec_kw={"wspace": 0, "hspace": 0}, figsize=fsize, dpi=200)
    # plt.figure(figsize=fsize)
    return fig, sp


def read_pickle(fname, c_stoke):
    powers = {}
    with np.load(fname, allow_pickle=True) as data:
        powers[c_stoke] = data["flux"].astype(float)
        waves = data["waves"].astype(float)
        freqs = data["freqs"] / 1e9
    return powers


def are_all_nan(inp):
    return np.all(np.isnan(inp))


def read_regions(reg_num):
    datas = {}
    for stokes in "IQU":
        stoke_file = f"{stokes}-regions-mpc-70-postest/reg_{reg_num}_{stokes}.npz"
        with np.load(stoke_file, allow_pickle=True) as sfile:
            datas[stokes] = sfile["flux_jybm"].astype(float)
            if "waves" not in datas:
                datas["waves"] = sfile["waves"]
            if "freqs" not in datas:
                datas["freqs"] = sfile ["freqs"]
    return datas


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
        "poln_power": "mx", "frac_poln": "g+"}
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

            # logging.info(f"Reg {reg_name[1]}, Stokes {stokes}")
            # if c_stoke != "I": 
            #     print("Not stokes I")
            #     continue
            
            with np.load(stokes, allow_pickle=True) as data:
                polns[c_stoke] = data["flux_jybm"].astype(float)
                waves = data["waves"].astype(float)
                freqs = data["freqs"] / 1e9

            sp[row, col].plot(
                waves, polns[c_stoke], colours[c_stoke], 
                markersize=marker_size, label=c_stoke, alpha=0.4)

        sp[row, col].set_title(f"Reg {reg_name[1]}", y=1.0, pad=-14, size=9)
        sp[row, col].set_xscale(xscale)
        sp[row, col].set_yscale(xscale)

        # for power plots
        # polns["poln_power"] = linear_polzn(polns["Q"], polns["U"])
        # sp[row, col].plot(waves, polns["poln_power"], colours[c_stoke],
        #     markersize=marker_size, label="frac_pol", alpha=0.5)
        
        # for fractional polarization
        # polns["frac_poln"] = fractional_polzn(polns["I"], polns["Q"], polns["U"])
        # sp[row, col].plot(waves, polns["frac_poln"], colours[c_stoke], 
        #     markersize=marker_size, label="frac_pol", alpha=0.5)
        
        # ax2 = sp[row,col].twinx()
        # ax2.set_xlim(left=np.min(freqs), right=np.max(freqs))
        # ax2.set_xlabel("Frequency GHz")

        if np.prod((i+1)%grid_size_sq==0 or (n_qf<grid_size_sq and i==n_qf-1)):
            # Remove empties
            empties = [i for i, _ in enumerate(sp.flatten()) if not _.lines]
            for _ in empties:
                fig.delaxes(sp.flatten()[_])
            
            logging.info(f"Starting the saving process: Group {int(i/grid_size_sq)}")
            fig.tight_layout()
            fig.legend(list(polns.keys()), bbox_to_anchor=(1, 1.01), markerscale=3, ncol=3)
            # fig.suptitle("Q and U vs Lambda**2")
            fig.savefig(f"{outfile}-{int(i/grid_size_sq)}", bbox_inches='tight')
            plt.close("all")
            logging.info(f"Plotting done for {outfile}-{int(i/grid_size_sq)}")


def make_out_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    return os.path.relpath(dir_name)



if __name__ == "__main__":
    # [70, 50, 10, 7]
    # testing = "-test-"
    testing = "-postest"
    for i in [70]:
        # plot_dir = f"./plots-U-regions-mpc-test-{i}/"
        # file_core = f"regions-mpc{testing}-{i}"
        file_core = f"regions-mpc-{i}{testing}"
        plot_dir = make_out_dir(f"plots-QU-{file_core}")

        plot_spectra(file_core, f"{plot_dir}/QU-regions-mpc{testing}{i}")
        # plot_spectra(file_core, f"{plot_dir}/QU-regions-mpc{testing}{i}-logx", xscale="log")
    
