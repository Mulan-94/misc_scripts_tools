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
    
