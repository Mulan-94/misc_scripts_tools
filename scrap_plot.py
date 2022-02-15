#### For plotting
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from ipdb import set_trace
from itertools import product


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


if __name__ == "__main__":
    plot_spectra("regions-mpc-10", "QU-regions-mpc-10") 