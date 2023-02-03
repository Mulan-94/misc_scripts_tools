import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import subprocess

from ipdb import set_trace

def read_npz(ins, compress=False):
    with np.load(ins, allow_pickle=True) as vn:
        data = dict(vn)
        
        if compress:
            mask = data.pop("mask")
            data = {k: np.ma.masked_array(data=v, mask=mask).compressed()
                    for k, v in data.items()}

    return data



######################
### mark all the selections with some select color

# sels = np.loadtxt("sel.txt")
# sels = sels.astype(int)
# with open("beacons.reg", "r") as fil:
#     beacons = fil.readlines()


# for count, line in enumerate(beacons[3:], 1):
#     if count in sels:
#         line = line.replace("\n", " color=#0025E9 width=2\n")
#         beacons[2+count] = line


# with open("beacons-2peak.reg", "w") as fil:
#     fil.writelines(beacons)

######################

def main():
    weirdos = subprocess.check_output("ls -v weirdo-data/*.npz", shell=True)
    weirdos = weirdos.decode().split("\n")[:-1]

    weirdos_rm = subprocess.check_output("ls -v weirdo-rm-data/*.npz", shell=True)
    weirdos_rm = weirdos_rm.decode().split("\n")[:-1]


    odir = "diagnostic"
    if not os.path.isdir(odir):
        os.mkdir(odir)

    for weirdo, weirdo_rm in zip(weirdos, weirdos_rm):
        dat = read_npz(weirdo)
        dat_rm = read_npz(weirdo_rm)

        print(f"region: {weirdo}")

        plt.close("all")
        fig, ax = plt.subplots(ncols=3, nrows=2, sharex=False, figsize=(16,9))
        ax[0,0].errorbar(dat["lambda_sq"], dat["Q"], yerr=dat["Q_err"], label="Q", fmt="o", ecolor="red")
        ax[0,0].errorbar(dat["lambda_sq"], dat["U"], yerr=dat["U_err"], label="U", fmt="o", ecolor="red")
        ax[0,0].legend()

        ax[0,1].errorbar(dat["lambda_sq"], dat["I"], yerr=dat["I_err"], label="Stokes I", fmt="o", ecolor="red")
        ax[0,1].errorbar(dat["lambda_sq"], np.abs(dat["lpol"]), yerr=dat["lpol_err"], label="| P |", fmt="o", ecolor="red")
        ax[0,1].set_yscale("log")
        ax[0,1].legend()

        ax[0,2].plot(dat_rm["depths"], np.abs(dat_rm["fclean"]), label="| FDF |")
        ax[0,2].plot(dat_rm["depths"], np.real(dat_rm["fclean"]), label="Real", linestyle="-.", lw=0.7)
        ax[0,2].plot(dat_rm["depths"], np.imag(dat_rm["fclean"]), label="Imag", linestyle="--")
        ax[0,2].legend()

        ax[1,0].plot(dat["lambda_sq"], dat["snr"], label="SNR")
        ax[1,0].plot(dat["lambda_sq"], dat["psnr"], label="Polarised SNR")
        ax[1,0].set_xlabel("$\lambda ^2$")
        ax[1,0].set_yscale("log")
        ax[1,0].legend()

        ax[1,1].errorbar(dat["lambda_sq"], dat["fpol"], yerr=dat["fpol_err"], label=r"Fpol $\frac{|P|}{I}$", fmt="o", ecolor="red")
        ax[1,1].set_xlabel("$\lambda ^2$")
        ax[1,1].legend()

        ax[1,2].plot(dat_rm["depths"], np.abs(dat_rm["rmtf"]), label="rmtf")
        ax[1,2].plot(dat_rm["depths"], np.real(dat_rm["rmtf"]), label="Real", linestyle="-.", lw=0.7)
        ax[1,2].plot(dat_rm["depths"], np.imag(dat_rm["rmtf"]), label="Imag", linestyle="--")
        ax[1,2].legend()
        

        fig.tight_layout()

        oname = os.path.join(odir, f"{os.path.basename(weirdo)}.png")
        plt.savefig(oname, dpi=200)


if __name__ == "__main__":
    main()