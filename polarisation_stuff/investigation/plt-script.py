import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

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
    weirdos = glob("weirdo-data/*.npz")

    odir = "diagnostic"
    if not os.path.isdir(odir):
        os.mkdir(odir)

    for weirdo in weirdos:
        dat = read_npz(weirdo)
        print(f"region: {weirdo}")
        plt.close("all")
        fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, figsize=(10,10))
        ax[0,0].errorbar(dat["lambda_sq"], dat["Q"], yerr=dat["Q_err"], label="Q", fmt="o", ecolor="red")
        ax[0,0].errorbar(dat["lambda_sq"], dat["U"], yerr=dat["U_err"], label="U", fmt="o", ecolor="red")
        ax[0,0].legend()

        ax[0,1].errorbar(dat["lambda_sq"], dat["I"], yerr=dat["I_err"], label="Stokes I", fmt="o", ecolor="red")
        ax[0,1].errorbar(dat["lambda_sq"], np.abs(dat["lpol"]), yerr=dat["lpol_err"], label="| P |", fmt="o", ecolor="red")
        ax[0,1].set_yscale("log")
        ax[0,1].legend()

        ax[1,0].plot(dat["lambda_sq"], dat["snr"], label="SNR")
        ax[1,0].plot(dat["lambda_sq"], dat["psnr"], label="Polarised SNR")
        ax[1,0].set_xlabel("$\lambda ^2$")
        ax[0,1].set_yscale("log")
        ax[1,0].legend()

        ax[1,1].errorbar(dat["lambda_sq"], dat["fpol"], yerr=dat["fpol_err"], label=r"Fpol $\frac{|P|}{I}$", fmt="o", ecolor="red")
        ax[1,1].set_xlabel("$\lambda ^2$")
        ax[1,1].legend()    
        

        fig.tight_layout()

        oname = os.path.join(odir, f"{os.path.basename(weirdo)}.png")
        plt.savefig(oname)


if __name__ == "__main__":
    main()