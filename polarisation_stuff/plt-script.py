import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import subprocess
from concurrent import futures
from functools import partial

from ipdb import set_trace


def read_npz(ins, compress=False):
    with np.load(ins, allow_pickle=True) as vn:
        data = dict(vn)
        
        if compress:
            mask = data.pop("mask")
            data = {k: np.ma.masked_array(data=v, mask=mask).compressed()
                    for k, v in data.items()}

    return data



############################################
### mark all the selections with some select color
############################################

def mark_them(sname, regname):
    """
    Mark the selections in current file, will coexist with the
    unselected
    sname: str
        File containing numbers of the selected regions
    regname: str
        Name of the region file being sifted through
    """
    sels = np.loadtxt(sname).astype(int)
    with open(regname, "r") as fil:
        beacons = fil.readlines()

    print(f"Found {sels.size} regions")


    for count, line in enumerate(beacons[3:], 1):
        if count in sels:
            line = line.replace("\n", " color=#0025E9 width=2\n")
            beacons[2+count] = line

    with open(os.path.basename(regname).replace(".reg", "-v2.reg"), "w") as fil:
        fil.writelines(beacons)

    print("Done writing the file")

    return


def select_them(sname, regname):
    """
    Put selections in new file
    sname: str
        File containing numbers of the selected regions
    regname: str
        Name of the region file being sifted through
    """
    sels = np.loadtxt(sname).astype(int)
    with open(regname, "r") as fil:
        beacons = fil.readlines()

    print(f"Found {sels.size} regions")

    news = []
    for count, line in enumerate(beacons[3:], 1):
        if count in sels:
            line = line.replace("\n", " color=#0025E9 width=2\n")
            news.append(line)

    news = beacons[:3]+news

    with open(os.path.basename(regname).replace(".reg", "-v2.reg"), "w") as fil:
        fil.writelines(news)

    print("Done writing the file")

    return

# mark_them("sel.txt", "beacons.reg")
# mark_them("sels-v2.txt", "weirdo-s3/regions/regions-default-valid.reg")

############################################

def staff(weirdo, weirdo_rm, odir):
    print(f"region: {weirdo}")  
    
    dat = read_npz(weirdo)
    dat_rm = read_npz(weirdo_rm)
    plot(dat, dat_rm, odir)
    return True

def plot(dat, dat_rm, odir):

    if "lpol" not in dat:
        dat["lpol"] = dat["Q"] + 1j*dat["U"]

    plt.close("all")
    fig, ax = plt.subplots(ncols=3, nrows=2, sharex=False, figsize=(16,9))
    ax[0,0].errorbar(dat["lambda_sq"], dat["Q"], yerr=dat["Q_err"],
        label="Q", fmt="o", ecolor="red")
    ax[0,0].errorbar(dat["lambda_sq"], dat["U"], yerr=dat["U_err"],
        label="U", fmt="o", ecolor="red")
    ax[0,0].legend()

    ax[0,1].errorbar(dat["lambda_sq"], dat["I"], yerr=dat["I_err"],
        label="Stokes I", fmt="o", ecolor="red")
    ax[0,1].errorbar(dat["lambda_sq"], np.abs(dat["lpol"]),
        yerr=dat["lpol_err"], label="| P |", fmt="o", ecolor="red")
    ax[0,1].set_yscale("log")
    ax[0,1].legend()

    ax[0,2].plot(dat_rm["depths"], np.abs(dat_rm["fclean"]),
        label="| FDF |")
    ax[0,2].axhline(np.abs(dat_rm["fclean"]).max()/2, label="fwhm", linestyle=":", color="k")
    ax[0,2].plot(dat_rm["depths"], np.real(dat_rm["fclean"]),
        label="Real", linestyle="-.", lw=0.7)
    ax[0,2].plot(dat_rm["depths"], np.imag(dat_rm["fclean"]),
        label="Imag", linestyle="--")
    ax[0,2].legend()

    ax[1,0].plot(dat["lambda_sq"], dat["snr"], label="SNR")
    ax[1,0].plot(dat["lambda_sq"], dat["psnr"], label="Polarised SNR")
    ax[1,0].set_xlabel(r"$\lambda ^2$")
    ax[1,0].set_yscale("log")
    ax[1,0].legend()

    ax[1,1].errorbar(dat["lambda_sq"], dat["fpol"], yerr=dat["fpol_err"],
        label=r"Fpol $\frac{|P|}{I}$", fmt="o", ecolor="red")
    ax[1,1].set_xlabel(r"$\lambda ^2$")
    ax[1,1].legend()


    rmtf = dat_rm["rmtf"]
    ax[1,2].plot(dat_rm["depths"], np.abs(rmtf),
        label="rmtf")
    ax[1,2].axhline(np.abs(rmtf).max()/2, label="fwhm", linestyle=":", color="k")
    ax[1,2].plot(dat_rm["depths"], np.real(rmtf),
        label="Real", linestyle="-.", lw=0.7)
    ax[1,2].plot(dat_rm["depths"], np.imag(rmtf),
        label="Imag", linestyle="--")
    ax[1,2].legend()
    

    fig.tight_layout()

    oname = os.path.join(odir, f"{dat_rm['reg_num']}.png")
    plt.savefig(oname, dpi=200)
    print("----")
    return True



def main():
    weirdos = subprocess.check_output("ls -v weirdo-data/*.npz", shell=True)
    weirdos = weirdos.decode().split("\n")[:-1]

    weirdos_rm = subprocess.check_output("ls -v weirdo-rm-data/*.npz", shell=True)
    weirdos_rm = weirdos_rm.decode().split("\n")[:-1]


    odir = "diagnostic"
    if not os.path.isdir(odir):
        os.mkdir(odir)


    with futures.ProcessPoolExecutor() as executor:
        res = list(executor.map(partial(staff, odir=odir),
            weirdos, weirdos_rm))

    # for weirdo, weirdo_rm in zip(weirdos, weirdos_rm):
    #     staff(weirdo, weirdo_rm, odir)


if __name__ == "__main__":

    """
    Generate files required for this script using
    
    python qu_pol/scrappy/scrap/scrappy.py -rs 3 -idir ../../relevant-images --threshold 50 -odir weirdo -ref-image ../../i-mfs.fits -nri ../../i-mfs.fits -rf XUSTOM-REGION.reg -nrf noise-region.reg 

    python qu_pol/scrappy/rmsynthesis/rm_synthesis.py -id weirdo-s3/los-data -od weirdo-s3/los-rm-data -md 400 --depth-step 1 -np

    ln -s weirdo-s3/los-data weirdo-data; 
    ln -s weirdo-s3/los-rm-data weirdo-rm-data

    python plt-script.py 

    """
    main()