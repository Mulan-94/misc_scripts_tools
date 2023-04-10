import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
import subprocess
from ipdb import set_trace


def make_ord_flists(dir):
    sstr = os.path.join(dir, "*-fitparameters.txt")
    mlist = subprocess.check_output(
        f"ls -v {sstr}",
        shell=True)
    mlist = mlist.decode().split("\n")[:-1]
    return mlist

def read_data(fname):
    data = np.loadtxt(fname)
    if data.size ==0:
        print(f"File is {fname} empty")
        return None

    outs = dict(
        evidence=data[-8], evidence_err=data[-7],
        aic=data[-3], bic=data[-2])
    return outs


def compare_models(model1, model2):
    dual = read_data(model1)
    single = read_data(model2)

    if None in [single, dual]:
        return

    bayes_factor = dual["evidence"] - single["evidence"]


    # The smaller the AIC the better
    # so if negative, dual is better than single
    # aics = dual["aic"] - single["aic"]
    # bics = dual["bic"] - single["bic"]

    # comps = dict(bf=bayes_factor, bf_err=None, aic_diff=aics, bic_diff=bics)
    return bayes_factor,  dual["evidence"],  single["evidence"]



def plot_infos(regs, bf, aics, bics):

    fig, ax = plt.subplots(figsize=(10,10), ncols=2)

    # ax.axhline(y=0, linestyle=":", color="black", label="For Model")
    # ax.axhline(y=1, linestyle="--", color="red", label="Weak")
    # ax.axhline(y=3, linestyle="--", color="yellow", label="Positive")
    # ax.axhline(y=5, linestyle="--", color="green", label="Strong")
    ax[0].axhspan(0, 1, facecolor="black", alpha=0.2, label="Weak")
    ax[0].axhspan(1, 3,  facecolor="red", alpha=0.2, label="Positive")
    ax[0].axhspan(3, 5,  facecolor="yellow", alpha=0.2, label="Strong")

    # ax[0].error_bar(regs, bf)
    ax[0].scatter(regs, bf)
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"ln(BF) = [ln(Z_{dual} - ln(Z_single)]")
    ax[0].set_xlabel("Region")


    # ax[1].axhspan(0, 1, linestyle=":", facecolor="black", alpha=0.2, label="Weak")
    # ax[1].axhspan(1, 3, linestyle="--", facecolor="red", alpha=0.2, label="Positive")
    # ax[1].axhspan(3, 5, linestyle="--", facecolor="yellow", alpha=0.2, label="Strong")

    # ax[1].plot(regs, aics, "bo", label=r"AIC$_{dual}$ - AIC$_{single}$")
    # ax[1].plot(regs, bics, "ro", label=r"BIC$_{dual}$ - BIC$_{single}$")

    ax[1].plot(regs, np.abs(aics), "bo", label=r"Evidence dual ")
    ax[1].plot(regs, np.abs(bics), "ro", label=r"Evidence single$")
    ax[1].set_ylabel("Evidence")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Region")

    fig.legend()
    fig.tight_layout() 

    fig.savefig("comparision-model.png", dpi=200)



def main():
    duals = make_ord_flists("weirdo-s3/qu-fits-dual")
    singles = make_ord_flists("weirdo-s3/qu-fits-single")

    print(f"single - {len(singles)} and dual - {len(duals)}")

    if len(duals) != len(singles):
        print("Number of Files not matching")
        os._exit(-1)

    aics = np.zeros(len(duals))
    bf = np.zeros(len(duals))
    bics = np.zeros(len(duals))
    regs = np.zeros(len(duals))

    
    for dual, single in zip(duals, singles):
        # assuming here that the regions are coninouts from 1 -> N
        reg = int(os.path.basename(dual).split("-")[1].split("_")[-1]) - 1
        comparison = compare_models(dual, single)
    
        if comparison is not None:
            bf[reg], aics[reg], bics[reg] = comparison
        else:
             bf[reg], aics[reg], bics[reg] = [np.nan]*3
        regs[reg] = reg+1
    plot_infos(regs, bf, aics, bics)


if __name__ == "__main__":
    main()