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
    aics = dual["aic"] - single["aic"]
    bics = dual["bic"] - single["bic"]

    # comps = dict(bf=bayes_factor, bf_err=None, aic_diff=aics, bic_diff=bics)
    # comps = bayes_factor,  dual["evidence"],  single["evidence"]
    comps = bayes_factor, aics, bics
    return comps



def plot_infos(regs, bf, aics, bics):

    fig, ax = plt.subplots(figsize=(6,10), nrows=3)

    # ax[0].axhspan(0, 1, facecolor="black", alpha=0.2, label="Weak")
    # ax[0].axhspan(1, 3,  facecolor="red", alpha=0.2, label="Positive")
    # ax[0].axhspan(3, np.nanmax(bf),  facecolor="yellow", alpha=0.2,
    #             label="Strong")

    # ax[0].error_bar(regs, bf)
    ax[0].scatter(regs, bf)
    # ax[0].set_yscale("log", font="monospace", fontsize=15)
    ax[0].set_ylabel(r"ln(BF) = ln(Z$_{dual}$) - ln(Z$_{single}$)", font="monospace", fontsize=15)
    ax[0].set_xlabel("LoS", font="monospace", fontsize=15)
    # ax[0].legend()


    # ax[1].plot(regs, np.abs(aics), "bo", label=r"Evidence dual ")
    # ax[1].plot(regs, np.abs(bics), "ro", label=r"Evidence single$")
    ax[1].plot(regs, aics, "bo", label=r"AIC$_{dual}$ - AIC$_{single}$")
    ax[1].set_ylabel(r"AIC$_{dual}$ - AIC$_{single}$", font="monospace", fontsize=15)
    # ax[1].set_yscale("log", font="monospace", fontsize=15)
    ax[1].set_xlabel("Los", font="monospace", fontsize=15)
    # ax[1].legend()


    ax[2].plot(regs, bics, "ro", label=r"BIC$_{dual}$ - BIC$_{single}$")
    ax[2].set_xlabel("LoS", font="monospace", fontsize=15)
    # ax[2].legend()
    ax[2].set_ylabel(r"BIC$_{dual}$ - BIC$_{single}$", font="monospace", fontsize=15)
    fig.tight_layout() 
    fig.savefig("dual-vs-single-comp-model.pdf", dpi=200)

    # #############3 Bayesian factor only #################
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(regs, bf)
    ax.set_ylabel(r"ln(BF) = ln(Z$_{dual}$) - ln(Z$_{single}$)", font="monospace", fontsize=15)
    ax.set_xlabel("LoS", font="monospace", fontsize=15)
    fig.tight_layout() 
    fig.savefig("dual-vs-single-comp-model-bf.pdf", dpi=200)




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

    
    maxa = np.round(np.percentile(aics, 99.8), -2)
    
    
    aics = np.ma.masked_outside(aics, -maxa, maxa)
    # bics = np.ma.masked_outside(bics, -maxb, maxb)
    bics = np.ma.masked_array(data=bics, mask=aics.mask)
    bf = np.ma.masked_array(data=bf, mask=aics.mask)
    
    plot_infos(regs, bf, aics, bics)


if __name__ == "__main__":
    main()