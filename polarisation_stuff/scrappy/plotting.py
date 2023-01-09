import os
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors

from image_utils import (get_wcs, read_regions_as_pixels,
    image_noise, read_fits_image)
from rm_math import lambda_sq, freq_sqrt
from scraplog import snitch

plt.style.use("bmh")


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


def format_lsq(inp, func):
    """
    Converting and formating output
    Funxtions expdcted lambda_sq, and freq_sqrt
    """
    inp = func(inp)
    return [float(f"{_:.2f}") for _ in inp]


def plot_spectra(file_core, outfile, xscale="linear", ymin=None,
    ymax=None, xmin=None, xmax=None, plot_qu=False, plot_frac_pol=True, plot_linear_pol=False):
    """
    file_core: str
        core of the folders where the data are contained
    outfile: str
        prefix name of the output file
    """
    # r: red, b: blue, k: black
    colours = {
        'Q': {'color': 'r', 'marker': '2', "label": "Q"},
        'U': {'color': 'b', 'marker': '1', "label": "U"},
        'I': {'color': 'k', 'marker': 'o', "label": "I"},
        'lpol': {'color': 'g', 'marker': '+', "label": "Linear Poln"},
        'fpol': {'color': 'm', 'marker': 'x', "label": "Fractional Poln"}
        }

    fight = lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1])
    data_files = sorted(glob(f"./iqu-{file_core}/*.npz"), key=fight)
    n_qf = len(data_files)
    snitch.info(f"Found {n_qf} QUI files")

    # rationale is a 4:3 aspect ratio, max is this value x 3 = 12:9
    # toget respsective sizes, add length+width and mult by ratio
    rows = 9 if n_qf > 108 else int(np.ceil(3/7*(np.sqrt(n_qf)*2)))
    cols = 12 if n_qf > 108 else int(np.ceil(4/7*(np.sqrt(n_qf)*2)))
    grid_size_sq = rows*cols

    snitch.info("Starting plots")
    plt.close("all")
    plots = 0
    for i, data_file in enumerate(data_files):
        if i % grid_size_sq == 0:
            fig, sp = create_figure((rows, cols), fsize=(50, 30), sharey=False)
            rc = product(range(rows), range(cols))
        
        reg_name = os.path.splitext(os.path.basename(data_file))[0].split("_")
        with np.load(data_file, allow_pickle=True) as data:
            # these frequencies are in Hz
            datas = {k: v for k, v in data.items()}
        
        datas["waves"] = lambda_sq(datas["freqs"], datas["chan_width"])
        row, col = next(rc)

        if plot_frac_pol:
            specs = colours["fpol"]
            specs.update(dict(s=marker_size*4.1, alpha=0.7))
            sp[row, col].scatter(datas["waves"], datas["fpol"], **specs)


        if plot_linear_pol:
            specs = colours["lpol"]
            specs.update(dict(s=marker_size*4.1, alpha=0.7))
            sp[row, col].scatter(datas["waves"], datas["lpol"], **specs)


        if plot_qu:
            for stoke in "QU":
                specs = colours[stoke]
                specs.update(dict(s=marker_size*4.1, alpha=0.7))
                sp[row, col].scatter(datas["waves"], datas[stoke], **specs)


        if not np.all(np.isnan(datas["fpol"])):
            plots +=1
        else:
            continue


        if ymax or ymin:
            sp[row, col].set_ylim(ymax=ymax, ymin=ymin)
        
        # sp[row, col].set_xlim(xmax=xmax, xmin=xmin)
        
        sp[row, col].set_title(f"Reg {reg_name[1]}", y=1.0, pad=-20, size=9)
        sp[row, col].set_xscale(xscale)
        sp[row, col].set_yscale(xscale)
        sp[row, col].xaxis.set_tick_params(labelbottom=True)

        # adding in the extra x-axis for wavelength
        new_ticklocs = np.linspace((1.2*datas["waves"].min()), (0.9*datas["waves"].max()), 8)
        ax2 = sp[row, col].twiny()
        ax2.set_xlim(sp[row, col].get_xlim())
        ax2.set_xticks(new_ticklocs)
        ax2.set_xticklabels(format_lsq(new_ticklocs, "freq_sqrt"))
        ax2.tick_params(axis="x",direction="in", pad=-15)

        if row/rows == 0:
            plt.setp(ax2, xlabel="Freq GHz")

        if np.prod((i+1)%grid_size_sq==0 or (n_qf<grid_size_sq and i==n_qf-1)):
            # Remove empties
            empties = [i for i, _ in enumerate(sp.flatten()) if (not _.lines) and (not _.collections)]
            for _ in empties:
                fig.delaxes(sp.flatten()[_])
            
            snitch.info(f"Starting the saving process: Group {int(i/grid_size_sq)}")
            fig.tight_layout(h_pad=3)
            legs = active_legends(fig)
            fig.legend(legs, bbox_to_anchor=(1, 1.01), markerscale=3, ncol=len(legs))

            # fig.suptitle("Q and U vs $\lambda^2$")
            oname = f"{outfile}-{int(i/grid_size_sq)}-{xscale}"
            
            plt.setp(sp[:,0], ylabel="Frac Pol")
            plt.setp(sp[-1,:], xlabel="Wavelength m$^2$")
    
            fig.savefig(oname, bbox_inches='tight')
            plt.close("all")
            snitch.warning(f"Plotting done for {oname}")
    snitch.info(f"We have: {plots}/{n_qf} plots")


def plot_spectra_singles(file_core, outfile, xscale="linear", ymin=None,
    ymax=None, xmin=None, xmax=None, plot_qu=False, plot_frac_pol=True, plot_linear_pol=False):
    """
    file_core: str
        core of the folders where the data are contained
    outfile: str
        prefix name of the output file
    """
    # r: red, b: blue, k: black
    colours = {
        'Q': {'color': 'r', 'marker': '2', "label": "Q"},
        'U': {'color': 'b', 'marker': '1', "label": "U"},
        'I': {'color': 'k', 'marker': 'o', "label": "I"},
        'lpol': {'color': 'g', 'marker': '+', "label": "Linear Poln"},
        'fpol': {'color': 'm', 'marker': 'x', "label": "Fractional Poln"}
        }

    fight = lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1])

    data_files = sorted(glob(f"./iqu-{file_core}/*.npz"), key=fight)
    n_qf = len(data_files)
    snitch.info(f"Found {n_qf} QUI files")

    snitch.info("Starting plots")
    plt.close("all")
    plots = 0
    # data_points = []
    for i, data_file in enumerate(data_files):
        
        fig, sp = create_figure((1, 1), fsize=(16, 9), sharey=False)
        
        reg_name = os.path.splitext(os.path.basename(data_file))[0].split("_")

        with np.load(data_file, allow_pickle=True) as data:
            # these frequencies are already in GHZ
            datas = {k: v for k, v in data.items()}
        
        datas["waves"] = lambda_sq(datas["freqs"], datas["chan_width"])

        if plot_frac_pol:
            specs = colours["fpol"]
            specs.update(dict(s=marker_size*4.1, alpha=0.7))
            sp.scatter(datas["waves"], datas["fpol"], **specs)


        if plot_linear_pol:
            specs = colours["lpol"]
            specs.update(dict(s=marker_size*4.1, alpha=0.7))
            sp.scatter(datas["waves"], datas["lpol"], **specs)


        if plot_qu:
            for stoke in "QU":
                specs = colours[stoke]
                specs.update(dict(s=marker_size*4.1, alpha=0.7))
                sp.scatter(datas["waves"], datas[stoke], **specs)


        if not np.all(np.isnan(datas["fpol"])):
            plots +=1
        else:
            continue


        if ymax or ymin:
            sp.set_ylim(ymax=ymax, ymin=ymin)
        # sp.set_xlim(xmax=xmax, xmin=xmin)
        
        sp.set_title(f"Reg {reg_name[1]}", y=1.0, pad=-20, size=9)
        sp.set_xscale(xscale)
        sp.set_yscale(xscale)
        sp.xaxis.set_tick_params(labelbottom=True)

        # adding in the extra x-axis for wavelength
        new_ticklocs = np.linspace((1.2*datas["waves"].min()), (0.9*datas["waves"].max()), 8)
        ax2 = sp.twiny()
        ax2.set_xlim(sp.get_xlim())
        ax2.set_xticks(new_ticklocs)
        ax2.set_xticklabels(format_lsq(new_ticklocs, "freq_sqrt"))
        ax2.tick_params(axis="x",direction="in", pad=-15)

        
        plt.setp(ax2, xlabel="Freq GHz")
            
            
        fig.tight_layout(h_pad=3)
        # legs = active_legends(fig)
        # fig.legend(legs, bbox_to_anchor=(1, 1.01), markerscale=3, ncol=len(legs))
        sp.legend(bbox_to_anchor=(1, 1.05), markerscale=3, ncol=4)

        # fig.suptitle("Q and U vs $\lambda^2$")
        
        oname = f"{outfile}-{'_'.join(reg_name)}-{xscale}"
        
        plt.setp(sp, ylabel="Frac Pol")
        plt.setp(sp, xlabel="Wavelength m$^2$")

        fig.savefig(oname, bbox_inches='tight')
        plt.close("all")
        snitch.info(f"Plotting done for {oname}")
    snitch.info(f"We have: {plots}/{n_qf} plots")
    # snitch.info(f"Regions with >4 data points in fracpol {data_points}")


def overlay_regions_on_source_plot(reg_file, ref_image, noise_rfile, threshold=1):
    """
    Plotting noise data above noise in the I image. Also overlays the regions
    on the output plot

    reg_file:
        Name of region file contained
    ref_image:
        The I- image to be used for WCS coordinate positions, get the noise value,
        and is what will be plotted. From the location of this image, the rest of
        the images (q|u)-mfs.fits will try and be gotten and plotted as well
    noise_rfile: str | float
        Name of the region file containing the noise region, or the noise itself

    The plot will be stored in the regions directory under the directory created
    for the output of this script
    """
    wcs = get_wcs(ref_image)

    reg_file += "" if ".reg" in reg_file else ".reg"
    noise_rfile += "" if ".reg" in noise_rfile else ".reg"

    noise_reg, = read_regions_as_pixels(noise_rfile, wcs)

    chosen = read_regions_as_pixels(reg_file, wcs)
    
    plt.close("all")
    fig, ax = plt.subplots(
            figsize=(6, 10), ncols=1, nrows=3, sharex=True, sharey=True,
            gridspec_kw={"wspace":0, "hspace":0})

    ax = ax.flatten()
    snitch.info("Plotting selected regions over I (Q, U) images")
    i_data = None

    for _, im in enumerate("iqu"):

        now_image = ref_image.replace("i-mfs", f"{im}-mfs")

        # ignore if this file does not exist
        if not os.path.isfile(now_image):
            snitch.info(f"{now_image} not found")
            continue

        image = read_fits_image(now_image)
        image_data = image.data

        noise = image_noise(noise_reg, image_data)
        
        if im == "i":
            image_data = np.ma.masked_less(np.abs(image_data), noise*threshold)
            i_data = image_data
        else:
            image_data = np.ma.masked_array(data=image_data, mask=i_data.mask)

        lims = np.array([_.center.xy for _ in chosen], dtype=int)
        xmin, ymin = np.min(lims, axis=0)
        xmax, ymax = np.max(lims, axis=0)
        wiggle = 50
        
        ax[_].imshow(image_data, origin="lower", cmap="coolwarm",
        norm=colors.LogNorm(vmin=image_data.min(), vmax=image_data.max())
        # vmin=data.min(), vmax=data.max()
        ) 
        ax[_].set_title(f"{im.upper()}; Only > t {threshold} x n {noise:.6f} in log")
        ax[_].set_xlim(xmin-wiggle, xmax+wiggle)
        ax[_].set_ylim(ymin-wiggle, ymax+wiggle)

        for choice in chosen:
            x,y,r = np.array(np.round([*choice.center.xy, choice.radius]),dtype=int)
            ax[_].add_patch(plt.Circle((x,y), radius=r, color="indigo", fill=False, lw=0.5))


    fig.tight_layout()
    oname = os.path.splitext(reg_file)[0] + ".png"

    if os.path.isfile(oname):
        name, ext = os.path.splitext(oname)
        oname = name + "-a" + ext

    
    plt.savefig(oname, dpi=400)
    snitch.info(f"Overlay saved at: {oname}")

