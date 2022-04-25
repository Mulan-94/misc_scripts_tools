#!/bin/python3
"""
References
==========
https://www.extragalactic.info/~mjh/radio-flux.html
https://github.com/mhardcastle/radioflux/blob/master/radioflux/radioflux.py#L95-L109
https://science.nrao.edu/facilities/vla/proposing/TBconv
https://www.eaobservatory.org/jcmt/faq/how-can-i-convert-from-mjybeam-to-mjy/
"""

import argparse
import logging
import os
import regions
import matplotlib.pyplot as plt
import numpy as np
import warnings

# from casatasks import imstat
from astropy.io import fits
from glob import glob
from concurrent import futures
from functools import partial
from time import perf_counter
from itertools import product, chain
from ipdb import set_trace

plt.style.use("bmh")

light_speed = 3e8
marker_size = 10

# ignore overflow errors, assume these to be mostly flagged data
warnings.simplefilter("ignore")

l_handler = logging.FileHandler("xcrapping.log", mode="w")
l_handler.setLevel(logging.INFO)

s_handler = logging.StreamHandler()
s_handler.setLevel(logging.INFO)


logging.basicConfig(level=logging.DEBUG,
    datefmt='%H:%M:%S %d.%m.%Y',
    format="%(asctime)s - %(levelname)s - %(message)s", 
    handlers=[l_handler, s_handler])

def timer(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        logging.info(f"'{func.__name__}' run in: {perf_counter()-start:.2f} sec")
        return result
    return wrapper


class IOUtils:
    @staticmethod
    def make_out_dir(dir_name):
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        return os.path.relpath(dir_name)


    @staticmethod
    def read_sorted_filnames(fname):
        with open(fname, "r") as post:
            items = post.readlines()
            items = [_.replace("\n", "") for _ in items]
        return items

    @staticmethod
    def generate_regions(reg_fname, factor=50, max_w=572, max_h=572):
        """
        Create a DS9 region file containing a bunch of regions
        factor: int
            In my case, the size of the region ie length / widh t
            reg_fname:
                File name for the resulting region files
            max_*:
                Maximum pixel image height or width. 
                So that regions don't go beyound image dims
        """
        # left to right
        width_range = 74, 569

        # bottom to top
        height_range = 190, 450
        header = [
            "# Region file format: DS9 CARTA 2.0.0",
            ('global color=green dashlist=8 3 width=1 ' +
            'font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 ' +
            'edit=1 move=1 delete=1 include=1 source=1'),
            "physical"
        ]

        pts = []
        count = 0
        for height in range(*height_range, factor):
            for width in range(*width_range, factor):
                # pts.append("circle({}, {}, {}) # color=#2EE6D6 width=2".format(width, height+factor, factor/2))
                width = max_h if width > max_w else width
                height = max_h if height > max_h else height
                pts.append(
                    "box({}, {}, {}, {}) # color=#2EE6D6 width=2 text={{reg_{}}}".format(
                        width, height, factor, factor, count))
                count += 1

        if ".reg" not in reg_fname:
            reg_fname += f"-{factor}.reg"

        if not os.path.isfile(reg_fname):
            with open(reg_fname, "w") as fil:
                pts = [p+"\n" for p in header+pts]
                fil.writelines(pts)

            logging.info(f"Regions file written to: {reg_fname}")
        else:
            logging.info(f"File already available at: {reg_fname}")
        return reg_fname


class Plotting:
    def __init__(self):
        pass

    @staticmethod
    def format_lsq(inp, func):
        """
        Converting and formating output
        Funxtions expdcted lambda_sq, and freq_sqrt
        """
        inp = getattr(MathUtils, func)(inp)
        return [float(f"{_:.2f}") for _ in inp]

    @staticmethod
    def active_legends(fig):
        legs = { _.get_legend_handles_labels()[-1][0] for _ in fig.axes
                if len(_.get_legend_handles_labels()[-1])>0 }
        return list(legs)

    @staticmethod
    def create_figure(grid_size, fsize=(20, 10), sharex=True, sharey=False):
        fig, sp = plt.subplots(*grid_size, sharex=sharex, sharey=sharey,
            # gridspec_kw={"wspace": 0, "hspace": 1}, 
            figsize=fsize, dpi=200)
        # plt.figure(figsize=fsize)
        return fig, sp

    @classmethod
    def plot_spectra(cls, file_core, outfile, xscale="linear", ymin=None,
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
        data_files = sorted(glob(f"./IQU-{file_core}/*.npz"), key=fight)
        n_qf = len(data_files)
        logging.info(f"Found {n_qf} QUI files")

        # rationale is a 4:3 aspect ratio, max is this value x 3 = 12:9
        # toget respsective sizes, add length+width and mult by ratio
        rows = 9 if n_qf > 108 else int(np.ceil(3/7*(np.sqrt(n_qf)*2)))
        cols = 12 if n_qf > 108 else int(np.ceil(4/7*(np.sqrt(n_qf)*2)))
        grid_size_sq = rows*cols

        logging.info("Starting plots")
        plt.close("all")
        plots = 0
        for i, data_file in enumerate(data_files):
            if i % grid_size_sq == 0:
                fig, sp = cls.create_figure((rows, cols), fsize=(50, 30), sharey=False)
                rc = product(range(rows), range(cols))
            
            reg_name = os.path.splitext(os.path.basename(data_file))[0].split("_")
            with np.load(data_file, allow_pickle=True) as data:
                # these frequencies are in Hz
                datas = {k: v for k, v in data.items()}
            
            datas["waves"] = MathUtils.lambda_sq(datas["freqs"])
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
            ax2.set_xticklabels(cls.format_lsq(new_ticklocs, "freq_sqrt"))
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
                legs = cls.active_legends(fig)
                fig.legend(legs, bbox_to_anchor=(1, 1.01), markerscale=3, ncol=len(legs))

                # fig.suptitle("Q and U vs $\lambda^2$")
                oname = f"{outfile}-{int(i/grid_size_sq)}-{xscale}"
                
                plt.setp(sp[:,0], ylabel="Frac Pol")
                plt.setp(sp[-1,:], xlabel="Wavelength m$^2$")
        
                fig.savefig(oname, bbox_inches='tight')
                plt.close("all")
                logging.info(f"Plotting done for {oname}")
        logging.info(f"We have: {plots}/{n_qf} plots")

    @classmethod
    def plot_spectra_singles(cls, file_core, outfile, xscale="linear", ymin=None,
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

        data_files = sorted(glob(f"./IQU-{file_core}/*.npz"), key=fight)
        n_qf = len(data_files)
        logging.info(f"Found {n_qf} QUI files")

        logging.info("Starting plots")
        plt.close("all")
        plots = 0
        # data_points = []
        for i, data_file in enumerate(data_files):
            
            fig, sp = cls.create_figure((1, 1), fsize=(16, 9), sharey=False)
            
            reg_name = os.path.splitext(os.path.basename(data_file))[0].split("_")

            with np.load(data_file, allow_pickle=True) as data:
                # these frequencies are already in GHZ
                datas = {k: v for k, v in data.items()}
            
            datas["waves"] = MathUtils.lambda_sq(datas["freqs"])

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
            ax2.set_xticklabels(cls.format_lsq(new_ticklocs, "freq_sqrt"))
            ax2.tick_params(axis="x",direction="in", pad=-15)

            
            plt.setp(ax2, xlabel="Freq GHz")
               
                
            fig.tight_layout(h_pad=3)
            # legs = cls.active_legends(fig)
            # fig.legend(legs, bbox_to_anchor=(1, 1.01), markerscale=3, ncol=len(legs))
            sp.legend(bbox_to_anchor=(1, 1.05), markerscale=3, ncol=4)

            # fig.suptitle("Q and U vs $\lambda^2$")
            
            oname = f"{outfile}-{'_'.join(reg_name)}-{xscale}"
            
            plt.setp(sp, ylabel="Frac Pol")
            plt.setp(sp, xlabel="Wavelength m$^2$")
    
            fig.savefig(oname, bbox_inches='tight')
            plt.close("all")
            logging.info(f"Plotting done for {oname}")
        logging.info(f"We have: {plots}/{n_qf} plots")
        # logging.info(f"Regions with >4 data points in fracpol {data_points}")



class MathUtils:
    def __init__(self):
        pass

    @staticmethod
    def rms(in_data):
        """Root mean square"""
        return np.sqrt(np.square(in_data).mean())

    @staticmethod
    def ssq(in_data):
        """Sum of squares"""
        return np.nansum(np.square(in_data))

    @staticmethod
    def sqrt_ssq(*args):
        """Square root of sum of squares"""
        squares = [np.square(x) for x in args]
        squares = np.sqrt(np.sum(squares, axis=0))
        return squares

    @staticmethod
    def are_all_nan(inp):
        return np.all(np.isnan(inp))

    @staticmethod
    def are_all_zeroes(inp):
        return np.all(inp==0)

    @staticmethod
    def is_infinite(inp):
        return np.isinf(inp)

    @classmethod
    def linear_polzn(cls, stokes_q, stokes_u, noise=None, thresh=10):
        lin_pol = cls.sqrt_ssq(stokes_q, stokes_u)

        # if noise:
        #     noise_floor = noise * thresh
        #     #if there's a value less than noise floor, set it to nan
        #     if any(lin_pol < noise_floor):
        #         lin_pol[np.where(lin_pol <noise_floor)] = np.nan
        return lin_pol

    @classmethod
    def fractional_polzn(cls, stokes_i, stokes_q, stokes_u, noise=None, thresh=10):
        linear_pol = cls.linear_polzn(stokes_q, stokes_u, noise=noise, thresh=thresh)
        frac_pol = linear_pol / stokes_i
        return frac_pol

    @staticmethod
    def lambda_sq(freq_ghz):
        #speed of light in a vacuum
        global light_speed
        # frequency to wavelength
        wave = light_speed/freq_ghz
        return np.square(wave)

    @staticmethod
    def freq_sqrt(lamsq):
        #speed of light in a vacuum
        global light_speed

        lamsq = np.sqrt(lamsq)
        # frequency to wavelength
        freq_ghz = (light_speed/lamsq)
        return freq_ghz


class FitsManip:
    """Concerning Regions and FITS images"""

    @staticmethod
    def get_box_dims(reg):
        """
        *** For use in imsats ***
        --------------------------
        Create a valid region entry. 
        
        reg: :obj:`regions.Region`
            A region object
        """
        # blc: bottom left corner, trc: top right corner
        # blc_x, trc_x, blc_y, trc_y
        box = ",".join(
            [str(getattr(reg.bounding_box, x)) 
                for x in "ixmin iymin ixmax  iymax".split()])
        return box

    @staticmethod
    def get_imstat_box_dims(reg):
        """
        *** For use in imsats ***
        --------------------------
        Get box dimensions for CASA's imstat

        blc: bottom left corner, trc: top right corner
        This function is to deal with the off by -1 issue with casa's
        imstats TRC argument. The problem was that while using imstats and doing
        a box selection, setting the TRC value to eg (100, 100) would lead to the 
        selection of pixels upto (101, 101). So for example, I want pixels btwn
        blc(0,90) and trc(2, 92), I would expect to get back stats for 4=(2x2) pixels
        however, I get back stats for 9 pixels because 9=(3x3)

        I therefore reduce the trc values by 1 and return the values in the form:
            blc_x, blc_y, trc_x, trc_y

        reg: :obj:`regions.Region`
            A region object
        """
        box = []
        for x in "ixmin iymin ixmax  iymax".split():
            val = getattr(reg.bounding_box, x)
            if "max" in x:
                val -= 1
            box.append(str(val))
        return ",".join(box)


    @staticmethod
    def get_useful_data(fname):
        """
        Get some useful information from FITS image. Also infer the stokes type

        fname: :obj:`fname`
            FITS image name
        """
        data = {}
        try:
            int(os.path.basename(fname).split("-")[-2])
            data["stokes"] = os.path.basename(fname).split("-")[-3]
        except ValueError:
            data["stokes"] = os.path.basename(fname).split("-")[-2]

        with fits.open(fname) as hdul:
            # data["bmaj"] = hdul[0].header["BMAJ"]
            # data["bmin"] = hdul[0].header["BMIN"]
            data["freqs"] = hdul[0].header["CRVAL3"]
            data["fnames"] = fname
            data["data"] = hdul[0].data.squeeze()
        return data

    
    @classmethod
    def get_noise(cls, noise_reg, fname, data=None):
        if not isinstance(noise_reg, regions.Region):
            noise = noise_reg
        else:
            if data is None:
                im_data = cls.get_useful_data(fname)
                data = im_data["data"]
            # Get image noise from standard deviation of a sourceless region
            noise_cut = cls.get_data_cut(noise_reg, data)
            noise = np.nanstd(noise_cut)
        return noise


    @classmethod
    def extract_stats2(cls, fname, reg, noise_reg, sig_factor=10):
        """
        Extract statistics from a specified image region
        fname :obj:`str`
            FITS image name
        reg: :obj:`regions.Region`
            A region object
        noise_reg: :obj:`regions.region`
            A specified know noise region object
        sig_factor: :obj:`int | float`
            Sigma factor. Threshold above which signal should be above noise
        """
        # get image data
        im_data = cls.get_useful_data(fname)
        data = im_data.pop("data")

        noise = cls.get_noise(noise_reg, fname, data=data)

        
        # Get mean flux over the pixels
        intense_cut = cls.get_data_cut(reg, data)


        if (MathUtils.are_all_nan(intense_cut) or MathUtils.are_all_zeroes(intense_cut) or
            MathUtils.is_infinite(noise)):
            # skip all the nonsence if all the data is Nan
            logging.debug(f"Skipping region:{reg.meta['label']} {fname} because NaN/Zeroes/inf ")
            # im_data["flux_jybm"] = im_data["flux_jy"] = None
            im_data["flux_jybm"] = im_data["noise"] = None
            return im_data

        # flux per beam sum
        # flux_jybm = np.nansum(cls.get_data_cut(reg, data))
        
        # flux_jybm = np.nanmean(intense_cut)
        # cpixx, cpixy = np.array(intense_cut.shape)//2
        cpixx, cpixy = np.ceil(np.array(intense_cut.shape)/2).astype(int)
        flux_jybm = intense_cut[cpixx, cpixy]

        if (flux_jybm > sig_factor * noise):
        # if flux_jybm > noise:
            im_data["flux_jybm"] = flux_jybm
        else:
            im_data["flux_jybm"] = None
    
        im_data["noise"] = noise
        return im_data

    @staticmethod
    def get_flux(header, flux_sum):
        """
        Calculate region flux in Jansky ie from Jy/beam -> Jy
        header:
            FITS image header
        flux_sum:
            Sum of all pixels in a given region
        """
        bmaj = np.abs(header["BMAJ"])
        bmin = np.abs(header["BMIN"])
        cdelt1 = header["CDELT1"]
        cdelt2 = header["CDELT2"]
        # from definitions of FWHM
        gfactor = 2.0 * np.sqrt(2.0 * np.log(2.0))
        beam_area = np.abs((2 * np.pi * (bmaj/cdelt1) * (bmin/cdelt2)) / gfactor**2)
        if flux_sum>0:
            flux = flux_sum / beam_area

        else:
            flux = None
        return flux

    @staticmethod
    def get_data_cut(reg, data):
        """
        Returns a data array containing only data from specified region

        reg: :obj:`regions.Region`
            Region of interest
        data: :obj:`np.ndarray`
            Data array of a given fits file
        """
        reg_mask = reg.to_mask()
        data_cut = reg_mask.cutout(data)
        return data_cut

    @classmethod
    def get_image_stats2(cls, file_core, images, regs, noise_reg, sig_factor):
        fluxes, waves = [], []
        logging.info("starting get_image_stats")
        for reg in regs:
            logging.info(f"Region: {reg.meta['label']}")
            with futures.ProcessPoolExecutor(max_workers=16) as executor:
                results = executor.map(
                    partial(cls.extract_stats2, reg=reg, noise_reg=noise_reg, sig_factor=sig_factor), images
                    )
            
            # # some testbed
            # results = []
            # for im in images:
            #     results.append(cls.extract_stats2(im, reg=reg, noise_reg=noise_reg, sig_factor=sig_factor))

            outs = {_: {"flux_jybm": [], "freqs": [],"fnames": [], "noise": []}
                        for _ in "IQU"}

            for res in results:
                outs[res["stokes"]]["flux_jybm"].append(res["flux_jybm"])
                outs[res["stokes"]]["freqs"].append(res["freqs"])
                outs[res["stokes"]]["fnames"].append(res["fnames"])
                outs[res["stokes"]]["noise"].append(res["noise"])
            
            checks = [outs.get(k)["flux_jybm"] for k in "IQU"][0]
        
            if not all(_ is None for _ in checks):
                # contains I, Q, U, noise, freqs, fpol, lpol
                fout = {k: np.asarray(v["flux_jybm"], dtype=np.float) for k,v in outs.items()}
                fout.update({v: outs["I"][v] for v in ["noise", "freqs"]})
                
                fout["lpol"] = MathUtils.linear_polzn(fout["Q"], fout["U"],
                        noise=noise_reg, thresh=sig_factor)

                fout["fpol"] = MathUtils.fractional_polzn(fout["I"], fout["Q"],
                    fout["U"], noise=noise_reg, thresh=sig_factor)

                out_dir = IOUtils.make_out_dir(f"IQU-{file_core}")
                outfile = os.path.join(out_dir, f"{reg.meta['label']}")
                np.savez(outfile, **fout)

        logging.info(f"Done saving data files")
        logging.info( "--------------------")
        return



def parser():
    print(
        """
        \r===================================================================+
        \rExamples                                                           |                 
        \r========                                                           |                                
        \r\nplotting only                                                    |                                           
        \r    python scrap.py -p 50 20 -t mzima-t10 -plp -piqu --plot-grid   |
        \r\nstats and plotting                                               |                                             
        \r    python scrap.py -f clean-small.txt -rs 50 20 -ap \             |
        \r          -t mzima-t10-v2 --threshold 10 \                         |
        \r          --noise -0.0004 -ap -plp -piqu                           | 
        \r\nwith region files                                                |
        \r      python scrap.py -rf regions/beacons-20-chosen.reg \          |
        \r        -f clean-small.txt -rs 20 -t chosen --threshold 10         |                                           
        \r===================================================================+
        """
    )
    parsing = argparse.ArgumentParser()
    parsing.add_argument("-f", "--infile", dest="in_list", type=str,
        default="post.txt", metavar="",
        help="File containing an organised list of the input image names."+
        "Can easily be done with 'ls *-image.fits > whatever.txt'")
    parsing.add_argument("-rf", "--region-file", dest="reg_file", type=str,
        default=None, metavar="", 
        help="An input region file. Otherwise, one will be generated.")
    parsing.add_argument("-rs", "--region-size", dest="reg_size", nargs="+",
        type=int, default=[], metavar="", 
        help="Create regions of this pixel size and perform analyses on them")
    parsing.add_argument("-t", "--testing", dest="testing", metavar="", type=str,
        help="Testing prefix. Will be Prepended with '-'", default=None)

    parsing.add_argument("--threshold", dest="thresh", metavar="", type=int,
        default=10,
        help="Noise factor threshold above which to extract. This will be threshold * noise_sigma")
    parsing.add_argument("--noise", dest="noise", metavar="", type=float,
        default=None,
        help="Noise value above which to extract. Default is automatic determine.")
    
    #plotting arguments
    parsing.add_argument("--plot-grid", dest="plot_grid", action="store_true",
        help="Enable to make gridded plots")
    parsing.add_argument("-ap", "--auto-plot", dest="auto_plot", action="store_true",
        help="Plot all the specified region pixels")
    parsing.add_argument("-piqu", "--plot-iqu", dest="plot_qu", action="store_true",
        help="Plot Q and U values")
    parsing.add_argument("-pfp", "--plot-frac-pol", dest="plot_frac_pol", action="store_true",
        help="Plot Fractional polarization")
    parsing.add_argument("-plp", "--plot-linear-pol", dest="plot_linear_pol", action="store_true",
        help="Plot linear polarization power")
    parsing.add_argument("--ymax", dest="ymax", type=float, 
        help="Y axis max limit")
    parsing.add_argument("--ymin", dest="ymin", type=float,
        help="Y axis min limit")
    parsing.add_argument("--xmax", dest="xmax", type=float, 
        help="Y axis max limit")
    parsing.add_argument("--xmin", dest="xmin", type=float,
        help="Y axis min limit")
    parsing.add_argument("-p", "--plot", dest="plot", nargs="*", type=int, 
        metavar="",
        help="Make plots for these region sizes manually. These will be linearly scaled")
    parsing.add_argument("-ps", "--plot-scales", dest="plot_scales", metavar="",
        default=["linear"], nargs="*", choices=["linear", "log"],
        help="Scales for the plots. Can be a space separated list of different scales.")

    return parsing


if __name__ == "__main__":
    opts = parser().parse_args()

    if opts.testing is None:
        testing = ""
    else:
        testing = "-" + opts.testing

    plotter = Plotting.plot_spectra if opts.plot_grid else Plotting.plot_spectra_singles

    if opts.reg_size:
        sortkey = lambda x: int(os.path.basename(x).split("-")[0])

        """
        Making sorted files using the following procedure
        # Create a list of all the Q, U folder contents only
        1. ls -rt channelised/*-*[0-9]/*[0-9][0-9][0-9][0-9]-*image* > post.txt

        # OPen the file with sublime and
        1b. Delete all MFS files listed
        2. Find all (using search bar ctrl +H) with Q-image.fits

        3. select all and modify to the fname of I files
        4. Change the containing dir for the i files because they end in '-I'. Done using regex
            find: 
                channelised/(\d*)-(\d*)/37b-QU-for-RM-(\d*)-(\d*)-I-(\d{4})-image.fits
            replace: 
                channelised/$1-$2-I/37b-QU-for-RM-$3-$4-I-$5-image.fits
        THEY WILL  BE IN POST.TXT [Now given as a commandline argument]
        """

        for factor in opts.reg_size:
            start = perf_counter()
            file_core = f"regions-mpc-{factor}{testing}"
            
            if opts.reg_file:
                reg_file = opts.reg_file
                logging.info(f"Using {reg_file} as region file")
            else:
                reg_file = IOUtils.generate_regions(f"regions/beacons", factor=factor)
            
            regs = regions.Regions.read(reg_file, format="ds9")
            
            images = IOUtils.read_sorted_filnames(opts.in_list)
            logging.info(f"Working on Stokes IQU")
            logging.info(f"With {len(regs)} regions")
            logging.info(f"And {len(images)} images ({len(images)//3} X IQU)")

            if opts.noise:
                noise_reg = opts.noise
            else:
                noise_reg, = regions.Regions.read("regions/noise_area.reg", format="ds9")
                logging.info(f"Getting noise from {images[0]}")
                noise_reg = FitsManip.get_noise(noise_reg, images[0], data=None)
            
            logging.info(f"Noise is       : {noise_reg}")
            logging.info(f"Sigma threshold: {opts.thresh}")
            logging.info(f"Noise threshold: {opts.thresh * noise_reg}")
            
            bn = FitsManip.get_image_stats2(file_core, images, regs, noise_reg, sig_factor=opts.thresh)

            if opts.auto_plot:
                logging.info("Autoplotting is enabled")
                plot_dir = IOUtils.make_out_dir(f"plots-IQU-{file_core}")
                pout = os.path.join(plot_dir,  f"IQU-{file_core}")
                plotter(file_core, 
                    f"{plot_dir}/IQU-regions-mpc{testing}-{factor}",
                    ymin=opts.ymin, ymax=opts.ymax, xmin=opts.xmin, xmax=opts.xmax,
                    plot_qu=opts.plot_qu, plot_frac_pol=opts.plot_frac_pol,
                    plot_linear_pol=opts.plot_linear_pol)

            logging.info(f"Finished factor {factor} in {perf_counter() - start} seconds")
            logging.info("======================================")

    if opts.plot:
        " python scrap.py -p 50 -t toto"

        logging.info(f"Plotting is enabled for regions {opts.plot}")
        for factor in opts.plot:
            for scale in opts.plot_scales:
                file_core = f"regions-mpc-{factor}{testing}"
                plot_dir = IOUtils.make_out_dir(f"plots-IQU-{file_core}")
                pout = os.path.join(plot_dir,  f"IQU-{file_core}")
                plotter(file_core, 
                    f"{plot_dir}/IQU-regions-mpc{testing}-{factor}",
                    xscale=scale, ymin=opts.ymin, ymax=opts.ymax,
                    xmin=opts.xmin, xmax=opts.xmax, plot_qu=opts.plot_qu,
                    plot_frac_pol=opts.plot_frac_pol,
                    plot_linear_pol=opts.plot_linear_pol)