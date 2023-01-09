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
import shutil
import warnings

# from casatasks import imstat
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord, FK5
from astropy.wcs import WCS
from glob import glob
from concurrent import futures
from functools import partial
from time import perf_counter
from itertools import product, chain
from regions import Regions, PixCoord, CirclePixelRegion, RectanglePixelRegion
from pprint import pformat
from matplotlib import colors
from ipdb import set_trace


plt.style.use("bmh")

light_speed = 3e8
marker_size = 10
FPOL_FILTER = []
MFT = 0.7

def configure_logger(out_dir):
    # ignore overflow errors, assume these to be mostly flagged data
    warnings.simplefilter("ignore")

    formatter = logging.Formatter(
        datefmt='%H:%M:%S %d.%m.%Y',
        fmt="%(asctime)s : %(levelname)s - %(message)s")
    
    _ = IOUtils.make_out_dir(out_dir)
    l_handler = logging.FileHandler(
        os.path.join(out_dir, "xcrapping.log"), mode="w")
    l_handler.setLevel(logging.WARNING)
    l_handler.setFormatter(formatter)

    s_handler = logging.StreamHandler()
    s_handler.setLevel(logging.INFO)
    s_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.addHandler(l_handler)
    logger.addHandler(s_handler)
    return logger


def timer(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        snitch.info(f"'{func.__name__}' run in: {perf_counter()-start:.2f} sec")
        return result
    return wrapper


class IOUtils:
    @staticmethod
    def make_out_dir(dir_name):
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        return os.path.relpath(dir_name)


    @staticmethod
    def read_sorted_filnames(fname):
        with open(fname, "r") as post:
            items = post.readlines()
            items = [_.replace("\n", "") for _ in items]
        return items

    @staticmethod
    def choose_valid_regs(fname, reg, noise_reg, threshold=20):
        """
        Extract statistics from a specified image region
        fname :obj:`str`
            FITS image name for some stokes I
        reg: :obj:`regions.Region`
            A region object
        noise_reg: :obj:`regions.region`
            A specified know noise region object
        threshold: :obj:`int | float`
            I tested 20, seems better than most. I recommend
        """
        valid, invalid = True, False

        # get image data
        im_data = FitsManip.get_useful_data(fname)
        data = im_data.pop("data")

        image_noise = FitsManip.get_noise(noise_reg, fname, data=data)
        # Get mean flux over the pixels
        intense_cut = FitsManip.get_data_cut(reg, data)

        if np.max(intense_cut.compressed()) > threshold*image_noise:
            return valid
        else:
            return invalid

    @staticmethod
    def read_region_as_pixels(reg_file, wcs):
        # convert to pixel values otherwise we don't get to_mask method
        regs = Regions.read(reg_file, format="ds9").regions
        for _, reg in enumerate(regs):
            regs[_] = regs[_].to_pixel(wcs)
        return regs

    @classmethod
    def write_valid_regions(cls, regfile, fname, threshold=20, min_rnoise=None, overwrite=True):

        if overwrite:
            # read whatever was written out to begin witfromh
            snitch.info("Determining valid regions")
            with open(regfile, "r") as fil:
                lines = fil.readlines()

            # get wcs information through whatever image is used here in fname
            # Usually and hopefully it is an I image
            wcs = IOUtils.get_wcs(fname)

            regs = IOUtils.read_region_as_pixels(regfile, wcs)
            noise_fname = os.path.join(os.path.dirname(regfile), "noise_area.reg")
            if min_rnoise is None:
                noise_reg, = IOUtils.read_region_as_pixels(noise_fname, wcs)
            else:
                noise_reg = min_rnoise

            #identify what is thought to be valid
            chosen = []
            for _, reg in enumerate(regs):
                vals = cls.choose_valid_regs(fname, reg, noise_reg, threshold=threshold)
                if vals:
                    chosen.append(_)
            
            snitch.warning(f"{len(chosen)} / {len(regs)} regions found to be valid")
            snitch.info(f"Overwriting into {regfile}")
            
            #write back to the same file the valid regions
            with open(regfile, "w") as fil:
                nlines = []
                for i, c in enumerate(chosen, 1):
                    if "los" not in lines[c+3]:
                        new = lines[c+3].split("#")[0] + f" # {i},los text={{reg_{i}}}\n"
                        nlines.append(new)
                    else:
                        nlines.append(lines[c+3])
                nlines = lines[:3] + nlines
                fil.writelines(nlines)

        return


    @classmethod
    def overlay_regions_on_source_plot(cls, regfile, fname, noise_fname, threshold):
        """
        Plotting noise data above noise in the I image. Also overlays the regions
        on the output plot

        regfile:
            Name of region file contained
        fname:
            The I- image to be used for WCS coordinate positions, get the noise value,
            and is what will be plotted. From the location of this image, the rest of
            the images (q|u)-mfs.fits will try and be gotten and plotted as well
        noise_fname: str | float
            Name of the region file containing the noise region, or the noise itself

        The plot will be stored in the regions directory under the directory created
        for the output of this script
        """
        wcs = IOUtils.get_wcs(fname)
        chosen = IOUtils.read_region_as_pixels(regfile, wcs)

        if isinstance(noise_fname, str):
            noise_reg, = IOUtils.read_region_as_pixels(noise_fname, wcs)
            noise = FitsManip.get_noise(noise_reg, fname)
        else:
            noise = noise_fname
        
        plt.close("all")
        fig, ax = plt.subplots(
                figsize=(6, 10), ncols=1, nrows=3, sharex=True, sharey=True,
                gridspec_kw={"wspace":0, "hspace":0})

        ax = ax.flatten()
        snitch.info("Plotting selected regions over I (Q, U) images")
        i_data = None
        for _, im in enumerate("iqu"):
            image = os.path.join(os.path.dirname(fname),f"{im}-mfs.fits")
            # ignore if this file does not exist
            if not os.path.isfile(image):
                snitch.info(f"{image} not found")
                continue

            data = fits.getdata(f"{im}-mfs.fits").squeeze()
            
            if im == "i":
                data = np.ma.masked_less(np.abs(data), noise*threshold)
                i_data = data
            else:
                data = np.ma.masked_array(data=data, mask=i_data.mask)

            lims = np.array([_.center.xy for _ in chosen], dtype=int)
            xmin, ymin = np.min(lims, axis=0)
            xmax, ymax = np.max(lims, axis=0)
            wiggle = 50
            
            ax[_].imshow(data, origin="lower", cmap="coolwarm",
            norm=colors.LogNorm(vmin=data.min(), vmax=data.max())
            # vmin=data.min(), vmax=data.max()
            ) 
            ax[_].set_title(im)
            ax[_].set_xlim(xmin-wiggle, xmax+wiggle)
            ax[_].set_ylim(ymin-wiggle, ymax+wiggle)

            for choice in chosen:
                x,y,r = np.array(np.round([*choice.center.xy, choice.radius]),dtype=int)
                ax[_].add_patch(plt.Circle((x,y), radius=r, color="indigo", fill=False, lw=0.5))

        fig.suptitle(f"thresh x noise: {threshold} x {noise:.6f} = {threshold*noise:.6f}, log scale\n"+
            "Showing only data above this")
        fig.tight_layout()
        oname = os.path.join(
            os.path.dirname(regfile),
            f"iqu-data-above-noise-plot-t{int(threshold)}-r{r}.png")

        if os.path.isfile(oname):
            name, ext = os.path.splitext(oname)
            oname = name + "-a" + ext

        plt.savefig(oname, dpi=400)
        snitch.info(oname)

    
    @staticmethod
    def world_to_pixel_coords(ra, dec, wcs_ref):
        """
        Convert world coordinates to pixel coordinates.
        The assumed reference is FK5
        ra: float
            Right ascension in degrees
        dec: float
            Declination in degrees
        wcs_ref:
            Image to use for WCS information

        Returns
        -------
            x and y pixel information
        """
        if isinstance(wcs_ref, str):
            wcs = IOUtils.get_wcs(wcs_ref)
        else:
            wcs = wcs_ref
        world_coord = FK5(ra=ra*u.deg, dec=dec*u.deg)
        skies = SkyCoord(world_coord)
        x, y = skies.to_pixel(wcs)
        return int(x), int(y)

    @staticmethod
    def get_wcs(wcs_ref, pixels=False):
        wcs = WCS(fits.getheader(wcs_ref))
        if wcs.naxis > 2:
            dropped = wcs.naxis - 2
            # remove extra and superflous axes. These become problematic
            for _ in range(dropped):
                    wcs = wcs.dropaxis(-1)

        if pixels:
            return wcs.pixel_shape
        else:
            return wcs
    
    @staticmethod
    def write_noise_region(reg_fname):
        snitch.warning("Also writting default noise region")
        header = [
            "# Region file format: DS9 CARTA 2.0.0",
            ('global color=#2EE6D6 dashlist=8 3 width=2 ' +
            'font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 ' +
            'edit=1 move=1 delete=1 include=1 source=1'),
            "FK5"
        ]

        with open(os.path.join(os.path.dirname(reg_fname), "noise_area.reg"), "w") as fil:
            fil.writelines(
                [p+"\n" for p in header+["circle(80.041875, -45.713452, 13.0000\")"]])
            snitch.info("Noise region written")

    @staticmethod
    def generate_regions(reg_fname, wcs_ref, factor=50, overwrite=True):
        """
        Ref: https://ds9.si.edu/doc/ref/region.html
        Create a DS9 region file containing a bunch of regions
        factor: int
            In my case, the size of the region ie length / widh t
            reg_fname:
                File name for the resulting region files
            max_*:
                Maximum pixel image height or width. 
                So that regions don't go beyound image dims
        """
        # converted by
        # pix = regions.PixCoord(569, 450).to_sky(wcs)
    
        # left to right
        # ra in degrees
        w_range = 80.04166306500294, 79.84454319889994

        # bottom to top
        # dec in degrees
        h_range = -45.81799666164118, -45.73325018138195

        max_w, max_h =  IOUtils.get_wcs(wcs_ref, pixels=True)
        wcs =  IOUtils.get_wcs(wcs_ref)
        start =  IOUtils.world_to_pixel_coords(w_range[0], h_range[0], wcs)
        end =  IOUtils.world_to_pixel_coords(w_range[1], h_range[1], wcs)

        width_range, height_range = (start[0], end[0]), (start[1], end[1])

        header = [
            "# Region file format: DS9 CARTA 2.0.0",
            ('global color=#2EE6D6 dashlist=8 3 width=2 ' +
            'font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 ' +
            'edit=1 move=1 delete=1 include=1 source=1'),
            "FK5"
        ]

        pts = []
        count = 0
        heights = list(range(*height_range, factor))
        widths = list(range(*width_range, factor))

        if max(heights) < max(height_range):
            heights.append(max(height_range))
            
        if min(heights) < min(height_range):
            heights.append(min(height_range))


        if max(widths) < max(width_range):
            widths.append(max(width_range))
        if min(widths) < min(width_range):
            widths.append(min(width_range))

        for height in range(*height_range, factor*2):
            for width in range(*width_range, factor*2):
                width = max_h if width > max_w else width
                height = max_h if height > max_h else height
        
                # sky = RectangulePixelRegion(PixCoord(width, height),
                #             width=factor, height=factor).to_sky(wcs)
                sky = CirclePixelRegion(PixCoord(width, height), radius=factor).to_sky(wcs)

                pts.append("circle({:.6f}, {:.6f}, {:.6f}\") # text={{reg_{}}}".format(
                            sky.center.ra.deg, sky.center.dec.deg, sky.radius.arcsec, count))
                count += 1

        
        if ".reg" not in reg_fname:
            reg_fname += f"-r{factor}.reg"

        if not os.path.isfile(reg_fname) or overwrite:
            with open(reg_fname, "w") as fil:
                pts = [p+"\n" for p in header+pts]
                fil.writelines(pts)
            snitch.warning(f"Regions file written to: {reg_fname}")
        else:
            snitch.warning(f"File already available at: {reg_fname}")

        
        IOUtils.write_noise_region(reg_fname)
            
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
                
                snitch.info(f"Starting the saving process: Group {int(i/grid_size_sq)}")
                fig.tight_layout(h_pad=3)
                legs = cls.active_legends(fig)
                fig.legend(legs, bbox_to_anchor=(1, 1.01), markerscale=3, ncol=len(legs))

                # fig.suptitle("Q and U vs $\lambda^2$")
                oname = f"{outfile}-{int(i/grid_size_sq)}-{xscale}"
                
                plt.setp(sp[:,0], ylabel="Frac Pol")
                plt.setp(sp[-1,:], xlabel="Wavelength m$^2$")
        
                fig.savefig(oname, bbox_inches='tight')
                plt.close("all")
                snitch.warning(f"Plotting done for {oname}")
        snitch.info(f"We have: {plots}/{n_qf} plots")

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

        data_files = sorted(glob(f"./iqu-{file_core}/*.npz"), key=fight)
        n_qf = len(data_files)
        snitch.info(f"Found {n_qf} QUI files")

        snitch.info("Starting plots")
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
            snitch.info(f"Plotting done for {oname}")
        snitch.info(f"We have: {plots}/{n_qf} plots")
        # snitch.info(f"Regions with >4 data points in fracpol {data_points}")



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
    def linear_polzn_power(cls, stokes_q, stokes_u):
        lin_pol = np.abs(cls.linear_polzn(stokes_q, stokes_u))
        return lin_pol
    
    @classmethod
    def linear_polzn_power_error(cls, p, q, u, q_err, u_err):
        res = np.square((q/p)) * np.square(q_err) + np.square((u/p)) * np.square(u_err)
        return np.sqrt(res)

    @classmethod
    def linear_polzn(cls, stokes_q, stokes_u):
        lin_pol = stokes_q + (1j*stokes_u)
        return lin_pol

    @classmethod
    def fractional_polzn(cls, stokes_i, stokes_q, stokes_u):
        frac_pol = cls.linear_polzn_power(stokes_q, stokes_u) / stokes_i
        return frac_pol
    
    @classmethod
    def fractional_polzn_error(cls, stokes_i, stokes_q, stokes_u, i_err, q_err, u_err):
        fpol = cls.fractional_polzn(stokes_i, stokes_q, stokes_u)
        p = cls.linear_polzn_power(stokes_q, stokes_u)
        p_err = cls.linear_polzn_power_error(p, stokes_q, stokes_u, q_err, u_err)
        res = np.abs(fpol) * np.sqrt(np.square((p_err/p)) + np.square((i_err/stokes_i)))
        return res
    
    @staticmethod
    def polarisation_angle(q, u):
        return 0.5 *np.arctan2(u, q)

    @staticmethod
    def polarisation_angle_error(p, q, u, q_err, u_err):
        "See A.11 Brentjens"
        err = (np.square(u*q_err) + np.square(q*u_err)) / (4 * p**4)
        return np.rad2deg(np.sqrt(err))

    @classmethod
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
        # fi the noise_reg is a number, return the same, otherwise, act
        if isinstance(noise_reg, float) or isinstance(noise_reg, int):
            noise = noise_reg
        else:
            if data is None:
                im_data = cls.get_useful_data(fname)
                data = im_data["data"]
            # Get image noise from standard deviation of a sourceless region
            noise_cut = cls.get_data_cut(noise_reg, data)
            # noise = np.nanstd(noise_cut)
            noise = MathUtils.rms(noise_cut)
        return noise


    @classmethod
    def extract_stats2(cls, fname, reg, global_noise, noise_reg, sig_factor=10):
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

        image_noise = cls.get_noise(noise_reg, fname, data=data)
        global_noise = global_noise[im_data["stokes"].lower()]
        
        # Get mean flux over the pixels
        intense_cut = cls.get_data_cut(reg, data)

        """
        Check if all values are 
        - nans
        - zeros
        - if the noise is infinite
        """
        if (MathUtils.are_all_nan(intense_cut) or MathUtils.are_all_zeroes(intense_cut) or
            MathUtils.is_infinite(image_noise)):
            # skip all the nonsence if all the data is Nan
            snitch.debug(f"Skipping region:{reg.meta['text']} {fname} because NaN/Zeroes/inf ")
            # im_data["flux_jybm"] = im_data["flux_jy"] = None
            im_data["flux_jybm"] = im_data["noise"] = im_data["image_noise"] = None
            return im_data
        
        #Using the center pixel
        # cpixx, cpixy = np.ceil(np.array(intense_cut.shape)/2).astype(int)
        # flux_jybm = intense_cut[cpixx, cpixy]
        ban = intense_cut.compressed()
        flux_jybm = ban[ban.size//2]


        if (np.abs(flux_jybm) > sig_factor * image_noise):
            im_data["flux_jybm"] = flux_jybm
        else:
            im_data["flux_jybm"] = None
    
        
        im_data["noise"] = global_noise
        im_data["image_noise"] = image_noise

        # Noting here that individual image noise is always > the MFS noise
        # snitch.warning(f"image noise > global nose: {image_noise > global_noise}")
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

        get the weighted cut
        # see: https://astropy-regions.readthedocs.io/en/stable/masks.html?highlight=cutout#making-image-cutouts-and-multiplying-the-region-mask

        reg: :obj:`regions.Region`
            Region of interest
        data: :obj:`np.ndarray`
            Data array of a given fits file
        """
        reg_mask = reg.to_mask()
        weighted_data_cut = reg_mask.multiply(data)
        weighted_data_cut = np.ma.masked_equal(weighted_data_cut,0)
        return weighted_data_cut

    @classmethod
    def get_image_stats2(cls, file_core, images, regs, global_noise, noise_reg,
        sig_factor, output_dir="scrap-outputs"):
        fluxes, waves = [], []
        snitch.info("starting get_image_stats")
        for reg in regs:
            snitch.info(f"Region: {reg.meta['text']}")
            with futures.ProcessPoolExecutor(max_workers=16) as executor:
                results = executor.map(
                    partial(cls.extract_stats2, reg=reg, global_noise=global_noise,
                    noise_reg=noise_reg, sig_factor=sig_factor), images
                    )
            
            # # some testbed DO NOT DELETE !!!!!!!
            # results = []
            # for im in images:
            #     results.append(
            #         cls.extract_stats2(im, reg=reg, global_noise=global_noise, 
            #             noise_reg=noise_reg, sig_factor=sig_factor)
            #         )

            
            outs = {_: {"flux_jybm": [], "freqs": [],"fnames": [], "noise": [], "image_noise": []}
                        for _ in "IQU"}
            
            for res in results:
                if res["stokes"] not in outs:
                    continue
                outs[res["stokes"]]["flux_jybm"].append(res["flux_jybm"])
                outs[res["stokes"]]["freqs"].append(res["freqs"])
                outs[res["stokes"]]["fnames"].append(res["fnames"])
                # global noise i.e from respective stoke Mfs image
                outs[res["stokes"]]["noise"].append(res["noise"])
                # noise for this specific image
                outs[res["stokes"]]["image_noise"].append(res["image_noise"])

            checks = [outs.get(k)["flux_jybm"] for k in "IQU"][0]

            if not all(_ is None for _ in checks):
                # contains I, Q, U, noise, freqs, fpol, lpol
                fout = {k: np.asarray(v["flux_jybm"], dtype=np.float) for k,v in outs.items()}
                fout["freqs"] = outs["I"]["freqs"]
                fout.update({f"{k}_mfs_noise".lower(): outs[k]["noise"] for k in "IQU"})
                fout.update({f"{v}_err": outs[v]["image_noise"] for v in "IQU"})
                
                fout["lpol"] = MathUtils.linear_polzn(fout["Q"], fout["U"])
                fout["lpol_err"] = MathUtils.linear_polzn_power_error(
                    fout["lpol"], fout["Q"], fout["U"],
                    fout["Q_err"], fout["U_err"])


                fout["fpol"] = MathUtils.fractional_polzn(fout["I"], fout["Q"],
                    fout["U"])
                fout["fpol_err"] = MathUtils.fractional_polzn_error(fout["I"],
                    fout["Q"], fout["U"], fout["I_err"], fout["Q_err"], fout["U_err"])

                fout["pangle"] = MathUtils.polarisation_angle(fout["Q"], fout["U"])
                fout["pangle_err"] = MathUtils.polarisation_angle_error(
                    fout["pangle"], fout["Q"], fout["U"], fout["Q_err"], fout["U_err"])

                # creating mask where ll values where fractional polz > 1 or <0
                mask = np.ma.masked_where(
                            (fout["fpol"]>1) | 
                            (fout["fpol"]<0) | 
                            (np.isnan(fout["fpol"])),
                        fout["fpol"] ).mask
                fout["mask"] = mask

                # if not all data is flagged and flagged data is less than 50%
                if (mask.sum() <= mask.size*MFT):
                    if mask.sum()>0:
                        snitch.warning(f"{reg.meta['text']}: flagged {mask.sum()}/{mask.size} points")
                    out_dir = IOUtils.make_out_dir(
                        os.path.join(output_dir, f"iqu-{file_core}"))
                    outfile = os.path.join(out_dir, f"{reg.meta['text']}")
                    np.savez(outfile, **fout)
                else:
                    snitch.warning(
                        f"{reg.meta['text']}: flagged {mask.sum()}/{mask.size} points. " + 
                        F"Skipping because >{MFT*100}% flagged, or not enough data.")
                    FPOL_FILTER.append(reg.meta['text'])
            else:
                snitch.warning(f"Skipping because there's no valid data in this region.")
            
        snitch.info(f"Done saving data files")
        snitch.info( "--------------------")
        return FPOL_FILTER



def parser():
    from textwrap import fill, dedent
    class BespokeFormatter(argparse.RawDescriptionHelpFormatter):
        def _fill_text(self, text, width, indent):
            wrap_width = 80
            return "_"*wrap_width + "\n\n Description\n\n" +\
                "\n".join([fill(dedent(_), width=wrap_width,
                                initial_indent=" ", subsequent_indent="  ",
                                tabsize=2)
                            for _ in text.splitlines()]) + \
                "\n" + "_"*wrap_width

    parsing = argparse.ArgumentParser(usage="%(prog)s [options]", add_help=True,
        formatter_class=BespokeFormatter,
        description="""
            Generate I, Q and U data for various LoS from image cubes.
            
            This script uses the total intesity MFS image to determine regions
            of the source with enough SNR and selects those as the lines of sight.
            For now, this is tuned specifically for Pictor A, but it may be 
            extended and revised in the future.

            The following could be generated from this script:

            .\t1. Region file containing the selected lines of sight
            .\t2. Plot showing locations of the regions on the source
            .\t3. Pickle files containing the following keys for the data:
            
                .\t\t- I \t\t - Q \t \t - U
                .\t\t- i_mfs_noise \t - q_mfs_noise \t - u_mfs_noise
                .\t\t- I_err \t - Q_err \t - U_err
                .\t\t- lpol \t\t - lpol_err
                .\t\t- fpol \t\t - fpol_err
                .\t\t- pangle \t - pangle_err
                .\t\t- mask \t\t - freqs
            .\ti.e for a single line of sight. Each LoS contains data for all
            .\tthe available channels
.\t
            .\t4. Optionaly, generate plots for fractional polarisation vs 
            lambda squared for each LoS
                    
            Examples
            ========

            plotting only
            $ python scrap.py -p 50 20 -t mzima-t10 -plp -piqu --plot-grid
            
            stats and plotting
            $ python scrap.py -f clean-small.txt -rs 50 20 -ap 
                -t mzima-t10-v2 --threshold 10 --noise -0.0004 -ap -plp -piqu
            
            with region files
            $ python scrap.py -rf regions/beacons-20-chosen.reg -f clean-small.txt
                -rs 20 -t chosen --threshold 10
            """
        )

    req_parsing = parsing.add_argument_group("Required Arguments")

    req_parsing.add_argument("-f", "--infile", dest="in_list", type=str,
        metavar="", required=True,
        help="File containing an organised list of the input image names."+
        "Can easily be done with 'ls *-image.fits > whatever.txt'")
    req_parsing.add_argument("-wcs-ref", "--wcs-reference", dest="wcs_ref",
        metavar="", default=None, required=True,
        help=("The image to use to get the reference WCS for region file " +
            "genenration. Not required if only plotting."))
    

    # Optional arguments
    opt_parsing = parsing.add_argument_group("Optional Arguments")

    opt_parsing.add_argument("-mft", "--minimum-flag-threshold", dest="mft",
    type=float, default=None, help="""
    Fraction of flags above which lines of sight should be ignored.  
    Can be useful if you want to plot all the generated LOS. Otherwise, they
    will be filtered out. The simple filter is that where values of fractional
    polarisation i are >1 or <0, this data is 'flagged'.
    behaviour: flag size > 0.7 of total data is flagged, ingore. 
    Max is 1, min is >0.
    Default 0.7
    """
    )
    opt_parsing.add_argument("--noverwrite", dest="noverwrite", action="store_false",
        help="Do not ovewrite everything along the way")
    opt_parsing.add_argument("-ro", "--regions-only", dest="r_only",
        action="store_true", help="Generate only region files with this script.")
    opt_parsing.add_argument("-rf", "--region-file", dest="reg_file", type=str,
        default=None, metavar="", 
        help="An input region file. Otherwise, one will be auto-generated.")
    opt_parsing.add_argument("-mrn", "--minimum-region-noise", dest="mrn", type=float,
        default=None, metavar="", 
        help="""If signal is above this noise * threshold in the I-MFS image,
        that will be set as a valid region. Otherwise we get noise automatically
        from an off source region in I-MFS. Note that It is used in conjunction with
        --threshold.
        """)
    opt_parsing.add_argument("-rs", "--region-size", dest="reg_size", nargs="+",
        type=int, default=[], metavar="", 
        help=("Create regions of this circle radius and perform analyses on them."+
        " If you want to set the data threshold, please use --threshold."))
    opt_parsing.add_argument("--threshold", dest="thresh", metavar="", type=int,
        default=10,
        help=("Noise factor threshold above which to extract."+
            " This will be threshold * RMS noise. This is what is also used to generate regions." +  
            " Noise is gotten from the I-MFS image."))
    opt_parsing.add_argument("-o", "--output-dir", dest="output_dir", type=str,
        default="scrap-outputs", metavar="",
        help="where to dump output")
    opt_parsing.add_argument("-t", "--testing", dest="testing", metavar="",
        type=str, default=None,
        help="Testing affixation. Will be Prepended with '-'. Default name is IQU something")
    opt_parsing.add_argument("--noise", dest="noise", metavar="", type=float,
        default=None, nargs="+",
        help="""
        Noise value above which to extract. If a single value is provided, this 
        will be the same noise used for the I, Q, and U images. If you wish to use
        different noise for each image, give this in three separated values.
        The Default values will be calculated from the (I|Q|U)MFS-images. 
        Must be in the order of I Q U.
        """)
    opt_parsing.add_argument("-nrfs", "--noise-refs", dest="nrfs", metavar="",
        default=["i-mfs.fits", "q-mfs.fits", "u-mfs.fits"], type=str, nargs="+",
        help="""The IQU images to use as noise refernces. If not supplied, this script
        will use the available (I|Q|U)-mfs images it finds. Must be in the order
        I Q U.
        """
    )
    
    
    #plotting arguments
    plot_parsing = parsing.add_argument_group("Plotting Arguments")
    plot_parsing.add_argument("--plot-grid", dest="plot_grid",
        action="store_true",
        help="Enable to make gridded plots")
    plot_parsing.add_argument("-ap", "--auto-plot", dest="auto_plot",
        action="store_true",
        help="Plot all the specified region pixels")
    plot_parsing.add_argument("-piqu", "--plot-iqu", dest="plot_qu",
        action="store_true",
        help="Plot Q and U values")
    plot_parsing.add_argument("-pfp", "--plot-frac-pol", dest="plot_frac_pol",
        action="store_true",
        help="Plot Fractional polarization")
    plot_parsing.add_argument("-plp", "--plot-linear-pol",
        dest="plot_linear_pol", action="store_true",
        help="Plot linear polarization power")
    plot_parsing.add_argument("--ymax", dest="ymax", type=float, 
        help="Y axis max limit")
    plot_parsing.add_argument("--ymin", dest="ymin", type=float,
        help="Y axis min limit")
    plot_parsing.add_argument("--xmax", dest="xmax", type=float, 
        help="Y axis max limit")
    plot_parsing.add_argument("--xmin", dest="xmin", type=float,
        help="Y axis min limit")
    plot_parsing.add_argument("-p", "--plot", dest="plot", nargs="*",
        type=int, metavar="",
        help="Make plots for these region sizes manually. These will be linearly scaled")
    plot_parsing.add_argument("-ps", "--plot-scales", dest="plot_scales", metavar="",
        default=["linear"], nargs="*", choices=["linear", "log"],
        help=("Scales for the plots. Can be a space separated list of " + 
            "different scales. Options are linear or log."))

    return parsing


if __name__ == "__main__":
    opts = parser().parse_args()
    snitch = configure_logger(opts.output_dir)
    
    MFT = opts.mft if opts.mft else MFT

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
            images = IOUtils.read_sorted_filnames(opts.in_list)
            nrfs = sorted(opts.nrfs)
            
            reg_dir = IOUtils.make_out_dir(os.path.join(opts.output_dir, "regions"))
            if opts.reg_file:
                reg_file = shutil.copy(opts.reg_file, reg_dir)
                IOUtils.write_noise_region(reg_file)
                # reg_file = os.path.basename(opts.reg_file)
                snitch.warning(f"Using {reg_file} as region file")
            else:
                # factor here is the size of radius of box or circle
                reg_dir = IOUtils.make_out_dir(os.path.join(opts.output_dir, "regions"))
                reg_file = IOUtils.generate_regions(
                    os.path.join(reg_dir, f"beacons-thresh-{opts.thresh}"), 
                    wcs_ref=opts.wcs_ref,
                    factor=factor, overwrite=opts.noverwrite)
                # because not user specified, I can edit however I want
                # will use the wcs ref as where to determin noise
                # this must be the I-MFS image
                snitch.warning(f"Using the noise from :   {nrfs[0]} to generate regions.")
                IOUtils.write_valid_regions(reg_file, fname=nrfs[0],
                    threshold=opts.thresh, min_rnoise=opts.mrn, overwrite=opts.noverwrite)

            
            ref_wcs = IOUtils.get_wcs(opts.wcs_ref)
            regs = IOUtils.read_region_as_pixels(reg_file, ref_wcs)
            
            snitch.info(f"Working on Stokes IQU")
            snitch.info(f"With {len(regs)} regions")
            snitch.info(f"And {len(images)} images ({len(images)//3} X IQU)")

            noise_reg, = IOUtils.read_region_as_pixels(
                    os.path.join(os.path.dirname(reg_file), "noise_area.reg"),
                    ref_wcs)

            if opts.noise:
                global_noise = [opts.noise[0]]*3 if len(opts.noise)==1 else opts.noise
                global_noise = {_i: _n for _i, _n in zip("iqu", global_noise)}
            else:
                snitch.info(f"Getting RMS noise(s) from {', '.join(opts.nrfs)}")
                global_noise = {}
                for _i , nrf in zip("iqu", sorted(opts.nrfs)):
                    global_noise[_i] = FitsManip.get_noise(noise_reg, nrf, data=None)

            _noise_ref = global_noise["i"]
            
            # generate region files only so end the loop. On to the next one
            snitch.warning(f"Noise factor :   {opts.thresh:8.3f}")
            for _k, _v in global_noise.items():
                snitch.warning(f"Noise for {_k}-mfs :   {_v:8.8f}")
                snitch.warning(f"Noise threshold {_k}-mfs :   {opts.thresh*_v:8.8f}")  
            snitch.warning("...............................................")
            IOUtils.overlay_regions_on_source_plot(
                    reg_file, opts.wcs_ref, opts.mrn or _noise_ref, opts.thresh)
            if opts.r_only:
                snitch.warning("Generated regions only as per request")
                continue
            

            fpol_filter = FitsManip.get_image_stats2(
                file_core, images, regs, global_noise, noise_reg,
                sig_factor=opts.thresh, output_dir=opts.output_dir)

            if len(fpol_filter)>0:
                snitch.warning("Removing the following regions: ")
                snitch.warning(pformat(" ".join(fpol_filter), compact=True, width=100))
                with open(reg_file, "r") as fname:
                    edited = dict()
                    for i, x in enumerate(fname.readlines()):
                        if i <3:
                            key = f"l{i}"
                        else:
                            key = int(x.split("{")[-1].split("}")[0].split("_")[-1])
                        edited[key] = x

                for _ in fpol_filter:
                    # coz list indexing starts from 0
                    del edited[int(_.split("_")[-1])]

                edited = list(edited.values())
                for _, _x in enumerate(edited[3:], 3):
                     edited[_] = edited[_].split("#")[0] + f"# {_-2},los\n"

                # retain the original region file and write into moded one
                _=shutil.copy(reg_file, os.path.splitext(reg_file)[0]+"-orig.reg")
                with open(reg_file, "w") as fname:
                    fname.writelines(edited)
                
                IOUtils.overlay_regions_on_source_plot(
                    reg_file, opts.wcs_ref, opts.mrn or _noise_ref, opts.thresh)


            if opts.auto_plot:
                snitch.info("Autoplotting is enabled")
                plot_dir = IOUtils.make_out_dir(
                    os.path.join(opts.output_dir, f"plots-iqu-{file_core}"))
                pout = os.path.join(plot_dir,  f"iqu-{file_core}")
                plotter(file_core, 
                    f"{plot_dir}/iqu-regions-mpc{testing}-{factor}",
                    ymin=opts.ymin, ymax=opts.ymax, xmin=opts.xmin, xmax=opts.xmax,
                    plot_qu=opts.plot_qu, plot_frac_pol=opts.plot_frac_pol,
                    plot_linear_pol=opts.plot_linear_pol)

            snitch.info(f"Finished factor {factor} in {perf_counter() - start} seconds")
            snitch.info("======================================")

    if opts.plot:
        " python scrap.py -p 50 -t toto"

        snitch.info(f"Plotting is enabled for regions {opts.plot}")
        for factor in opts.plot:
            for scale in opts.plot_scales:
                file_core = f"regions-mpc-{factor}{testing}"
                plot_dir = IOUtils.make_out_dir(
                    os.path.join(opts.output_dir, f"plots-iqu-{file_core}"))
                pout = os.path.join(plot_dir,  f"iqu-{file_core}")
                plotter(file_core, 
                    f"{plot_dir}/iqu-regions-mpc{testing}-{factor}",
                    xscale=scale, ymin=opts.ymin, ymax=opts.ymax,
                    xmin=opts.xmin, xmax=opts.xmax, plot_qu=opts.plot_qu,
                    plot_frac_pol=opts.plot_frac_pol,
                    plot_linear_pol=opts.plot_linear_pol)