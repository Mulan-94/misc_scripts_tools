import numpy as np
import argparse
import matplotlib.pyplot as plt
import warnings

from astropy.io import fits
from astropy.wcs import WCS
from glob import glob
from matplotlib import ticker
from glob import glob
import scipy.ndimage as snd

from matplotlib import colors
from matplotlib.ticker import PercentFormatter

import sys
import os

PATH = set(sys.path)
append = [
    f"{os.environ['HOME']}/git_repos/misc_scripts_n_tools/fits_related/",
    f"{os.environ['HOME']}/git_repos/misc_scripts_n_tools/qu_pol/scrappy/"
]

for ap in append:
    if not PATH.issuperset(ap):
        sys.path.append(ap)



import random_fits_utils as rfu
from utils.rmmath import frac_polzn_error, frac_polzn
from ipdb import set_trace


"""
A perfect reference for the quiver API

- https://www.pythonpool.com/matplotlib-quiver/


The following two snippets of code generate the same result. The only difference
is that:

1. This snippet provides values for:
  - the x- and y- coordinates of where the vectors will be located
  - With an assumed vector magnitude of 1, the x and y resolved components
    of the vector
    (u, v)

2. This snippet provides values for:
  - The magnitude of the vectors in x and y (u, v)
  - The orientation of the vector specified as an ANGLE!

Therefore, if we want all the vectors to have the same magnitudes but different
 orientation
Snippet 2 is the best obvious choice. Otherwise, if we want to vary the
 magnitudes of the vectors, choose snippet 1.

IF YOU USE ANGLES, SET U AND V TO 1
IF YOU USE RESOLVED U AND V, DON'T USE ANGLES. TOO MUCH WORK

# polarization angle starts from 0 North, apply this offset
ANGLE_OFFSET = 90

specs = dict(pivot='tail', headlength=0, width=0.012, scale=5, headwidth=1)

# snippet 1: same origin, different orientations
plt.quiver(0, 0, np.sin(0), np.cos(0),**specs, color="red");
plt.quiver(0, 0, np.sin(45), np.cos(45),**specs, color="green");
plt.quiver(0, 0, np.sin(90), np.cos(90),**specs, color="black");



# snippet 2: arbitrary origin, same manitude, different orientations, we can 
# specify the origins here. If we use angles, we don't have to resolve the x 
# and y components
plt.quiver(1, 1, angles=[0 + ANGLE_OFFSET],**specs, color="red");
plt.quiver(1, 1, angles=[45 + ANGLE_OFFSET],**specs, color="green");
plt.quiver(1, 1, angles=[90 + ANGLE_OFFSET],**specs, color="black");

# ANGLES MUST BE IN DEGREES!

! PolarIZATION ANGLE STARTS FROM THE NORTH!!!!!!!
https://link.springer.com/content/pdf/10.1007/s10686-016-9517-y.pdf
"""


warnings.filterwarnings("ignore", module="astropy")

spectest = "/home/andati/pica/reduction/testing_spectra"

FIGSIZE = (16,9)
EXT = ".png"
DPI = 100

# by how much to rotate the e vectors
ROT = 90

#For Polarization angles IAU Convention for 0 degree is at NORT, 
# but angles in matplotlIb start at x=0. Thus correct angle must be offset
ANGLE_OFFSET = 90

# Where we are dumping the output
global PFIGS
PFIGS = "paper-figs"

plt.style.use("seaborn")

def set_image_projection(image_obj):
    """
    image_obj:
        Matplotlib axis object from image e.g ax from
        fig, ax = plt.subplots(subplots_kw={"projection": wcs})
    """
    image_obj.get_transform('fk5')
    ra, dec = image_obj.coords
    ra.set_major_formatter('hh:mm:ss')
    dec.set_major_formatter('dd:mm:ss')
    image_obj.set_xlabel('J2000 Right Ascension')
    image_obj.set_ylabel('J2000 Declination')
    return image_obj



############################################################################
## contours plotting and vectors
############################################################################

# ref https://stackoverflow.com/questions/40939821/how-to-plot-a-vector-field-over-a-contour-plot-in-matplotlib


class PaperPlots:

    @staticmethod
    def figure_3_total_and_jets(i_image, mask, jet_mask, start=0.004, vmin=0,
            vmax=None, scale="linear", cmap="magma",
            output=f"{PFIGS}/3-total-intensity", kwargs=dict()):
        
        plt.close("all")
        """
        i_image:
            The data without masking. Will be zoomed in anyway]
        """
        scaling = {
            "linear": colors.Normalize,
            "log": colors.LogNorm,
            "symmetric": colors.SymLogNorm,
            "centered": colors.CenteredNorm,
            "power": colors.PowerNorm
        }
        wcs = rfu.read_image_cube(i_image)["wcs"]
        i_data = rfu.get_masked_data(i_image, mask)

        i_data = np.ma.masked_where(i_data<start, i_data)
        mask = i_data.mask


        jet_mask = np.logical_not(fits.getdata(jet_mask).squeeze())

        levels = contour_levels(start, i_data)

        wiggle = 20
        y, x = np.where(mask == False)
        jy, jx = np.where(jet_mask == False)
        
        
        xr = np.min(x)-wiggle, np.max(x)+wiggle
        yr = np.min(y)-wiggle, np.max(y)+wiggle

        # jet limits
        jxr = np.min(jx)-wiggle, np.max(jx)+wiggle
        jyr = np.min(jy)-wiggle, np.max(jy)+wiggle

        if vmax is None:
            vmax = np.percentile(i_data.compressed(), 70)

        if vmin<start or vmin is None:
            vmin = start

        use_scale = scaling.get(scale, "linear")(vmin=vmin, vmax=vmax, **kwargs)


        ####################################################################
        # Plotting the Total intesity WITHOUT contours
        ####################################################################
        plt.close("all")
        fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
        ax = set_image_projection(ax)
        cs = ax.imshow(i_data, origin="lower", norm=use_scale, cmap=cmap)
        ax.set_facecolor("black")
        plt.colorbar(cs, label="Total Intensity [Jy/bm]", pad=0, shrink=.95,)
        ax.set_xlim(*xr)
        ax.set_ylim(*yr)
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)
        # ax.set_facecolor("white")
        fig.canvas.draw()
        fig.tight_layout()
        print("Saving total intensity image at:   "+ output)
        fig.savefig(output+".png", dpi=DPI)
        

        ####################################################################
        # Plotting the Total intesity with contours
        ####################################################################
        plt.close("all")
        fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
        ax = set_image_projection(ax)
        cs = ax.imshow(i_data, origin="lower", norm=use_scale, cmap=cmap)
        ax.set_facecolor("black")
        ax.contour(i_data, colors="g", linewidths=1, origin="lower", levels=levels)
        plt.colorbar(cs, label="Total Intensity [Jy/bm]", pad=0, shrink=.95,)
        ax.set_xlim(*xr)
        ax.set_ylim(*yr)
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)
        fig.canvas.draw()
        fig.tight_layout()
        print("Saving total intensity image at:   "+ output+"-cont")
        fig.savefig(output+"-cont.png", dpi=DPI)


        ####################################################################
        # Plotting the jets now
        ####################################################################
        plt.close("all")
        fig, ax = plt.subplots(figsize=(16,7), subplot_kw={'projection': wcs})
        ax = set_image_projection(ax)
        cs = ax.imshow(i_data, origin="lower", norm=use_scale, cmap=cmap)
        ax.contour(i_data, colors="g", linewidths=1, origin="lower", levels=levels)
        plt.colorbar(cs, label="Total Intensity [Jy/bm]", pad=0.005,
            location="top", shrink=1, fraction=0.112)
        ax.set_xlim(*jxr)
        ax.set_ylim(*jyr)
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)
        fig.canvas.draw()
        fig.tight_layout()
        print("Saving jet image at:               "+ output)
        fig.savefig(output+"-jet.png", dpi=DPI)
        return


    @staticmethod
    def figure_3b_chandra(i_image, mask, jet_mask, chandra=None, chandra_jet=None, 
        start=4e-8, smooth_sigma=1, vmin=0, vmax=2, scale="linear", cmap="magma",
        output=f"{PFIGS}/chandra-3-total-intensity", kwargs=dict()):
        """
        i_image:
            The data without masking. Will be zoomed in anyway
        mask:
            The masked used from cleaning
        jet_mask:
            Mask for this jet
        
        """

        plt.close("all")
        scaling = {
            "linear": colors.Normalize,
            "log": colors.LogNorm,
            "symmetric": colors.SymLogNorm,
            "centered": colors.CenteredNorm,
            "power": colors.PowerNorm
        }
        wcs = rfu.read_image_cube(i_image)["wcs"]
        i_data = rfu.get_masked_data(i_image, mask)
        chandra_data = rfu.get_masked_data(chandra, mask)
        mask = i_data.mask

        i_data = np.ma.masked_where(i_data<start, i_data)

        jet_mask = np.logical_not(fits.getdata(jet_mask).squeeze())

        levels = contour_levels(start, chandra_data)

        wiggle = 20
        y, x = np.where(mask == False)
        jy, jx = np.where(jet_mask == False)
        
        
        
        xr = np.min(x)-wiggle, np.max(x)+wiggle
        yr = np.min(y)-wiggle, np.max(y)+wiggle

        # jet limits
        jxr = np.min(jx)-wiggle, np.max(jx)+wiggle
        jyr = np.min(jy)-wiggle, np.max(jy)+wiggle

        use_scale = scaling.get(scale, "linear")(vmin=vmin, vmax=vmax, **kwargs)       

        ####################################################################
        # Plotting the Total intesity with contours
        ####################################################################
        plt.close("all")
        fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
        ax = set_image_projection(ax)
        cs = ax.imshow(i_data, origin="lower", norm=use_scale, cmap=cmap)
        ax.set_facecolor("black")

    
        # ax.contour(chandra_data, colors="g", linewidths=1, origin="lower", levels=levels)
        ax.contour(snd.gaussian_filter(chandra_data, sigma=smooth_sigma), colors="g",
            linewidths=1, origin="lower", levels=levels)

        plt.colorbar(cs, label="Total Intensity [Jy/bm]", pad=0, shrink=.95,)
        ax.set_xlim(*xr)
        ax.set_ylim(*yr)
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)
        fig.canvas.draw()
        fig.tight_layout()
        print("Saving total intensity image at:   "+ output+"-cont")
        fig.savefig(output+"-cont.png", dpi=DPI)


        ####################################################################
        # Plotting the jets now
        ####################################################################
        plt.close("all")
        fig, ax = plt.subplots(figsize=(16,7), subplot_kw={'projection': wcs})
        ax = set_image_projection(ax)
        cs = ax.imshow(i_data, origin="lower", norm=use_scale, cmap=cmap)
        ax.contour(snd.gaussian_filter(chandra_data, sigma=smooth_sigma), colors="g",
            linewidths=1, origin="lower", levels=levels)
        plt.colorbar(cs, label="Total Intensity [Jy/bm]", pad=0.005,
            location="top", shrink=1, fraction=0.112)
        ax.set_xlim(*jxr)
        ax.set_ylim(*jyr)
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)
        fig.canvas.draw()
        fig.tight_layout()
        print("Saving jet image at:               "+ output)
        fig.savefig(output+"-jet.png", dpi=DPI)
        return
    
    @staticmethod
    def figure_4b_intensity_contours(i_image, mask, start=0.004, smooth_sigma=13,
        output="{PFIGS}/4b-intensity-contours.png"):

        i_data = rfu.get_masked_data(i_image, mask)
        # # replace nans with zeroes. Remember in mask images, the NOT MAsked area is set to 1
        i_data[np.where(np.isnan(i_data))] = 0
        mask = i_data.mask

        i_data = np.ma.masked_where(i_data<start, i_data)

        ydim, xdim = np.where(mask == False)
        wiggle = 20
        xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
        ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle

        wcs = rfu.read_image_cube(i_image)["wcs"]

        fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
        ax = set_image_projection(ax)

        # the image data
        levels = contour_levels(start, i_data)
        ax.contour(
            snd.gaussian_filter(i_data, sigma=smooth_sigma),
            colors="g", linewidths=0.5, origin="lower", levels=levels)

        cs = ax.imshow(i_data, origin="lower", cmap="coolwarm", vmin=0, vmax=.127,
                        aspect="equal")
        plt.colorbar(cs, label="Total Intensity [Jy/bm]", pad=0)
    
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)

        plt.xlim(*xlims)
        plt.ylim(*ylims)
        fig.canvas.draw()
        fig.tight_layout()
        
        print(f"Saving plot: {output}")
        
        plt.savefig(output, dpi=DPI)
        plt.close("all")

    
    @staticmethod
    def table2_core_flux_wfreq(cube, region, output=f"{PFIGS}/2-table.png"):
        """
        Monitor Change in the flux of the core with frequency

        cube: str
            Name of input i image cube
        region: str
            Region under investigation e.g pica core
            Pass the name of this CTRF region file

        """

        """
        see: https://casa.nrao.edu/docs/TaskRef/imstat-task.html
        # These are the display axes, the calculation of statistics occurs  
        # for each (hyper)plane along axes not listed in the axes parameter,  
        # in this case axis 2 (the frequency axis)  
        """
        from casatasks import imstat
        stats = imstat(cube, region=region, axes=[0,1])
        flux = stats["flux"]
        mean = stats["mean"]
        sigma = stats["sigma"]
        chans = np.arange(flux.size)

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(chans, flux, "ko", label="flux [Jy]")
        ax.plot(chans, snd.gaussian_filter(flux, sigma=3), "k--",
            label="flux fit")
        # ax.errorbar(chans, mean, yerr=sigma)
        ax.plot(chans, mean, "bo", label=r"$\mu$ [Jy/beam]")
        ax.plot(chans, snd.gaussian_filter(mean, sigma=3), "b--",
            label=r"$\mu$ fit ")
        ax.fill_between(chans, mean-sigma, mean+sigma, color="b",
            alpha=0.3, label=r"$\sigma$")

        ax.set_xlabel("Channel")
        ax.set_ylabel("Spectral fluxes*")
        plt.title("Flux change in Pictor A nucleus")
        plt.legend()
        print(f"Saving plot: {output}")
        
        plt.savefig(output, dpi=DPI)
        plt.close("all")

    
    @staticmethod
    def table3_lobe_flux_wfreq(elobe, wlobe,
        output=f"{PFIGS}/3-table-lobe-fluxes.png", smooth_sigma=10):
        """
        Monitor change of lobes' flux with frequncy
        (e|w)lobe:
            Cube of the eastern or western lobe. These can be generated using
            Fitstools in the following way:
            fitstool.py --prod i-image-cube.fits masks/west-lobe.fits -o west-lobe-cube.fits
            Note that we multiply the image x mask and not mask x image!

        see: https://casa.nrao.edu/docs/TaskRef/imstat-task.html
        # These are the display axes, the calculation of statistics occurs  
        # for each (hyper)plane along axes not listed in the axes parameter,  
        # in this case axis 2 (the frequency axis)  
        """
        from casatasks import imstat

        
        estats = imstat(elobe, axes=[0,1], stretch=True)
        wstats = imstat(wlobe, axes=[0,1], stretch=True)
        
        eflux = estats["flux"]
        wflux = wstats["flux"]

        chans = np.arange(eflux.size)

        fig, ax = plt.subplots(figsize=FIGSIZE, ncols=2, sharex=True)

        # Eastern lobe
        ax[0].plot(chans, eflux, "bo", label="flux [Jy]")
        ax[0].plot(
            chans, snd.gaussian_filter(eflux, sigma=smooth_sigma),
            "r--", label=fr"Gaussian fit {smooth_sigma}$\sigma$")
   
        ax[0].set_xlabel("Channel")
        ax[0].set_ylabel("Spectral fluxes*")
        ax[0].set_title(f"Eastern Lobe, Total Flux sum: {np.sum(eflux):.3f}")
        ax[0].minorticks_on()

        # Western lobe
        ax[1].plot(chans, wflux, "bo", label="flux [Jy]")
        ax[1].plot(
            chans, snd.gaussian_filter(wflux, sigma=smooth_sigma),
            "r--", label=fr"Gaussian fit {smooth_sigma}$\sigma$")

        ax[1].set_xlabel("Channel")
        ax[1].set_ylabel("Spectral fluxes*")
        ax[1].set_title(f"Western Lobe, Total Flux sum: {np.sum(wflux):.3f}")
        ax[1].minorticks_on()

        fig.suptitle("Flux change in Pictor A lobes")
        plt.legend()
        print(f"Saving plot: {output}")
        
        plt.savefig(output, dpi=DPI)
        plt.close("all")

    
    @staticmethod
    def figure_5b_spi_wintensity_contours(i_image, spi_image, mask, start=0.004,
        smooth_sigma=13, output=f"{PFIGS}/5b-spi-intensity-contours.png"):
        """
        i_image:
            The total intensity image
        spi_image:
            The SPI image
        mask:
            The image mask
        start:
            Where the contour levels should start
        """
        
        i_data = rfu.get_masked_data(i_image, mask)
        spi_data = rfu.get_masked_data(spi_image, mask)
        mask = i_data.mask

        # only show where i is greater than start
        spi_data = np.ma.masked_where(i_data<start, spi_data)

        ydim, xdim = np.where(mask == False)
        wiggle = 20
        xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
        ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle

        wcs = rfu.read_image_cube(spi_image)["wcs"]

        fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
        ax = set_image_projection(ax)

        cs = ax.imshow(spi_data,
                    origin="lower", cmap="coolwarm_r", vmin=-1.7,
                    vmax=-0.5, aspect="equal")
        plt.colorbar(cs, label="Spectral Index", pad=0)

        # the image data
        levels = contour_levels(start, i_data)
        ax.contour(
            snd.gaussian_filter(i_data, sigma=smooth_sigma),
            colors="k", linewidths=0.5, origin="lower", levels=levels)
    
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)

        plt.xlim(*xlims)
        plt.ylim(*ylims)
        fig.canvas.draw()
        fig.tight_layout()
        
        print(f"Saving plot: {output}")
        
        plt.savefig(output, dpi=DPI)
        plt.close("all")


    @staticmethod
    def figure_8_dop_magnetic_fields_contours(i_image, fp_image, angle_image,
        mask, start=0.004, smooth_sigma=1,
        output=f"{PFIGS}/8-dop-contours-mfs.png"):
        """
        LINES SHOW DEGREE OF POLARIZATION HERE!
        # Degree of polzn lines vs contours
        # we're not using cubes, we use the MFS images
        """
        if isinstance(i_image, str):
            i_data = rfu.get_masked_data(i_image, mask)
        else:
            i_data = i_image
        
        if isinstance(fp_image, str):
            fp_data = rfu.get_masked_data(fp_image, mask)
        else:
            fp_data = fp_image
        
        if isinstance(angle_image, str):
            angle_data = rfu.get_masked_data(angle_image, mask)
        else:
            angle_data = angle_image

        if isinstance(mask, str):
            mask_data = fits.getdata(mask).squeeze()
            mask_data = ~np.asarray(mask_data, dtype=bool).squeeze()
        else:
            mask_data = i_data.mask

    
        
        i_data = np.ma.masked_where(i_data<start, i_data)

        if "rad" in fits.getheader(angle_image)["BUNIT"].lower():
            angle_data = np.rad2deg(angle_data)
        
        # base the masks for the rest on I
        fp_data = np.ma.masked_where(i_data<start, fp_data)
        angle_data = np.ma.masked_where(i_data<start, angle_data) + ANGLE_OFFSET + ROT

        wcs = rfu.read_image_cube(mask)["wcs"]
        
        ydim, xdim = np.where(mask_data == False)
        wiggle = 20
        xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
        ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle


        fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
        ax = set_image_projection(ax)

        # the image data
        levels = contour_levels(start, i_data)
        ax.contour(
            snd.gaussian_filter(i_data, sigma=smooth_sigma),
            colors="k", linewidths=0.5, origin="lower", levels=levels)
       
        cs = ax.contourf(
            snd.gaussian_filter(i_data, sigma=smooth_sigma), 
            cmap="coolwarm", origin="lower", levels=levels,
            locator=ticker.LogLocator())

        plt.colorbar(cs, label="Total Intensity", pad=0)

        #################################
        skip = 7
        slicex = slice(None, fp_data.shape[0], skip)
        slicey = slice(None, fp_data.shape[-1], skip)
        col, row = np.mgrid[slicex, slicey]

        # get M vector by rotating E vector by 90
        angle_data = angle_data[slicex, slicey]
        fp_data = fp_data[slicex, slicey]

        # nornalize this
        
        scales = fp_data / np.ma.max(fp_data)
    
        # scale as amplitude
        u = scales * np.cos(angle_data)
        v = scales * np.sin(angle_data)
        # u = v = np.ones_like(angle_data) * scales

        qv = ax.quiver(
            row, col, u, v, angles=angle_data, pivot='tail', headlength=0,
            width=0.0012, scale=20, headwidth=1, color="black")
        
        #################################

    
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)

        plt.xlim(*xlims)
        plt.ylim(*ylims)
        fig.canvas.draw()
        fig.tight_layout()
        
        print(f"Saving plot: {output}")
        
        plt.savefig(output, dpi=DPI)
        plt.close("all")


    @staticmethod
    def figure_8b_dop_magnetic_fields_contours(i_image, fpol_image,
        angle_image, mask, start=0.004, smooth_sigma=1,
        output=f"{PFIGS}/8b-dop-contours-mfs.png"):
        """
        COLOUR MAP SHOWS FRACTIONA POLZN HERE!!

        linear poln, intensity contours, rm values vectors
        # Degree of polzn lines vs contours
        # we're not using cubes, we use the MFS images
        """

        if isinstance(i_image, str):
            i_data = rfu.get_masked_data(i_image, mask)
        else:
            i_data = i_image
        
        if isinstance(fpol_image, str):
            fpol_data = rfu.get_masked_data(fpol_image, mask)
        else:
            fpol_data = fpol_image
        
        if isinstance(angle_image, str):
            angle_data = rfu.get_masked_data(angle_image, mask)
        else:
            angle_data = angle_image

        if isinstance(mask, str):
            mask_data = fits.getdata(mask).squeeze()
            mask_data = ~np.asarray(mask_data, dtype=bool).squeeze()
        else:
            mask_data = i_data.mask

        if  not np.ma.is_masked(i_data):
            i_data = np.ma.masked_array(i_data, mask=mask_data, fill_value=np.nan)
        
        i_data = np.ma.masked_where(i_data<start, i_data)

        if "rad" in fits.getheader(angle_image)["BUNIT"].lower():
            angle_data = np.rad2deg(angle_data)
        
        # base the masks for the rest on I
        fpol_data = np.ma.masked_where(i_data<start, fpol_data)
        angle_data = np.ma.masked_where(i_data<start, angle_data) + ANGLE_OFFSET + ROT

        wcs = rfu.read_image_cube(mask)["wcs"]
        
        ydim, xdim = np.where(mask_data == False)
        wiggle = 20
        xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
        ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle


        fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
        ax = set_image_projection(ax)

        # the image data
        levels = contour_levels(start, i_data)

        # total intensity contours
        ax.contour(
            snd.gaussian_filter(i_data, sigma=smooth_sigma),
            colors="k", linewidths=0.5, origin="lower", levels=levels)
       
        # Fractional polarisation image
        cs = ax.imshow(fpol_data,
            cmap="coolwarm", origin="lower", vmin=0, vmax=0.7
            # norm=colors.LogNorm(vmin=fpol_data.min(), vmax=fpol_data.max())
            )
        

        plt.colorbar(cs, label="Fractional polarisation", pad=0)

        #################################
        skip = 7
        slicex = slice(None, fpol_data.shape[0], skip)
        slicey = slice(None, fpol_data.shape[-1], skip)
        col, row = np.mgrid[slicex, slicey]

        # get M vector by rotating E vector by 90
        angle_data = angle_data[slicex, slicey]

        # nornalize this, lenght of the vector
        scales = 0.03
    
        # scale as amplitude
        # u = scales * np.cos(angle_data)
        # v = scales * np.sin(angle_data)
        u = v = np.ones_like(angle_data) * scales
        

        qv = ax.quiver(
            row, col, u, v, angles=angle_data, pivot='tail', headlength=0,
            width=0.0012, scale=5, headwidth=1, color="black")
        
        #################################

    
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)

        plt.xlim(*xlims)
        plt.ylim(*ylims)
        fig.canvas.draw()
        fig.tight_layout()
        print(f"Saving plot: {output}")
        plt.savefig(output, dpi=DPI)
        plt.close("all")


    @staticmethod
    def figure_9a_fractional_poln(i_image, fp_image, mask, start=0.004,
        smooth_sigma=1, output=f"{PFIGS}/9a-dop-mfs.png", vmax=1):
        """
        # Degree of polzn lines without the contours
        This is determined from the FPOL map. 
        
        NOTE THAT: The FPOL map was generated
        by using the channel which had the maximum polarised intensity
        and dividing that with the corresponding I image.
    
        """
        plt.close("all")
        if isinstance(i_image, str):
            i_data = rfu.get_masked_data(i_image, mask)
        else:
            i_data = i_image
        
        if isinstance(fp_image, str):
            fp_data = rfu.get_masked_data(fp_image, mask)
        else:
            fp_data = fp_image
        
     
        if isinstance(mask, str):
            mask_data = fits.getdata(mask).squeeze()
            mask_data = ~np.asarray(mask_data, dtype=bool).squeeze()
        else:
            mask_data = i_data.mask

        
        i_data = np.ma.masked_where(i_data<start, i_data)
        # base the masks for the rest on I
        fp_data = np.ma.masked_where(i_data<start, fp_data)

        mask_data = i_data.mask

        ydim, xdim = np.where(mask_data == False)
        wiggle = 20
        xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
        ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle

        wcs = rfu.read_image_cube(mask)["wcs"]

        fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
        ax = set_image_projection(ax)

        # the image data
        i_levels = contour_levels(start, i_data)
        # total intensity contours
        ax.contour(
            snd.gaussian_filter(i_data, sigma=smooth_sigma),
            colors="k", linewidths=0.5, origin="lower", levels=i_levels)


        # if I use the mask here, we get a problematic image, creating new mask
        # where image is lower than `start` value
        levels = contour_levels(0.01, fp_data)
        # cs = ax.contourf(
        #     fp_data,
        #     cmap="coolwarm", origin="lower", levels=levels,#vmin=0, vmax=1
        #     )
        cs = ax.imshow(fp_data,
            origin="lower", cmap="coolwarm", aspect="equal", vmin=0, vmax=vmax,
            # norm=colors.LogNorm(vmin=fp_data.min(), vmax=1)
            )
        plt.colorbar(cs, label="Fractional Polarisation",
            pad=0)

        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)
        
        plt.xlim(*xlims)
        plt.ylim(*ylims)
        fig.canvas.draw()
        fig.tight_layout()
        print(f"Saving plot: {output}")
        
        plt.savefig(output, dpi=DPI)
        plt.close("all")


    @staticmethod
    def figure_10_linear_poln(i_image, lp_image, mask, start=0.004, smooth_sigma=1,
        output=f"{PFIGS}/10-lpol-mfs.png"):
        """
        NOTE THAT: The LPOL map was generated
        by using the channel which had the maximum polarised intensity per pixel.
        """

        if isinstance(i_image, str):
            i_data = rfu.get_masked_data(i_image, mask)
        else:
            i_data = i_image
        
        
        if isinstance(lp_image, str):
            lp_data = rfu.get_masked_data(lp_image, mask)
        else:
            lp_data = lp_image
        
        if isinstance(mask, str):
            mask_data = fits.getdata(mask).squeeze()
            mask_data = ~np.asarray(mask_data, dtype=bool).squeeze()
        else:
            mask_data = i_data.mask

        i_data = np.ma.masked_where(i_data<start, i_data)
        # base the masks for the rest on I
        lp_data = np.ma.masked_where(i_data<start, lp_data)

        ydim, xdim = np.where(mask_data == False)
        wiggle = 20
        xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
        ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle

        wcs = rfu.read_image_cube(mask)["wcs"]

        fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs},
                    )
        ax = set_image_projection(ax)
        
        cs = ax.imshow(lp_data,
            origin="lower", cmap="coolwarm", aspect="equal",
                # norm=colors.LogNorm(vmin=0.005, vmax=0.05)
                vmin=0.005, vmax=0.05
                )

         # the image data
        levels = contour_levels(start, i_data)
        # total intensity contours
        ax.contour(
            snd.gaussian_filter(i_data, sigma=smooth_sigma),
            colors="k", linewidths=0.5, origin="lower", levels=levels)


        plt.colorbar(cs, label="Polarised Brightness | P |", pad=0)
        
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)

        plt.xlim(*xlims)
        plt.ylim(*ylims)
        fig.canvas.draw()
        fig.tight_layout()
        
        print(f"Saving plot: {output}")
        plt.savefig(output, dpi=DPI)
        plt.close("all")


    @staticmethod
    def figure_rm_map(i_image, rm_map, mask, start=0.004, smooth_sigma=1,
        output=f"{PFIGS}/rm-map.png"):
        """
        Plot the RM-MAP
        Only where the total intensity is greater thab start
        """
        i_data = rfu.get_masked_data(i_image, mask)

        if isinstance(rm_map, str):
            rm_data = rfu.get_masked_data(rm_map, mask)
        else:
            rm_data = rm_map
        
        if isinstance(mask, str):
            mask_data = fits.getdata(mask).squeeze()
            mask_data = ~np.asarray(mask_data, dtype=bool).squeeze()
        else:
            mask_data = rm_data.mask

        i_data = np.ma.masked_where(i_data<start, i_data)
        # base the masks for the rest on I
        rm_data = np.ma.masked_where(i_data<start, rm_data)

        ydim, xdim = np.where(mask_data == False)
        wiggle = 20
        xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
        ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle

        wcs = rfu.read_image_cube(mask)["wcs"]

        fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
        ax = set_image_projection(ax)

        
        cs = ax.imshow(rm_data, origin="lower", cmap="coolwarm", 
            aspect="equal", vmin=-45, vmax=85)

        # the image data
        levels = contour_levels(start, i_data)
        ax.contour(i_data,
            colors="k", linewidths=0.5, origin="lower", levels=levels)

        plt.colorbar(cs, label="Rotation Measure", pad=0)
        
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)

        plt.xlim(*xlims)
        plt.ylim(*ylims)
        fig.canvas.draw()
        fig.tight_layout()
        
        print(f"Saving plot: {output}")
        plt.savefig(output, dpi=DPI)
        plt.close("all")


    @staticmethod
    def figure_rm_map_b_with_mf(i_image, rm_map, angle_image, mask,
        start=0.004, smooth_sigma=1, output=f"{PFIGS}/rm-map-wmf.png"):
        """
        Plot the RM-MAP
        Only where the total intensity is greater thab start
        """
        i_data = rfu.get_masked_data(i_image, mask)

        if isinstance(rm_map, str):
            rm_data = rfu.get_masked_data(rm_map, mask)
        else:
            rm_data = rm_map

        if isinstance(angle_image, str):
            angle_data = rfu.get_masked_data(angle_image, mask)
        else:
            angle_data = angle_image
        
        if isinstance(mask, str):
            mask_data = fits.getdata(mask).squeeze()
            mask_data = ~np.asarray(mask_data, dtype=bool).squeeze()
        else:
            mask_data = rm_data.mask

        i_data = np.ma.masked_where(i_data<start, i_data)
        # base the masks for the rest on I
        rm_data = np.ma.masked_where(i_data<start, rm_data)

        if "rad" in fits.getheader(angle_image)["BUNIT"].lower():
            angle_data = np.rad2deg(angle_data)


        angle_data = np.ma.masked_where(i_data<start, angle_data) + ANGLE_OFFSET + ROT

        ydim, xdim = np.where(mask_data == False)
        wiggle = 20
        xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
        ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle

        wcs = rfu.read_image_cube(mask)["wcs"]

        fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
        ax = set_image_projection(ax)

        
        cs = ax.imshow(rm_data, origin="lower", cmap="coolwarm", 
            aspect="equal", vmin=-45, vmax=85)


        # the image data
        levels = contour_levels(start, i_data)
        ax.contour(i_data,
            colors="k", linewidths=0.5, origin="lower", levels=levels)


        #################################
        skip = 7
        slicex = slice(None, angle_data.shape[0], skip)
        slicey = slice(None, angle_data.shape[-1], skip)
        col, row = np.mgrid[slicex, slicey]

        # get M vector by rotating E vector by 90
        angle_data = angle_data[slicex, slicey]

        # nornalize this, lenght of the vector
        scales = 0.03
    
        # scale as amplitude
        # u = scales * np.cos(angle_data)
        # v = scales * np.sin(angle_data)
        u = v = np.ones_like(angle_data) * scales

        qv = ax.quiver( 
            row, col, 
            u, v, 
            angles=angle_data,
            pivot='tail', headlength=0,
            width=0.0012, scale=5, headwidth=1, color="black")

 
        plt.colorbar(cs, label="Rotation Measure", pad=0)
        
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)

        plt.xlim(*xlims)
        plt.ylim(*ylims)
        fig.canvas.draw()
        fig.tight_layout()
        
        print(f"Saving plot: {output}")
        plt.savefig(output, dpi=DPI)
        plt.close("all")



    @staticmethod
    def figure_14_depolarisation(intensity, i_cube, q_cube, u_cube, mask,
        start=0.004, smooth_sigma=1, output=f"{PFIGS}/14-depolzn.png"):
        """
        Here we shall get the cube and use the first and last channels. We 
        therefore do not use the single generated FPOL map.

        intensity: 
            Single image with for the intensity contours. Usually the MFS
        (i|q|u)-image
            CUBES containing data for all the channels available. I will only use the
            first and last channels available int he cube
        """
        mask_data = fits.getdata(mask).squeeze()
        mask_data = ~np.asarray(mask_data, dtype=bool).squeeze()

        intensity = rfu.get_masked_data(intensity, mask)
        intensity = np.ma.masked_where(intensity<start, intensity)

        # select the first and last channels only
        i_data = fits.getdata(i_cube).squeeze()[[0,-1]]
        q_data = fits.getdata(q_cube).squeeze()[[0,-1]]
        u_data = fits.getdata(u_cube).squeeze()[[0,-1]]

        fpol = np.divide(np.abs(q_data + 1j*u_data), i_data)

        depoln = fpol[0]/fpol[-1]
        depoln = np.ma.masked_array(data=depoln, mask=intensity.mask,
            fill_value=np.nan)

        ydim, xdim = np.where(mask_data == False)
        wiggle = 20
        xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
        ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle

        wcs = rfu.read_image_cube(mask)["wcs"]

        levels = contour_levels(start, intensity)

        fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
        ax = set_image_projection(ax)

        ax.contour(
            snd.gaussian_filter(intensity, sigma=smooth_sigma),
            colors="k", linewidths=0.5, origin="lower", levels=levels)

        cs = ax.imshow(depoln,
                origin="lower", cmap="coolwarm", vmin=0, vmax=2, aspect="equal")
        plt.colorbar(cs,
            label="Depolarization ratio, [repolarization>1, depolarization <1]",
            pad=0)

        
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)

        plt.xlim(*xlims)
        plt.ylim(*ylims)
        fig.canvas.draw()
        fig.tight_layout()
        
        print(f"Saving plot: {output}")
        
        plt.savefig(output, dpi=DPI)
        plt.close("all")


    @staticmethod
    def figure_14_depolarisation_errmap(intensity, i_cube, q_cube, u_cube, mask,
        start=0.004, smooth_sigma=1, noise_file=None,
        output=f"{PFIGS}/14-depolzn-errors.png"):
        """
        Here we shall get the cube and use the first and last channels. We 
        therefore do not use the single generated FPOL map.

        intensity: 
            Single image with for the intensity contours. Usually the MFS
        (i|q|u)-image
            CUBES containing data for all the channels available. I will only use the
            first and last channels available int he cube
        """
        rg_data = dict(np.load(noise_file))
        errs = {key.lower(): rg_data[key][[0,-1], None, None] for key in 
            ["I_err", "Q_err", "U_err"]}
        mask_data = fits.getdata(mask).squeeze()
        mask_data = ~np.asarray(mask_data, dtype=bool).squeeze()

        intensity = rfu.get_masked_data(intensity, mask)
        intensity = np.ma.masked_where(intensity<start, intensity)

        # select the first and last channels only
        i_data = fits.getdata(i_cube).squeeze()[[0,-1]]
        q_data = fits.getdata(q_cube).squeeze()[[0,-1]]
        u_data = fits.getdata(u_cube).squeeze()[[0,-1]]


        # # # fpol = frac_polzn(i_data, q_data, u_data)
        # # # depoln = fpol[0]/fpol[-1]
        # # # depoln = np.ma.masked_array(data=depoln, mask=intensity.mask,
        # # #     fill_value=np.nan).compressed()
        # # # fpol_err = frac_polzn_error(i_data, q_data, u_data,
        # # #     errs["i_err"], errs["q_err"], errs["u_err"])
        # # # depoln_err = fpol_err[0]/fpol_err[-1]
        # # # depoln_err = np.ma.masked_array(data=depoln_err, mask=intensity.mask,
        # # #     fill_value=np.nan)
        # # # plt.errorbar(np.arange(depoln.size), 
        # # #     depoln, yerr=depoln_err.compressed(), 
        # # # fmt="o")
        # # # set_trace()
        # # # plt.savefig(output+"plt.png")


        
        fpol_err = frac_polzn_error(i_data, q_data, u_data,
            errs["i_err"], errs["q_err"], errs["u_err"])


        depoln_err = fpol_err[0]/fpol_err[-1]
        depoln_err = np.ma.masked_array(data=depoln_err, mask=intensity.mask,
            fill_value=np.nan)

        

        ydim, xdim = np.where(mask_data == False)
        wiggle = 20
        xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
        ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle

        wcs = rfu.read_image_cube(mask)["wcs"]

        levels = contour_levels(start, intensity)

        fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
        ax = set_image_projection(ax)

        ax.contour(
            snd.gaussian_filter(intensity, sigma=smooth_sigma),
            colors="k", linewidths=0.5, origin="lower", levels=levels)

        cs = ax.imshow(depoln_err,
                origin="lower", cmap="coolwarm", aspect="equal")
        plt.colorbar(cs,
            label="Depolarization error",
            pad=0)

        
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)

        plt.xlim(*xlims)
        plt.ylim(*ylims)
        fig.canvas.draw()
        fig.tight_layout()
        
        print(f"Saving plot: {output}")
        
        plt.savefig(output, dpi=DPI)
        plt.close("all")

    # @staticmethod
    def figure_12_13_rm_lobes_histogram(i_image, rm_image, emask, wmask,
        lmask, all_mask, start=0.004, smooth_sigma=0,
        output=f"{PFIGS}/12-rm-lobes.png"):
        """
        RM and histogram for lobes 

        i_image:
            Image to be used for the intensity contours. Usually MFS
        rm_image
            Image containing the required RMs
        emask
            Eastern lobe mask
        lmask
            Both lobes mask
        wmask
            Western lobe mask
        all_mask
            Pictor A mask
        start
            Where the contours should start 0.0552
        smooth_sigma
            Factor to smooth the contours
        """
        plt.close("all")
        i_data = rfu.get_masked_data(i_image, all_mask)

        i_data = np.ma.masked_where(i_data<start, i_data)

        rm_lobes = rfu.get_masked_data(rm_image, lmask)

        w_lobe = rfu.get_masked_data(rm_image, wmask)
        e_lobe = rfu.get_masked_data(rm_image, emask)

        wcs = rfu.read_image_cube(lmask)["wcs"]

        fig = plt.figure(figsize=FIGSIZE)
        
        image = plt.subplot2grid((3,3), (1,0), rowspan=2, colspan=2,
            projection=wcs)
        image = set_image_projection(image)
        
        # swich of these ticks
        image.tick_params(axis="x", top=False)
        image.tick_params(axis="y", right=False)
        # #image.axis("off")

        ca = image.imshow(
            rm_lobes, origin="lower", cmap="coolwarm", vmin=-45,
            vmax=85, aspect="equal")
        
        plt.colorbar(ca, location="right", shrink=0.90,
            label="RM", drawedges=False, pad=0)

        levels = contour_levels(start, i_data)
        image.contour(
            snd.gaussian_filter(i_data, sigma=smooth_sigma),
            colors="k", linewidths=0.5, origin="lower", levels=levels)
        

        # so that I can zoom into the image easily and automatically
        ydim, xdim = np.where(rm_lobes.mask == False)
        wiggle = 10
        bins = 50
        log = False
        plt.xlim(np.min(xdim)-wiggle, np.max(xdim)+wiggle)
        plt.ylim(np.min(ydim)-wiggle, np.max(ydim)+wiggle)

        west_hist = plt.subplot2grid((3,3), (1,2), rowspan=2, colspan=1)
        west_hist.hist(rm_lobes.compressed(), bins=bins, log=log,
            orientation="horizontal",fill=False, ls="--", lw=1,
            edgecolor="red",density=True, label="Source RMs",
            histtype="step")
        west_hist.hist(w_lobe.compressed(), bins=bins, log=log,
            orientation="horizontal",fill=False, ls="-", lw=1,
            edgecolor="blue",density=True, label="West lobe",
            histtype="step")

        west_hist.minorticks_on()
        west_hist.yaxis.tick_right()

        west_hist.set_title("Western Lobe (Right Hand) RM Distribution")
        west_hist.xaxis.set_visible(True)
        west_hist.xaxis.set_major_formatter(PercentFormatter(xmax=.1))
        west_hist.axes.set_xlabel("Data count")
        west_hist.legend()

        east_hist = plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=1)
        east_hist.hist(rm_lobes.compressed(), bins=bins, log=log,
            orientation="vertical",fill=False, ls="--", lw=1, edgecolor="red",
            density=True, label="Source RMs",
            histtype="step")
        east_hist.hist(e_lobe.compressed(), bins=bins, log=log,
            orientation="vertical", fill=False, ls="-", lw=1, edgecolor="blue",
            density=True, label="East lobe",
            histtype="step")
        east_hist.xaxis.tick_top()
        east_hist.minorticks_on()
        east_hist.set_title("Eastern Lobe (Left Hand) RM Distribution")
        east_hist.yaxis.set_visible(True)
        east_hist.yaxis.set_major_formatter(PercentFormatter(xmax=.1))
        east_hist.axes.set_ylabel("Data count")
        east_hist.legend()

        plt.subplots_adjust(wspace=.01, hspace=0)
        
        east_hist.set_xlim(-200, 200)
        west_hist.set_ylim(-200, 200)
        plt.tight_layout()
        fig.savefig(output)
        print(f"Output is at: {output}")


    def figure_12_13_rm_lobes_histogram_rick(i_image, rm_image, emask, wmask,
        lmask, all_mask, rick_rm, rick_east, rick_west, 
        start=0.004, smooth_sigma=0,
        output=f"{PFIGS}/12-rm-lobes-with-ricks.png"):
        """
        RM and histogram for lobes 

        i_image:
            Image to be used for the intensity contours. Usually MFS
        rm_image
            Image containing the required RMs
        emask
            Eastern lobe mask
        lmask
            Both lobes mask
        wmask
            Western lobe mask
        all_mask
            Pictor A mask
        start
            Where the contours should start 0.0552
        smooth_sigma
            Factor to smooth the contours
        """
        plt.close("all")
        i_data = rfu.get_masked_data(i_image, all_mask)

        i_data = np.ma.masked_where(i_data<start, i_data)

        rm_lobes = rfu.get_masked_data(rm_image, lmask)

        w_lobe = rfu.get_masked_data(rm_image, wmask)
        e_lobe = rfu.get_masked_data(rm_image, emask)

        # ricks data
        rw_lobe = rfu.get_masked_data(rick_rm, rick_west)
        re_lobe = rfu.get_masked_data(rick_rm, rick_east)

        wcs = rfu.read_image_cube(lmask)["wcs"]

        fig = plt.figure(figsize=FIGSIZE)
        
        image = plt.subplot2grid((3,3), (1,0), rowspan=2, colspan=2,
            projection=wcs)
        image = set_image_projection(image)
        
        # swich of these ticks
        image.tick_params(axis="x", top=False)
        image.tick_params(axis="y", right=False)
        # #image.axis("off")

        ca = image.imshow(
            rm_lobes, origin="lower", cmap="coolwarm", vmin=-45,
            vmax=85, aspect="equal")
        
        plt.colorbar(ca, location="right", shrink=0.90,
            label="RM", drawedges=False, pad=0)

        levels = contour_levels(start, i_data)
        image.contour(
            snd.gaussian_filter(i_data, sigma=smooth_sigma),
            colors="k", linewidths=0.5, origin="lower", levels=levels)
        

        # so that I can zoom into the image easily and automatically
        ydim, xdim = np.where(rm_lobes.mask == False)
        wiggle = 10
        bins = 50
        log = False
        plt.xlim(np.min(xdim)-wiggle, np.max(xdim)+wiggle)
        plt.ylim(np.min(ydim)-wiggle, np.max(ydim)+wiggle)

        west_hist = plt.subplot2grid((3,3), (1,2), rowspan=2, colspan=1)
        west_hist.hist(rm_lobes.compressed(), bins=bins, log=log,
            orientation="horizontal",fill=False, ls="--", lw=1,
            edgecolor="black",density=True, label="All RMs",
            histtype="step")
        west_hist.hist(w_lobe.compressed(), bins=bins, log=log,
            orientation="horizontal",fill=False, ls="-", lw=1,
            edgecolor="blue",density=True, label="West lobe",
            histtype="step")
        west_hist.hist(rw_lobe.compressed(), bins=bins, log=log,
            orientation="horizontal",fill=False, ls="-", lw=1,
            edgecolor="red",density=True, label="P97 West lobe",
            histtype="step")

        west_hist.minorticks_on()
        west_hist.yaxis.tick_right()

        west_hist.set_title("Western Lobe (Right Hand) RM Distribution")
        west_hist.xaxis.set_visible(True)
        west_hist.xaxis.set_major_formatter(PercentFormatter(xmax=.1))
        west_hist.axes.set_xlabel("Data count")
        west_hist.legend()

        east_hist = plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=1)
        east_hist.hist(rm_lobes.compressed(), bins=bins, log=log,
            orientation="vertical",fill=False, ls="--", lw=1, edgecolor="black",
            density=True, label="All RMs",
            histtype="step")
        east_hist.hist(e_lobe.compressed(), bins=bins, log=log,
            orientation="vertical", fill=False, ls="-", lw=1, edgecolor="blue",
            density=True, label="East lobe",
            histtype="step")
        east_hist.hist(re_lobe.compressed(), bins=bins, log=log,
            orientation="vertical", fill=False, ls="-", lw=1, edgecolor="red",
            density=True, label="P97 East lobe",
            histtype="step")
        east_hist.xaxis.tick_top()
        east_hist.minorticks_on()
        east_hist.set_title("Eastern Lobe (Left Hand) RM Distribution")
        east_hist.yaxis.set_visible(True)
        east_hist.yaxis.set_major_formatter(PercentFormatter(xmax=.1))
        east_hist.axes.set_ylabel("Data count")
        east_hist.legend()

        plt.subplots_adjust(wspace=.01, hspace=0)
        
        east_hist.set_xlim(-200, 200)
        west_hist.set_ylim(-200, 200)
        plt.tight_layout()
        fig.savefig(output)
        print(f"Output is at: {output}")




def fixer():
    """Text it fixit self contained testing thingy"""
    cubes = sorted(glob(
            os.path.join(".", os.environ["conv_cubes"], "*-image-cube.fits")
            ))[:3]

    imgs = sorted(glob("./*-mfs.fits"))
    mask_dir = os.environ["mask_dir"]
    products = os.environ["prods"]

    mask = f"{mask_dir}/true_mask.fits"
    jet_mask = f"{mask_dir}/jet.fits"
    prefix = "initial"

    rm_map = os.path.join(products, f"{prefix}-RM-depth-at-peak-rm.fits")
    fp_map = os.path.join(products, f"{prefix}-FPOL-at-max-lpol.fits")
    lp_map = os.path.join(products, f"{prefix}-max-LPOL.fits")
    pangle_map = os.path.join(products, f"{prefix}-PA-pangle-at-peak-rm.fits")
    # pangle_map = "rm-tools-test/no-f/pangle-FDF.fits"
    

    idata = rfu.read_image_cube(cubes[0])["data"]
    qdata = rfu.read_image_cube(cubes[1])["data"]
    udata = rfu.read_image_cube(cubes[2])["data"]

    chandra = [
        f"{spectest}/from_martin/chandra.fits",
        f"{spectest}/from_martin/chandra-jet.fits"
        ]

    for o_dir in ["fig3", "fig4", "fig8", "fig9", "fig10"]:
        if not os.path.isdir(os.path.join(PFIGS, o_dir)):
            os.makedirs(os.path.join(PFIGS, o_dir))

    
    # PaperPlots.figure_12_13_rm_lobes_histogram_rick(
    #     imgs[0], f"{products}/initial-RM-depth-at-peak-rm.fits",
    #     f"{mask_dir}/east-lobe.fits", f"{mask_dir}/west-lobe.fits", 
    #     f"{mask_dir}/lobes.fits", #f"{mask_dir}/no-core.fits"
    #     f"{mask_dir}/true_mask.fits",
    #     f"{spectest}/from_rick/4k-proj/band-l-and-c-LCPIC-10.RM10.2.FITS-projected.fits",
    #     f"{mask_dir}/rick-east-rm2.fits",
    #     f"{mask_dir}/rick-west-rm2.fits",)

    PaperPlots.figure_14_depolarisation_errmap(imgs[0], *cubes, mask,
        noise_file=f"{products}/scrap-outputs-s3/los-data/reg_1.npz")




def run_paper_mill():
    """
    TO BE RUN IN THE IPYTHON TERMINAL
    add

        %load_ext autoreload
        %autoreload 2
        import test_paper_plots as tpp

    What are used in this function?

    1. selected image cubes from
    intermediates/conv-cubes/i-image-cube.fits',
    
    2. The multifreq images from where this script is running:
        ./i-mfs.fits
    
    3. Path of the masks directory
    4. Lobe's cubes east and west. These are to be found in the current dir pre-generated
        ./(east|west)-lobe-cube.fits
    5. SPI image from
        ./products/spi-fitting/
    7. The generated maps from
        ./products/intial-blabla.fits
    8. CTRF fregion for the core of the galaxy found in masks
        ./masks/importat_regions/hotspots/core-ctrf

    """
    print("----------------------------")
    print("Running paper mill")
    print("----------------------------")
    cubes = sorted(glob(
            os.path.join(".", os.environ["conv_cubes"], "*-image-cube.fits")
            ))[:3]

    imgs = sorted(glob("./*-mfs.fits"))
    mask_dir = os.environ["mask_dir"]
    products = os.environ["prods"]

    mask = f"{mask_dir}/true_mask.fits"
    jet_mask = f"{mask_dir}/jet.fits"
    prefix = "initial"

    rm_map = os.path.join(products, f"{prefix}-RM-depth-at-peak-rm.fits")
    fp_map = os.path.join(products, f"{prefix}-FPOL-at-max-lpol.fits")
    lp_map = os.path.join(products, f"{prefix}-max-LPOL.fits")
    pangle_map = os.path.join(products, f"{prefix}-PA-pangle-at-peak-rm.fits")

    idata = rfu.read_image_cube(cubes[0])["data"]
    qdata = rfu.read_image_cube(cubes[1])["data"]
    udata = rfu.read_image_cube(cubes[2])["data"]

    chandra = [
        f"{spectest}/from_martin/chandra.fits",
        f"{spectest}/from_martin/chandra-jet.fits"
        ]

    for o_dir in ["fig3", "fig4", #"fig8", "fig9", "fig10"
        ]:
        if not os.path.isdir(os.path.join(PFIGS,o_dir)):
            os.makedirs(os.path.join(PFIGS,o_dir))

    print("starting")

    PaperPlots.table2_core_flux_wfreq(cubes[0],
        region=f"{mask_dir}/important_regions/hotspots/core-ctrf")
    PaperPlots.table3_lobe_flux_wfreq(elobe=f"{products}/east-lobe-cube.fits",
        wlobe=f"{products}/west-lobe-cube.fits")

    PaperPlots.figure_8_dop_magnetic_fields_contours(imgs[0], fp_map,
        pangle_map, mask)
    PaperPlots.figure_8b_dop_magnetic_fields_contours(imgs[0], fp_map,
        pangle_map, mask)
    PaperPlots.figure_9a_fractional_poln(imgs[0], fp_map, mask)
    PaperPlots.figure_10_linear_poln(imgs[0], lp_map, mask)
    PaperPlots.figure_14_depolarisation(imgs[0], *cubes, mask)
    PaperPlots.figure_14_depolarisation_errmap(imgs[0], *cubes, mask,
        noise_file=f"{products}/scrap-outputs-s3/los-data/reg_1.npz")


    PaperPlots.figure_3_total_and_jets(imgs[0], mask, jet_mask=jet_mask,
        output=f"{PFIGS}/fig3/total-intensity-max2", 
        vmin=0, vmax=2, scale="linear", cmap="magma")

    PaperPlots.figure_3_total_and_jets(imgs[0], mask, jet_mask=jet_mask,
        output=f"{PFIGS}/fig3/total-intensity-max-1e-1", 
        vmin=1e-2, vmax=1e-1, scale="linear", cmap="magma")

    PaperPlots.figure_3_total_and_jets(imgs[0], mask, jet_mask=jet_mask,
        output=f"{PFIGS}/fig3/total-intensity-power-scale", vmin=1e-2,
        vmax=0.55e-1,
        scale="power", cmap="magma", kwargs={"gamma": 3.8})

    # with chandra
    PaperPlots.figure_3b_chandra(imgs[0], mask, chandra=chandra[0],
        chandra_jet=chandra[1], jet_mask=jet_mask,
        output=f"{PFIGS}/fig3/chandra-total-intensity-max2", 
        vmin=0, vmax=2, scale="linear", cmap="magma")
    PaperPlots.figure_3b_chandra(imgs[0], mask, chandra=chandra[0],
        chandra_jet=chandra[1], jet_mask=jet_mask,
        output=f"{PFIGS}/fig3/chandra-total-intensity-max-1e-1", 
        vmin=1e-2, vmax=1e-1, scale="linear", cmap="magma")
    PaperPlots.figure_3b_chandra(imgs[0], mask, chandra=chandra[0],
        chandra_jet=chandra[1], jet_mask=jet_mask,
        output=f"{PFIGS}/fig3/chandra-total-intensity-power-scale",
        vmin=1e-2, vmax=0.55e-1,
        scale="power", cmap="magma", kwargs={"gamma": 3.8})
    

    images = {
        # "from_rick/pic-l-all-4k.fits": "fig4/4b-rick-intensity-contours-mpl.png", 
        "i-mfs.fits": f"{PFIGS}/fig4/4b-intensity-contours-mpl.png",
        }

    for im, out in images.items():
        PaperPlots.figure_4b_intensity_contours(im, mask, output=out)

    PaperPlots.figure_5b_spi_wintensity_contours(
        imgs[0],
        f"{products}/spi-fitting/spi-map.alpha.fits", mask, 
        output=f"{PFIGS}/5b-spi-with-contours-mpl.png")

    
    # Lobe stuff
    PaperPlots.figure_12_13_rm_lobes_histogram(
        imgs[0], f"{products}/initial-RM-depth-at-peak-rm.fits",
        f"{mask_dir}/east-lobe.fits", f"{mask_dir}/west-lobe.fits", 
        f"{mask_dir}/lobes.fits", #f"{mask_dir}/no-core.fits"
        f"{mask_dir}/true_mask.fits")

    PaperPlots.figure_12_13_rm_lobes_histogram_rick(
        imgs[0], f"{products}/initial-RM-depth-at-peak-rm.fits",
        f"{mask_dir}/east-lobe.fits", f"{mask_dir}/west-lobe.fits", 
        f"{mask_dir}/lobes.fits", #f"{mask_dir}/no-core.fits"
        f"{mask_dir}/true_mask.fits",
        f"{spectest}/from_rick/4k-proj/band-l-and-c-LCPIC-10.RM10.2.FITS-projected.fits",
        f"{mask_dir}/rick-east-rm2.fits",
        f"{mask_dir}/rick-west-rm2.fits",)


    PaperPlots.figure_rm_map(imgs[0], rm_map, mask, start=0.004, smooth_sigma=1)
    
    PaperPlots.figure_rm_map_b_with_mf(imgs[0], rm_map, pangle_map, mask)


    print("----------------------------")
    print("Paper mill stopped")
    print("----------------------------")


def contour_levels(start, data):
    # ratio between the levels is root(2)
    print("Generating contour levels")
    levels = [start * np.sqrt(2)**_ for _ in range(30)]
    levels = np.ma.masked_greater(levels, np.nanmax(data)).compressed()
    return levels


'''
def make_masks_from_ricks_data():

    fnames = [
    "band-l-and-c-LCPIC-10-NBL.RM.1.FITS",
    "band-l-and-c-LCPIC-10.RM10.2.FITS", #ricks_data.mask.fits
    "band-l-LPIC-ACBD-12.RM5.1.FITS",
    "band-l-and-c-LCPIC-10.RM10.1.FITS",
    "band-l-LPIC-ACBD-12.RM2.1.FITS",
    ]

    i_image = "I-MFS-image.fits"

    for fname in fnames:
        rfu.make_mask(fname, xy_dims=(572,572))


    fig, ax = plt.subplots(figsize=FIGSIZE, sharex=True, sharey=True, ncols=2, gridspec_kw={"wspace": 0 })
    ax[0].imshow(mask, origin="lower")
    ax[0].vlines(280, 0, 571)
    ax[0].hlines(280, 0, 571)
    ax[1].imshow(pica, origin="lower")
    ax[1].vlines(280, 0, 571)
    ax[1].hlines(280, 0, 571)
    # plt.show()

'''

def parser():
    ps = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        description="Script to genernetae some plots for my paper")
    # ps.add_argument("-rim", "--ref-i-image", dest="ref_image", 
    #     help="Reference I image for the contour plots")
    # ps.add_argument("--cube-name", dest="cube_names", nargs="+",
    #     help="Input I q AND U same resolution cubes")
    # ps.add_argument("--input-maps", dest="prefix", 
    #     required=True,
    #     help="Input prefix for the fpol rm and whatever maps")
    # ps.add_argument("--mask-name", dest="mask_name",
    #     help="Name of the full field mask to use"
    #     )
    # ps.add_argument("-elm", "--e-lobe-mask", dest="elobe_mask",
    #     help="Mask for the eastern lobe"
    #     )
    # ps.add_argument("-wlm", "--w-lobe-mask", dest="wlobe_mask",
    #     help="Mask for the western lobe"
    #     )
    ps.add_argument("-o", "-output-dir", dest="output_dir", default=None,
        help="Where to dump the outputs"
        )
    return ps

######################################################################################
# Main
######################################################################################

if __name__ == "__main__":
    opts = parser().parse_args()
    if opts.output_dir is not None:
        PFIGS = opts.output_dir

    run_paper_mill()
    # fixer()

    """       
    Running this script with
    
    python qu_pol/test_paper_plots.py --input-maps $prods/initial -rim i-mfs.fits --cube-name $conv_cubes/*-conv-image-cube.fits --mask-name $mask_dir/true_mask.fits -elm $mask_dir/east-lobe.fits -wlm $mask_dir/west-lobe.fits -o $prods/some-plots

    """