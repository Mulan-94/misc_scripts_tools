import numpy as np
import argparse
import matplotlib.pyplot as plt
import warnings

from astropy.io import fits
from astropy.wcs import WCS
from glob import glob
from matplotlib import ticker
from casatasks import imstat
from glob import glob
import scipy.ndimage as snd

from matplotlib.colors import LogNorm

import sys
import os

sys.path.append(f"{os.environ['HOME']}/git_repos/misc_scripts_n_tools/fits_related/")

import random_fits_utils as rfu
from ipdb import set_trace


warnings.filterwarnings("ignore", module="astropy")

FIGSIZE = (16,9)
EXT = ".svg"

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


######################################################################################
## Plot polarised intensity for each channel

def plot_polarised_intensity(data=None, im_name=None, mask=None, oup=None, ref_image=None):
    """
    Input
    -----
    data:
        Numpy array containing image data. If this is specified, the rest of 
        the args are ignored.
    im_name
        Masked polarisation intensity im_name
    mask_name
        Name of mask to be applied
    """

    # if strings read the image, otherwise assume data is in numpy array
    if isinstance(data, str):
        data = rfu.read_image_cube(data, mask=False)["data"]
    if mask is not None and isinstance(mask, str):
        mask = rfu.read_image_cube(mask)["data"]
    
    chans = data.shape[0]
    mid_chan = int(np.median(np.arange(chans)))
    data = np.abs(data)

    ydim, xdim = np.where(mask == False)
    wiggle = 10
    xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
    ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle

    wcs = rfu.read_image_cube(ref_image)["wcs"]

    """
    fig, ax = plt.subplots(
        figsize=FIGSIZE, ncols=6, nrows=int(np.ceil(chans/6)), sharex=True,
        sharey=True, gridspec_kw={"wspace":0 , "hspace":0}, subplot_kw={'projection': wcs})
    ax = ax.flatten()

    for chan in range(chans):
        lpol = np.ma.masked_array(data=data[chan], mask=mask)
        image = ax[chan].imshow(lpol, origin="lower", cmap="coolwarm", vmin=0, vmax=0.3)
        plt.xlim(*xlims)
        plt.ylim(*ylims)

    fig.subplots_adjust(right=0.8)
    # left, bottom, width, height
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(image, cax=cbar_ax)
    fig.tight_layout()
    """

    fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
    lpol = np.ma.masked_array(data=data[mid_chan], mask=mask)
    image = ax.imshow(lpol, origin="lower", cmap="coolwarm", vmin=0, vmax=0.1)
    
    ax = set_image_projection(ax)
    ax.tick_params(axis="x", top=False)
    ax.tick_params(axis="y", right=False)
    
    ax.set_title(f"Polarised Intensity of Middle Channel: "+f"{mid_chan}".zfill(4))
    plt.xlim(*xlims)
    plt.ylim(*ylims)
    fig.colorbar(image, shrink=0.85)
    fig.tight_layout()
    oname = oup + f"-lin_pol{EXT}"
    fig.savefig(oname)
    print(f"Output is at: {oname}")


######################################################################################
## Plot fractional polzn for each channel

def plot_fractional_polzn(data, mask, oup=None, ref_image=None):
    """
    Input
    -----
    data:
        Numpy array containing image data.  Otherwise, it is the name of
        the iamge where the data will be gotten from 
    mask_name  mask data
        Name of mask to be applied. or amsk. This is amust fro mny own sake
    """
   
    print("Starting frac pol plot")
   
    # if strings read the image, otherwise assume data is in numpy array
    if isinstance(data, str):
        data = rfu.read_image_cube(data, mask=False)["data"]
    if mask is not None and isinstance(mask, str):
        mask = rfu.read_image_cube(mask)["data"]
    
    chans = data.shape[0]
    mid_chan = int(np.median(np.arange(chans)))
    data = np.abs(data)

    ydim, xdim = np.where(mask == False)
    wiggle = 10
    xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
    ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle

    wcs = rfu.read_image_cube(ref_image)["wcs"]
    
    """
    fig, ax = plt.subplots(
        figsize=FIGSIZE, ncols=6, nrows=int(np.ceil(chans/6)), sharex=True,
        sharey=True, gridspec_kw={"wspace":0 , "hspace":0})
    ax = ax.flatten()

    for chan in range(chans):
        fpol = np.ma.masked_array(data=data[chan], mask=mask)
        image = ax[chan].imshow(fpol, origin="lower", cmap="coolwarm", vmin=0, vmax=0.3)
        plt.xlim(*xlims)
        plt.ylim(*ylims)

    fig.subplots_adjust(right=0.8)
    # left, bottom, width, height
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(image, cax=cbar_ax)
    fig.tight_layout()
    """

    fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
    fpol = np.ma.masked_array(data=data[mid_chan], mask=mask)
    image = ax.imshow(fpol, origin="lower", cmap="coolwarm", vmin=0, vmax=0.3)
    
    ax = set_image_projection(ax)
    ax.tick_params(axis="x", top=False)
    ax.tick_params(axis="y", right=False)
    
    ax.set_title(f"Fractional polarisation of Middle Channel: "+f"{mid_chan}".zfill(4))
    plt.xlim(*xlims)
    plt.ylim(*ylims)
    fig.colorbar(image, shrink=0.85)
    fig.tight_layout()
    oname = oup + f"-fpol{EXT}"
    fig.savefig(oname)
    print(f"Output is at: {oname}")

    return
    


######################################################################################
## contours plotting and vectors
######################################################################################

# ref https://stackoverflow.com/questions/40939821/how-to-plot-a-vector-field-over-a-contour-plot-in-matplotlib


class PaperPlots:

    @staticmethod
    def figure_4b(image, mask, start=0.004, smooth_sigma=13,
        output="4b-intensity-contours.png"):

        data = rfu.get_masked_data(image, mask)
        # # replace nans with zeroes. Remember in mask images, the NOT MAsked area is set to 1
        data[np.where(np.isnan(data))] = 0
        mask = data.mask

        ydim, xdim = np.where(mask == False)
        wiggle = 70
        xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
        ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle

        wcs = rfu.read_image_cube(image)["wcs"]

        fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
        ax = set_image_projection(ax)

        # the image data
        levels = contour_levels(start, data)
        ax.contour(
            snd.gaussian_filter(data, sigma=smooth_sigma),
            colors="g", linewidths=0.5, origin="lower", levels=levels)

        ax.imshow(data, origin="lower", cmap="magma", vmin=0, vmax=.127, aspect="equal")
    
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)

        plt.xlim(*xlims)
        plt.ylim(*ylims)
        
        print(f"Saving plot: {output}")
        
        plt.savefig(output, dpi=200)
        plt.close("all")

    
    @staticmethod
    def table2(cube, region, output="2-table.png"):
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
        stats = imstat(cube, region=region, axes=[0,1])
        flux = stats["flux"]
        mean = stats["mean"]
        sigma = stats["sigma"]
        chans = np.arange(flux.size)

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(chans, flux, "ko", label="flux [Jy]")
        ax.plot(chans, snd.gaussian_filter(flux, sigma=3), "k--", label="flux fit")
        # ax.errorbar(chans, mean, yerr=sigma)
        ax.plot(chans, mean, "bo", label=r"$\mu$ [Jy/beam]")
        ax.plot(chans, snd.gaussian_filter(mean, sigma=3), "b--", label=r"$\mu$ fit ")
        ax.fill_between(chans, mean-sigma, mean+sigma, color="b", alpha=0.3, label=r"$\sigma$")

        ax.set_xlabel("Channel")
        ax.set_ylabel("Spectral fluxes*")
        plt.title("Flux change in Pictor A nucleus")
        plt.legend()
        print(f"Saving plot: {output}")
        
        plt.savefig(output, dpi=200)
        plt.close("all")

    
    @staticmethod
    def table3(elobe, wlobe, output="3-table-lobe-fluxes.png", smooth_sigma=10):
        plt.style.use("seaborn")
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
        
        plt.savefig(output, dpi=200)
        plt.close("all")

    
    @staticmethod
    def figure_5b(intensity, image, mask, start=0.004, smooth_sigma=13,
        output="5b-spi-intensity-contours.png"):
        """
        image:
            The SPI image
        mask:
            The image mask
        start:
            Where the contour levels should start
        """
        
        intensity = rfu.get_masked_data(intensity, mask)
        data = rfu.get_masked_data(image, mask)
        mask = data.mask

        ydim, xdim = np.where(mask == False)
        wiggle = 20
        xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
        ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle

        wcs = rfu.read_image_cube(image)["wcs"]

        fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
        ax = set_image_projection(ax)


        cs = ax.imshow(data, origin="lower", cmap="coolwarm_r", vmin=-1.7, vmax=-0.5, aspect="equal")
        plt.colorbar(cs, label="Spectral Index")

        # the image data
        levels = contour_levels(start, intensity)
        ax.contour(
            snd.gaussian_filter(intensity, sigma=smooth_sigma),
            colors="k", linewidths=0.5, origin="lower", levels=levels)
    
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)

        plt.xlim(*xlims)
        plt.ylim(*ylims)
        
        print(f"Saving plot: {output}")
        
        plt.savefig(output, dpi=200)
        plt.close("all")


    @staticmethod
    def figure_8(i_image, q_image, u_image, mask, start=0.004, smooth_sigma=1,
        output="8-dop-contours-mfs.png"):
        """
        # Degree of polzn lines vs contours
        # we're not using cubes, we use the MFS images
        """
        if isinstance(i_image, str):
            i_data = rfu.get_masked_data(i_image, mask)
        else:
            i_data = i_image
        
        if isinstance(q_image, str):
            q_data = rfu.get_masked_data(q_image, mask)
        else:
            q_data = q_image
        
        if isinstance(u_image, str):
            u_data = rfu.get_masked_data(u_image, mask)
        else:
            u_data = u_image

        if isinstance(mask, str):
            mask_data = fits.getdata(mask).squeeze()
            mask_data = ~np.asarray(mask_data, dtype=bool).squeeze()
        else:
            mask_data = i_data.mask

        if  not np.ma.is_masked(i_data):
            i_data = np.ma.masked_array(i_data, mask=mask_data)
        if  not np.ma.is_masked(q_data):
            q_data = np.ma.masked_array(q_data, mask=mask_data)
        if  not np.ma.is_masked(u_data):
            u_data = np.ma.masked_array(u_data, mask=mask_data)
            

        lpol = np.ma.abs(q_data + 1j*u_data)
        fpol = np.ma.divide(lpol, i_data)
        p_angle = 0.5 * np.ma.arctan2(u_data, q_data)

        # # replace nans with zeroes. Remember in mask images, the NOT MAsked area is set to 1
        # data[np.where(np.isnan(data))] = 0

        wcs = rfu.read_image_cube(mask)["wcs"]
        
        ydim, xdim = np.where(mask_data == False)
        wiggle = 70
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

        plt.colorbar(cs, label="Total Intensity")

        #################################
        skip = 5
        slicex = slice(None, fpol.shape[0], skip)
        slicey = slice(None, fpol.shape[-1], skip)
        col, row = np.mgrid[slicex, slicey]

        # get M vector by rotating E vector by 90
        p_angle = p_angle[slicex, slicey] #+ (np.pi/2)
        fpol = fpol[slicex, slicey]

        # nornalize this
        
        fpol = np.ma.masked_greater(fpol, 1)
        fpol = np.ma.masked_less(fpol, 0)
        scales = fpol / np.ma.max(fpol)
    
        # scale as amplitude
        u = scales * np.cos(p_angle)
        v = scales * np.sin(p_angle)

        qv = ax.quiver(
            row, col, u, v, angles="xy", pivot='tail', headlength=4,
            width=0.0008, scale=5, headwidth=0)
        
        #################################

    
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)

        plt.xlim(*xlims)
        plt.ylim(*ylims)
        
        print(f"Saving plot: {output}")
        
        plt.savefig(output, dpi=200)
        plt.close("all")


    @staticmethod
    def figure_9a(i_image, q_image, u_image, mask, start=0.004, smooth_sigma=1,
        output="9a-dop-mfs.png"):
        """
        # Degree of polzn lines without the contours
    
        """
        if isinstance(i_image, str):
            i_data = rfu.get_masked_data(i_image, mask)
        else:
            i_data = i_image
        
        if isinstance(q_image, str):
            q_data = rfu.get_masked_data(q_image, mask)
        else:
            q_data = q_image
        
        if isinstance(u_image, str):
            u_data = rfu.get_masked_data(u_image, mask)
        else:
            u_data = u_image

        if isinstance(mask, str):
            mask_data = fits.getdata(mask).squeeze()
            mask_data = ~np.asarray(mask_data, dtype=bool).squeeze()
        else:
            mask_data = i_data.mask

        if  not np.ma.is_masked(i_data):
            i_data = np.ma.masked_array(i_data, mask=mask_data)
        if  not np.ma.is_masked(q_data):
            q_data = np.ma.masked_array(q_data, mask=mask_data)
        if  not np.ma.is_masked(u_data):
            u_data = np.ma.masked_array(u_data, mask=mask_data)
            
 
        lpol = np.abs(q_data + 1j*u_data)
        fpol = np.divide(lpol, i_data)
        # # replace nans with zeroes. Remember in mask images, the NOT MAsked area is set to 1
        # data[np.where(np.isnan(data))] = 0
        mask_data = i_data.mask

        ydim, xdim = np.where(mask_data == False)
        wiggle = 70
        xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
        ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle

        wcs = rfu.read_image_cube(mask)["wcs"]

        fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
        ax = set_image_projection(ax)


        cs = ax.imshow(fpol, origin="lower", cmap="coolwarm", vmin=0, vmax=.7, aspect="equal")
        plt.colorbar(cs, label="Degree of polarisation [Fractional Polarisation]")

        
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)

        plt.xlim(*xlims)
        plt.ylim(*ylims)
        
        print(f"Saving plot: {output}")
        
        plt.savefig(output, dpi=200)
        plt.close("all")


    @staticmethod
    def figure_10(q_image, u_image, mask, start=0.004, smooth_sigma=1,
        output="10-lpol-mfs.png"):
        """
        # Degree of polzn lines vs contours
        # we're not using cubes, we use the MFS images
        """
        
        if isinstance(q_image, str):
            q_data = rfu.get_masked_data(q_image, mask)
        else:
            q_data = q_image
        
        if isinstance(u_image, str):
            u_data = rfu.get_masked_data(u_image, mask)
        else:
            u_data = u_image

        if isinstance(mask, str):
            mask_data = fits.getdata(mask).squeeze()
            mask_data = ~np.asarray(mask_data, dtype=bool).squeeze()
        else:
            mask_data = q_data.mask

        if  not np.ma.is_masked(q_data):
            q_data = np.ma.masked_array(q_data, mask=mask_data)
        if  not np.ma.is_masked(u_data):
            u_data = np.ma.masked_array(u_data, mask=mask_data)

        lpol = np.abs(q_data + 1j*u_data)
        
        # # replace nans with zeroes. Remember in mask images, the NOT MAsked area is set to 1
        # data[np.where(np.isnan(data))] = 0
        mask_data = q_data.mask

        ydim, xdim = np.where(mask_data == False)
        wiggle = 70
        xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
        ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle

        wcs = rfu.read_image_cube(mask)["wcs"]

        fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
        ax = set_image_projection(ax)

        
        cs = ax.imshow(lpol, origin="lower", cmap="coolwarm", aspect="equal",
        # norm=LogNorm(vmin=0.005, vmax=0.05)
        vmin=0.005, vmax=0.05
        )
        plt.colorbar(cs, label="Linear Polarisation Power |P|")
        
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)

        plt.xlim(*xlims)
        plt.ylim(*ylims)
        
        print(f"Saving plot: {output}")
        
        plt.savefig(output, dpi=200)
        plt.close("all")


    @staticmethod
    def figure_14(intensity, i_image, q_image, u_image, mask, start=0.004,
        smooth_sigma=1, output="14-depolzn.png"):
        """
        # Degree of polzn lines without the contours
        # here we shall get the cube and use the first and last channels
        intesity: 
            Single image with for the intensity contours. Usually the MFS
        (i|q|u)-image
            Cubes containing data for all the channels available. I will only use the
            first and last channels available int he cube
        """
        mask_data = fits.getdata(mask).squeeze()
        mask_data = ~np.asarray(mask_data, dtype=bool).squeeze()

        intensity = rfu.get_masked_data(intensity, mask)
        i_data = fits.getdata(i_image).squeeze()
        q_data = fits.getdata(q_image).squeeze()
        u_data = fits.getdata(u_image).squeeze()

 
        lpol = np.abs(q_data + 1j*u_data)
        fpol = np.divide(lpol, i_data)[[0,-1]]

        fpol = np.ma.masked_greater(fpol, 1)
        fpol = np.ma.masked_less(fpol, 0)


        fpol = np.ma.masked_array(fpol.data, mask=np.logical_or(fpol.mask, mask_data))

        depoln = fpol[0]/fpol[-1]

        ydim, xdim = np.where(mask_data == False)
        wiggle = 70
        xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
        ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle

        wcs = rfu.read_image_cube(mask)["wcs"]

        levels = contour_levels(start, i_data)

        fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
        ax = set_image_projection(ax)

        ax.contour(
            snd.gaussian_filter(intensity, sigma=smooth_sigma),
            colors="k", linewidths=0.5, origin="lower", levels=levels)

        cs = ax.imshow(depoln, origin="lower", cmap="coolwarm", vmin=0, vmax=2, aspect="equal")
        plt.colorbar(cs, label="Depolarization ratio, [repolarization>1, depolarization <1]")

        
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)

        plt.xlim(*xlims)
        plt.ylim(*ylims)
        
        print(f"Saving plot: {output}")
        
        plt.savefig(output, dpi=200)
        plt.close("all")

    
    @staticmethod
    def figure_12_13(intensity, rm_image, emask, wmask, lmask, all_mask,
        start=0.0552, smooth_sigma=0, output="12-rm-lobes.png"):
        """
        RM and histogram for lobes 

        intensity:
            Image to be used for the intensity contours. Usually MFS
        rm_image
            Image containing the required RMs
        emask
            Eastern lobe mask
        wmask
            Western lobe mask
        all_mask
            Pictor A mask
        start
            Where the contours should start
        smooth_sigma
            Factor to smooth the contours
        """
        plt.close("all")
        rm_lobes = rfu.get_masked_data(rm_image, lmask)
        intensity = rfu.get_masked_data(intensity, all_mask)

        w_lobe = rfu.get_masked_data(rm_image, wmask)
        e_lobe = rfu.get_masked_data(rm_image, emask)

        wcs = rfu.read_image_cube(lmask)["wcs"]

        fig = plt.figure(figsize=FIGSIZE)
        
        image = plt.subplot2grid((3,3), (1,0), rowspan=2, colspan=2, projection=wcs)
        image = set_image_projection(image)
        
        # swich of these ticks
        image.tick_params(axis="x", top=False)
        image.tick_params(axis="y", right=False)
        # #image.axis("off")

        ca = image.imshow(
            rm_lobes, origin="lower", cmap="coolwarm", vmin=30,
            vmax=80, aspect="equal")
        
        plt.colorbar(ca, location="right", shrink=0.90, pad=0.01,
            label="RM", drawedges=False)

        levels = contour_levels(start, intensity)
        image.contour(
            snd.gaussian_filter(intensity, sigma=smooth_sigma),
            colors="k", linewidths=0.5, origin="lower", levels=levels)
        

        # so that I can zoom into the image easily and automatically
        ydim, xdim = np.where(rm_lobes.mask == False)
        wiggle = 10
        bins = 50
        plt.xlim(np.min(xdim)-wiggle, np.max(xdim)+wiggle)
        plt.ylim(np.min(ydim)-wiggle, np.max(ydim)+wiggle)

        west_hist = plt.subplot2grid((3,3), (1,2), rowspan=2, colspan=1)
        west_hist.hist(w_lobe.compressed(), bins=bins, log=False,
            orientation="horizontal",fill=False, ls="--", lw=1, edgecolor="blue", 
            histtype="step")

        west_hist.minorticks_on()
        west_hist.yaxis.tick_right()

        west_hist.set_title("Western Lobe (Right Hand) RM Distribution")
        west_hist.xaxis.set_visible(False)

        east_hist = plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=1)
        east_hist.hist(e_lobe.compressed(), bins=bins, log=False,
            orientation="vertical", fill=False, ls="--", lw=1, edgecolor="blue",
            histtype="step")
        east_hist.xaxis.tick_top()
        east_hist.minorticks_on()
        east_hist.set_title("Eastern Lobe (Left Hand) RM Distribution")
        east_hist.yaxis.set_visible(False)
        
        plt.subplots_adjust(wspace=.01, hspace=0)
        
        east_hist.set_xlim(-200, 200)
        west_hist.set_ylim(-200, 200)

        fig.tight_layout()
        fig.savefig(output)
        print(f"Output is at: {output}")


def run_paper_mill():
    """
    TO BE RUN IN THE IPYTHON TERMINAL
    add

        %load_ext autoreload
        %autoreload 2
        import test_paper_plots as tpp

    What are used in this function?

    1. selected image cubes from
    intermediates/selection-cubes/i-image-cube.fits',
    
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
            os.path.join(".", os.environ["sel_cubes"], "*-image-cube.fits")
            ))[:3]

    imgs = sorted(glob("./*-mfs.fits"))
    mask_dir = os.environ["mask_dir"]
    products = os.environ["prods"]

    for o_dir in ["fig4", "fig8", "fig9", "fig10"]:
        if not os.path.isdir(o_dir):
            os.mkdir(o_dir)


    mask = f"{mask_dir}/true_mask.fits"

    idata = rfu.read_image_cube(cubes[0])["data"]
    qdata = rfu.read_image_cube(cubes[1])["data"]
    udata = rfu.read_image_cube(cubes[2])["data"]


    PaperPlots.table2(cubes[0], region=f"{mask_dir}/important_regions/hotspots/core-ctrf")
    PaperPlots.table3(elobe="east-lobe-cube.fits", wlobe="west-lobe-cube.fits")

    PaperPlots.figure_8(*imgs, mask)
    PaperPlots.figure_9a(*imgs, mask)
    PaperPlots.figure_10(*imgs[1:], mask)
    PaperPlots.figure_14(imgs[0], *cubes, mask)


    images = {
        # "from_rick/pic-l-all-4k.fits": "fig4/4b-rick-intensity-contours-mpl.png", 
        "i-mfs.fits": "fig4/4b-intensity-contours-mpl.png",
        }

    for im, out in images.items():
        PaperPlots.figure_4b(im, mask, output=out)

    PaperPlots.figure_5b(
        imgs[0],
        f"{products}/spi-fitting/alpha-diff-reso.alpha.fits", mask, 
        output="5b-spi-with-contours-mpl.png")

    for _ in range(idata.shape[0]):
        PaperPlots.figure_8(
            idata[_], qdata[_], udata[_], mask, output=f"fig8/poln{_}.png")
        PaperPlots.figure_9a(
            idata[_], qdata[_], udata[_], mask, output=f"fig9/9a-dop-chan-{_}.png")
        PaperPlots.figure_10(
            qdata[_], udata[_], mask, output=f"fig10/10-lpol-chan-{_}.png")

    # Lobe stuff
    PaperPlots.figure_12_13(
        imgs[0], f"{products}/initial-RM-depth-at-peak-rm.fits",
        f"{mask_dir}/east-lobe.fits", f"{mask_dir}/west-lobe.fits", 
        f"{mask_dir}/lobes.fits", #f"{mask_dir}/no-core.fits"
        f"{mask_dir}/true_mask.fits")




def contour_levels(start, data):
    # ratio between the levels is root(2)
    print("Generating contour levels")
    levels = [start * np.sqrt(2)**_ for _ in range(30)]
    levels = np.ma.masked_greater(levels, np.nanmax(data)).compressed()
    return levels


def add_contours(axis, data, levels=None):
    """
    contours: total intensity

    Input
    -----
    axis:
        Mpl axes object onto which to add contours
    data:
        Data from which contours will be calculated
    levels: tuple; optional
        Levels for the contours
    """

    # some functions for the contour levels that can be used
    sqrt = lambda x: np.sqrt(x)
    linear = lambda x: x
    square = lambda x: np.square(x)
    power = lambda x, alpha: ((alpha**x)-1) / (alpha -1)


    if levels is None:
        # setup contour levels factor of root two between n+1 and n
        levels = contour_levels(0.004, data)

    # contour lines
    axis.contour(data, colors="k", linewidths=0.5, origin="lower", levels=levels)
    # ax.clabel(cs, inline=True, fontsize=10)


    #filled contours
    cs = axis.contourf(data, cmap="coolwarm", origin="lower", levels=levels,
                locator=ticker.LogLocator())
    plt.colorbar(cs)

    return axis
    

def add_magnetic_vectors(axis, fpol_data, pangle_data):
    """
    vector length: degree of poln (fpol)
    orientation: position angle
    """
    skip = 5
    slicex = slice(None, fpol_data.shape[0], skip)
    slicey = slice(None, fpol_data.shape[-1], skip)
    col, row = np.mgrid[slicex, slicey]

    # get M vector by rotating E vector by 90
    pangle_data = pangle_data[slicex, slicey] + (np.pi/2)
    fpol_data = fpol_data[slicex, slicey]

    # normalize this
    
    fpol_data = np.ma.masked_greater(fpol_data, 1)
    scales = fpol_data / fpol_data.max()
    u = np.cos(pangle_data) * scales
    v = np.sin(pangle_data) * scales


    # ax.contourf(row, col, Z)
    # plt.axis([2200, 2770, 2050, 2250])
    qv = axis.quiver(
        row, col, u, v, angles="xy", pivot='tail', headlength=4,
        width=0.0008, scale=5, headwidth=0)

    return axis


def plot_intensity_vectors(i_name, fpol_name, pa_name, mask=None, oup=None):
    i_data = rfu.get_masked_data(i_name, mask)
    fpol_data = rfu.get_masked_data(fpol_name, mask)
    pa_data = rfu.get_masked_data(pa_name, mask)
    
    mask = i_data.mask

    ydim, xdim = np.where(mask == False)
    wiggle = 10
    xlims = np.min(xdim)-wiggle, np.max(xdim)+wiggle
    ylims = np.min(ydim)-wiggle, np.max(ydim)+wiggle

    wcs = rfu.read_image_cube(i_name)["wcs"]

    fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': wcs})
    ax = set_image_projection(ax)

    ax = add_contours(ax, i_data)
    ax = add_magnetic_vectors(ax, fpol_data, pa_data)
    ax.tick_params(axis="x", top=False)
    ax.tick_params(axis="y", right=False)
    plt.xlim(*xlims)
    plt.ylim(*ylims)
    # fig.tight_layout()
    oname = oup + f"-intense_mfield.png"
    print(f"Output is at: {oname}")
    fig.savefig(oname)



###################################
# plot lobes with the histograms left and right

def plot_rm_for_lobes(rot_meas_image, e_mask, w_mask, vmin=None, vmax=None, oup=None, ref_image=None):
    rot_meas = rfu.read_image_cube(rot_meas_image)["data"]
    w_lobe_mask = rfu.read_image_cube(w_mask, mask=True)["data"]
    e_lobe_mask = rfu.read_image_cube(e_mask, mask=True)["data"]

    w_lobe = rfu.get_masked_data(rot_meas_image, w_mask)
    e_lobe = rfu.get_masked_data(rot_meas_image, e_mask)
    
    lobes_mask = np.bitwise_and(w_lobe_mask, e_lobe_mask)
    lobes = np.ma.masked_array(rot_meas, mask=lobes_mask)


    fig = plt.figure(figsize=FIGSIZE)
    
    wcs = rfu.read_image_cube(ref_image or rot_meas_image)["wcs"]
    image = plt.subplot2grid((3,3), (1,0), rowspan=2, colspan=2, projection=wcs)
    ca = image.imshow(
        lobes, origin="lower", cmap="magma", vmin=vmin, vmax=vmax, aspect="equal")
    plt.colorbar(ca, location="right", shrink=0.90, pad=0.01, label="RM", drawedges=False)

    image = set_image_projection(image)

    # swich of these ticks
    image.tick_params(axis="x", top=False)
    image.tick_params(axis="y", right=False)
    # #image.axis("off")

    # so that I can zoom into the image easily and automatically
    ydim, xdim = np.where(lobes.mask == False)
    wiggle = 10
    plt.xlim(np.min(xdim)-wiggle, np.max(xdim)+wiggle)
    plt.ylim(np.min(ydim)-wiggle, np.max(ydim)+wiggle)

    west_hist = plt.subplot2grid((3,3), (1,2), rowspan=2, colspan=1)
    west_hist.hist(w_lobe.compressed(), bins=20, log=True,
        orientation="horizontal",fill=False, ls="--", lw=1, edgecolor="blue", 
        histtype="step")
    west_hist.yaxis.tick_right()

    west_hist.set_title("Western Lobe (Right Hand) RM Distribution")
    west_hist.xaxis.set_visible(False)

    east_hist = plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=1)
    east_hist.hist(e_lobe.compressed(), bins=20, log=False,
        orientation="vertical", fill=False, ls="--", lw=1, edgecolor="blue",
        histtype="step")
    east_hist.xaxis.tick_top()
    east_hist.set_title("Eastern Lobe (Left Hand) RM Distribution")
    east_hist.yaxis.set_visible(False)
    
    plt.subplots_adjust(wspace=.01, hspace=0)

    fig.tight_layout()
    oname = oup + f"-lobes_rm{EXT}"
    fig.savefig(oname)
    print(f"Output is at: {oname}")
    


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


def parser():
    ps = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        description="Script to genernetae some plots for my paper")
    ps.add_argument("-rim", "--ref-i-image", dest="ref_image", 
        help="Reference I image for the contour plots")
    ps.add_argument("--cube-name", dest="cube_names", nargs="+",
        help="Input I q AND U same resolution cubes")
    ps.add_argument("--input-maps", dest="prefix", 
        required=True,
        help="Input prefix for the fpol rm and whatever maps")
    ps.add_argument("--mask-name", dest="mask_name",
        help="Name of the full field mask to use"
        )
    ps.add_argument("-elm", "--e-lobe-mask", dest="elobe_mask",
        help="Mask for the eastern lobe"
        )
    ps.add_argument("-wlm", "--w-lobe-mask", dest="wlobe_mask",
        help="Mask for the western lobe"
        )
    ps.add_argument("-o", "-output-dir", dest="output_dir",
        help="Where to dump the outputs"
        )
    return ps

######################################################################################
# Main
######################################################################################

if __name__ == "__main__":

    run_paper_mill()

    """
        opts = parser().parse_args()

        # names for the various rm products
        map_names = {
            "amp" : f'{opts.prefix}-p0-peak-rm.fits',
            "angle" : f'{opts.prefix}-PA-pangle-at-peak-rm.fits',
            "fpol" : f'{opts.prefix}-FPOL-at-center-freq.fits',
            "rm" : f'{opts.prefix}-RM-depth-at-peak-rm.fits'
        }

        images = [map_names[_] for _ in "amp angle rm".split()]
        
        pica_i_data = rfu.get_masked_data(opts.ref_image, opts.mask_name)
        pangle_data = rfu.read_image_cube(map_names["angle"])["data"]
        pica_mask = pica_i_data.mask

        stokes = {}
        for cube in opts.cube_names:
            stoke = os.path.basename(cube)[0]
            stokes[stoke] = rfu.read_image_cube(cube)["data"]

        stokes["l_pol"] = stokes["q"] + 1j*stokes["u"]
        stokes["f_pol"] = (stokes["q"]/stokes["i"]) + ((1j*stokes["u"])/stokes["i"])
        
        # plot lobes and their dispersion
        plot_rm_for_lobes(
            rot_meas_image=map_names["rm"],
            e_mask=opts.elobe_mask, w_mask=opts.wlobe_mask,
            vmin=-100, vmax=100, oup=opts.prefix, ref_image=opts.ref_image)


        # plot fpol
        plot_fractional_polzn(stokes["f_pol"], oup=opts.prefix, mask=pica_mask,
            ref_image=opts.ref_image)
        
        plot_polarised_intensity(
            stokes["l_pol"], oup=opts.prefix, ref_image=opts.ref_image,
            mask=pica_mask)

        plot_intensity_vectors(
            opts.ref_image, map_names["fpol"], map_names["angle"],
            mask=opts.mask_name, oup=opts.prefix)


        '''
        Running this script with
        
        python qu_pol/test_paper_plots.py --input-maps $prods/initial -rim i-mfs.fits --cube-name $conv_cubes/*-conv-image-cube.fits --mask-name $mask_dir/true_mask.fits -elm $mask_dir/east-lobe.fits -wlm $mask_dir/west-lobe.fits -o $prods/some-plots
        '''

    """