import argparse

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


def parser():
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

    genopts = parsing.add_argument_group("General arguments")
    parsing.add_argument("-todo", "--todo", dest="todo", 
        type=str, metavar="",
        help="""A string containing to do items. Specify using:
        (r): generate regions,
        (l): generate LOS data,
        (p): generate plots.
        Default is 'rl'
        """)
    genopts.add_argument("--noverwrite", dest="noverwrite", action="store_false",
        help="Do not ovewrite everything along the way. Default is overwrite")
    genopts.add_argument("-o", "-odir", "--output-dir", dest="odir", type=str,
        default=None, metavar="",
        help="where to dump output")    
    genopts.add_argument("--threshold", dest="threshold", metavar="", type=int,
        default=None,
        help="""If SNR below which data will not be considered. Default is 3""")
    genopts.add_argument("-j", "--nworkers", dest="nworkers", metavar="",
        type=int, default=None,
        help="How many workers to use for prcessing"
    )

    

    regopts = parsing.add_argument_group("Region generation arguments")
    regopts.add_argument("-ro", "--regions-only", dest="regions_only",
        action="store_true", help="Only generate the region files")
    
    regopts.add_argument("-mrn", "--minimum-region-noise", dest="rnoise",
        type=float, default=None, metavar="", 
        help="""Specific noise floor to generate the regions.
        """)
    regopts.add_argument("-ref-image", "--reference-image", dest="wcs_ref",
        metavar="", default=None, required=True,
        help="""Ther reference image that will be used to generate the default
        region file. Must be the stokes I MFS image. 
        This image will also be used to get the reference WCS for region file
        genenration. Default is a file named 'i-mfs.fits'
        """)
    regopts.add_argument("-nri", "--noise-reference-image", dest="noise_ref",
        metavar="", default=None, required=True,
        help="""The total intensity image used to get the noise reference.
        Defaults to a file named 'i-mfs.fits'."""
        )
    regopts.add_argument("-rf", "--region-file", dest="rfile", type=str,
        default=None, metavar="", 
        help="""An input region file. Otherwise, one will be auto-generated.
        Genereated regions will be stored here
        """)
    regopts.add_argument("-rs", "--region-size", dest="reg_size", #nargs="+",
        type=int, #default=[],
         metavar="", 
        help=("Create regions of this circle radius and perform analyses on them."+
        " If you want to set the data threshold, please use --threshold."))
    regopts.add_argument("-xr", "--x-range", dest="x_range", metavar="",
        type=float, nargs=2, default=None,
        help="""Space separated list containing RA in degrees within which 
        to generate regions""")
    regopts.add_argument("-yr", "--y-range", dest="y_range", metavar="",
        type=float, nargs=2, default=None,
        help="""Space separated list containing DEC in degrees within which 
        to generate regions""")
    regopts.add_argument("-m", "--mask", dest="mask", metavar="",
        type=str, default=None,
        help="""Mask containing the area where LoS should be restricted.
        This will supercede -xr and -yr options. They are mutually exclusive.
        """)



    losopts = parsing.add_argument_group("LoS Data generation arguments")
    losopts.add_argument("-idir", "--image_dir", dest="image_dir", type=str,
        metavar="", required=True,
        help="Where the channelised I, Q and U images are")
    losopts.add_argument("-lo", "--los-only", dest="los_only",
        action="store_true",
        help="Only generate the line of sight data files")
    losopts.add_argument("-mft", "--minimum-flag-threshold", dest="mft",
        type=float, default=None, metavar="", help="""
        Fraction of flags above which lines of sight should be ignored.  
        Can be useful if you want to plot all the generated LOS. Otherwise, they
        will be filtered out. The simple filter is that where values of fractional
        polarisation i are >1 or <0, this data is 'flagged'.
        behaviour: flag size > 0.7 of total data is flagged, ingore. 
        Max is 1, min is >0.
        Default 0.7
        """
    )
    losopts.add_argument("-psnr", "--use-polzd-snr", action="store_true",
        dest="polzd_snr",
        help="""Use to elect use of polarised SNR to determine valid LoS. 
        Default S/N used is total intensity: rms noise""")

  
    
    #plotting arguments  
    plot_parsing = parsing.add_argument_group("Plotting Arguments")
    plot_parsing.add_argument("-po", "--plots-only", dest="plots_only",
        action="store_true", help="Only do plots")
    plot_parsing.add_argument("--plot-grid", dest="plot_grid",
        action="store_true",
        help="Enable to make gridded plots")
    plot_parsing.add_argument("-piqu", "--plot-iqu", dest="plot_qu",
        action="store_true",
        help="Plot Q and U values")
    plot_parsing.add_argument("-pfp", "--plot-frac-pol", dest="plot_frac_pol",
        action="store_true",
        help="Plot Fractional polarization")
    plot_parsing.add_argument("-plp", "--plot-linear-pol",
        dest="plot_linear_pol", action="store_true",
        help="Plot linear polarization power")
    plot_parsing.add_argument("--ymax", dest="ymax", type=float, metavar="",
        help="Y axis max limit")
    plot_parsing.add_argument("--ymin", dest="ymin", type=float,metavar="",
        help="Y axis min limit")
    plot_parsing.add_argument("--xmax", dest="xmax", type=float, metavar="",
        help="Y axis max limit")
    plot_parsing.add_argument("--xmin", dest="xmin", type=float,metavar="",
        help="Y axis min limit")
    plot_parsing.add_argument("-p", "--plot", dest="plot", nargs="*",
        type=int, metavar="",
        help="Make plots for these region sizes manually. These will be linearly scaled")
    plot_parsing.add_argument("-ps", "--plot-scales", dest="plot_scales", metavar="",
        default=["linear"], nargs="*", choices=["linear", "log"],
        help=("Scales for the plots. Can be a space separated list of " + 
            "different scales. Options are linear or log."))

    return parsing

