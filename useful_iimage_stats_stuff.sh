"""
This file contains some code snippets that I found useful while I was dealing with sectioning images into regions and getting the appropriate stats for those regions. They are gotten from various source, amongst which are:
. CARTAVIS backend code
. CASA documentation
. Astropy regions documentation
"""
# 37b-QU-for-RM-0-255-MFS-Q-image.fits statistics
# rotbox[[74.000000pix, 190.000000pix], [50.000000pix, 50.000000pix], 0.000000deg]
# rotbox(wcs:FK5)[[5:20:09.9945847177, -45:48:19.7879995436], [50.0000000000", 50.0000000000"], 0.000000deg]
# Statistic	Value	Unit
NumPixels
Sum  
FluxDensity
Mean    
StdDev  
Min 
Max    
Extrema  
RMS     
SumSq  


# from https://github.com/CARTAvis/carta-backend/blob/dev/src/ImageStats/BasicStatsCalculator.tcc
mean = _sum / _num_pixels;
stdDev = _num_pixels > 1 ? sqrt((_sum_squares - (_sum * _sum / _num_pixels)) / (_num_pixels - 1)) : NAN;
rms = sqrt(_sum_squares / _num_pixels);

_min_val = std::min(_min_val, other._min_val);
    _max_val = std::max(_max_val, other._max_val);
    _num_pixels += other._num_pixels;
    _sum += other._sum;
    _sum_squares += other._sum_squares;


######## FROM TIGGER
[49:100,164:215] min -0.02927, max 0.1882, mean -0.0008863, std 0.01954, sum -2.305, np 2601




#######3 DS9
# eg	sum	error		area		surf_bri		surf_err
# 				(arcsec**2)		(sum/arcsec**2)	(sum/arcsec**2)



### CASA IMSTAT
# see https://casa.nrao.edu/casadocs/casa-6.1.0/imaging/image-analysis/image-selection-parameters


"""
https://casa.nrao.edu/casadocs/casa-6.1.0/usingcasa/obtaining-and-installing

python3.6 -m venv casa6
pip install --index-url https://casa-pip.nrao.edu/repository/pypi-casa-release/simple casatools
pip install --index-url https://casa-pip.nrao.edu/repository/pypi-casa-release/simple casatasks
python3 -m casatools --update-user-data

"""
box(blc, trc)
#imstat(imagename="channelised/0-255/37b-QU-for-RM-0-255-MFS-Q-image.fits",axes=-1,region="",box="48,164,98,214",chans="",stokes="",listit=True,verbose=True,mask="",stretch=False,logfile="",append=True,algorithm="classic",fence=-1,cente)




# reg_file = "channelised/beacons2-pix-mod.reg"
file_core = "regions-mpc"

reg_file = generate_regions(factor=50, "regions/beacons2.reg")
regs = regions.Regions.read(reg_file, format="ds9")

noise_reg = regs.pop(-1)

start = perf_counter()

for stokes in "Q U".split():
    images = sorted(glob(f"./channelised/*/*00*{stokes}*image*"))
    print(f"Working on Stokes {stokes}")
    
    # out_dir = make_out_dir(f"{stokes}-{file_core}")
    # for reg in regs:
    #     #Q and U for each region
    #     fluxes, waves = [], []
    #     print(f"Region: {reg.meta['label']}")

    #     with futures.ProcessPoolExecutor() as executor:
    #         results = executor.map(
    #             partial(extract_stats, reg=reg, noise_reg=noise_reg), images
    #             )

        # for fname in images:
        #     res = extract_stats(fname, reg, noise_reg, sig_factor=10)
        #     if res is not None:
        #         flux, wave = res
        #         fluxes.append(flux)
        #         waves.append(wave)
        # set_trace()
        # np.savez(os.path.join(out_dir, f"{reg.meta['label']}_{stokes}"), 
        #     flux=fluxes, waves=waves)
    get_image_stats(stokes, file_core, images, regs)

plot_spectra(file_core, f"QU-{file_core}")

print(f"Finished in {perf_counter() - start} seconds")
print("======================================")