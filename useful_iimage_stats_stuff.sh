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



'''
calculate the SPECtral index of some images CASA spi
1. Select the highest and lowest frequency images
2. Convolve them to the same resolution
3. Multiply conv-image x mask (IN THAT ORDER) using fitstools
4. Run the following funciton on those two images
'''

from casatasks import immath
from casatasks import exportfits


def casa_fit(im1, im2, output="casa-fit"):
    immath(imagename=[im1, im2], mode='spix', outfile='casa-fit.im')
    exportfits(imagename=f"{output}.im", fitsimage=f"{output}.fits")
    return

