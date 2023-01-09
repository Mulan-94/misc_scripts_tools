#!/bin/bash
set -e

# redirect output to this logfile just in case
# see: https://serverfault.com/questions/103501/how-can-i-fully-log-all-bash-scripts-actions
exec 1>garf_web.log 2>&1

#read -p "Which directory to get the files from" indir;
#read -p "Where to dump the files on the website" outdir;


if [[ -d images ]];
then
    echo "Deleting current image dir";
    rm -r images;
fi

if [[ -d plots ]];
then
    echo "Deleting current plots dir";
    rm -r plots;
fi

# create images directory
mkdir images

indir="/home/andati/pica/reduction/experiments/emancipation/6-not-derotated-after-6-kunislope-t8-f64-iquv/00-polarimetry";
outdir="/data/andati/pica_data";


echo "# copy necessary stuff from server ie. mfs image, rm map and regions file"
rsync -aHP andati@garfunkel:$indir/i-mfs.fits images/pica-I-mfs.fits
rsync -aHP andati@garfunkel:$indir/products/initial-RM-depth-at-peak-rm.fits images/rm-map.fits
rsync -aHP andati@garfunkel:$indir/products/scrap-outputs/regions/beacons-thresh-*.reg images/pica-beacons-fk5.reg
echo "FITS COPY DONE"
echo "-----------------------------------------------";

echo "# copy the plots from server"
echo "============================"
rsync -aHP andati@garfunkel:$indir/products/bokeh-plots .
echo "PLOTS COPY DONE"
echo "-----------------------------------------------";

mv bokeh-plots plots;

echo "Copy files to cygnus data"

echo "1. Deleting the current plots and images dirs in cygnus"
echo "======================================================="
ssh andati@cygnus "rm -r $outdir/plots $outdir/images"

echo "2. Copying over"
echo "==============="
rsync -aHP images plots andati@cygnus:$outdir/ ;

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";
