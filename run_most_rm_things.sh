#!/bin/bash
set -e

echo "******************************************************";
echo "   Welcome to MOST THINgs RM with Lexy";
echo "******************************************************";
echo "\n Place this script where the products will be placed"
echo "This script requires some preliminary setup as follows";
echo "In this same directory"
echo -e "\n  1. Access to qu_pol from misc_tools_n_scripts/qu_pol";
echo "  2. Access to plotting_bmaj_bmin.py from misc_tools_n_scripts/fits_related/";
echo -e"  3. Images being worked on should be in this dirs' parent directory '../'\n";
echo "******************************************************";



# REference: https://devconnected.com/how-to-check-if-file-or-directory-exists-in-bash/
echo "checking if the required scripts exist";
if [[ ! -f plotting_bmaj_bmin.py ]]
then
	echo "plotting_bmaj_bmin.py FILE does not exist. Creating";
	ln -s $HOME/git_repos/misc_scripts_n_tools/fits_related/plotting_bmaj_bmin.py;
fi

if [[ ! -d qu_pol ]]
then
	echo "qu_pol DIR does not exist. Creating";
	ln -s $HOME/git_repos/misc_scripts_n_tools/qu_pol;
fi


echo "Setting up variables, and the selection of channels";
stokes="I Q U V";
sel=("03" "04" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "42" "43" "44" "45" "46" "47" "48" "49" "50" "51" "52" "53" "54" "55" "60" "72" "73" "74");

orig_cubes="intermediates/original-cubes";
sel_cubes="intermediates/selection-cubes";
conv_cubes="intermediates/conv-selection-cube";
plots="intermediates/beam-plots";
prods="products";


echo "Make these directories";
mkdir -p $orig_cubes;
mkdir -p $sel_cubes;
mkdir -p $conv_cubes;
mkdir -p $plots;
mkdir -p $prods;

for s in $stokes;
	do
		echo "make cubes from ALL the output images: ${s^^}";
		fitstool.py --stack=$orig_cubes/${s,,}-cube.fits:FREQ ../*-[0-9][0-9][0-9][0-9]-$s-image.fits;

		echo "Plot the beams to identify which should be flagged";
		python plotting_bmaj_bmin.py -c $orig_cubes/${s,,}-cube.fits -o $plots/orig-bmaj-bmin-$s
	done;


echo "copy relevant channels' images to this folder for IQUV";
for n in ${sel[@]};
	do
		cp ../*-[0-9][0-9]$n-*-*image* .;
	done


echo "Save the names of the selected images. Simpleton workaround for using cubes in the scap.py script :("
ls *-[0-9][0-9][0-9][0-9]*-image.fits >> selected-freq-images.txt


echo "write out the selected freqs into a single file for easier work. This is for use in the RM synth for wavelength":
for im in *-[0-9][0-9][0-9][0-9]*-I-image.fits;
	do
		fitsheader -k CRVAL3 $im |  grep -i CRVAL3 >> frequencies.txt;
	done;

echo "Cleanup freqs file by replacing CRVAL3 = and all spaces after it with emptiness";
sed -i "s/CRVAL3  =\ \+//g" frequencies.txt


for s in $stokes;
	do
		echo "Make the selection cubes: ${s^^}";
		fitstool.py --stack=$sel_cubes/${s,,}-sel-cube.fits:FREQ  -F "*[0-9][0-9][0-9][0-9]-$s-*image*";
		
		echo "Convolve the cubes to the same resolution";
		spimple-imconv -image $sel_cubes/${s,,}-sel-cube.fits -o $conv_cubes/${s,,}-conv ;

		echo "Just check if the beam sizes are the same";
		python plotting_bmaj_bmin.py -c $conv_cubes/${s,,}-conv-cube.fits -o $plots/conv-bmaj-bmin-$s;
	done


echo "Delete the copied image files";
rm *-[0-9][0-9][0-9][0-9]-*image.fits



echo "Generate various interesting LoS above some certain threshold";
python qu_pol/scrap.py -rs 20 -t circle-t0.05 -f selected-freq-images.txt --threshold 0.05 --output-dir $prods ;

echo "Perfrom RM synthesis for various lines of sight generated from previous step and plot the output";
python qu_pol/rm_synthesis.py -id IQU-regions-mpc-*circle-t0.05 -od $prods/rm-plots -md 1200

echo "Generate interactive plots for the various LoS";
python qu_pol/bokeh/plot_bk.py -id IQU-regions-mpc-*-circle --yaml qu_pol/bokeh/plots.yml 


echo "Do some RM maps, fpol maps and other maps";
echo "Using my mask here, Don't know where yours is but if this step fails, check on that";
python qu_pol/pica_rm-x2.py -q $conv_cubes/q-conv.fits -u $conv_cubes/u-conv.fits -i $conv_cubes/i-conv.fits -ncore 120 -o ./products/initial -mask ../../masks/true_mask.fits -f frequencies.txt 

