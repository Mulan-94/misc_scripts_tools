#!/bin/bash
set -e

echo -e "******************************************************";
echo -e "   Welcome to MOST THINgs RM with Lexy";
echo -e "******************************************************";
echo -e "\n Place this script where the products will be placed"
echo -e "This script requires some preliminary setup as follows";
echo -e "In this same directory"
echo -e "\n  1. Access to qu_pol from misc_tools_n_scripts/qu_pol";
echo -e "  2. Access to plotting_bmaj_bmin.py from misc_tools_n_scripts/fits_related/";
echo -e "  3. Images being worked on should be in this dirs' parent directory '../'\n";
echo -e "  NOTE: PLEASE SUPPLY WHERE TO FIND YOUR MASSKSKSSSS!!!!!"
echo -e "******************************************************";


echo "To be run as:";
echo -e "\t ./run_most_rm_things location/of/mask/dir";
echo "e.g"
echo -e "\t ./run_most_rm_things.sh /home/andati/pica/reduction/experiments/emancipation/masks-572"
echo -e "\t ./run_most_rm_things.sh /home/andati/pica/reduction/experiments/emancipation/masks"

echo -e "\n\nReading the mask dir"
if [[ $1 = "" ]];
then
	echo "Mask dir not specified, I will use the default one";
	export mask_dir=$HOME/pica/reduction/experiments/emancipation/masks;
	echo "Mask dir:  $mask_dir";
else
	echo "Mask dir:  $1";
	export mask_dir=$1;
fi



# REference: https://devconnected.com/how-to-check-if-file-or-directory-exists-in-bash/
echo -e"\nchecking if the required scripts exist";
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


echo -e "\nSetting up variables, and the selection of channels";
stokes="I Q U";

echo -e "\n############################################################"
echo "get the selected channels. Should be stored in a file called selected-channels"
echo -e "############################################################\n"

if [[ ! -f selected-channels.txt ]]
then
	cp ../selected-channels.txt .
fi
sel=($(echo $(cat selected-channels.txt)))


export orig_cubes="intermediates/original-cubes";
export sel_cubes="intermediates/selection-cubes";
export conv_cubes="intermediates/conv-selection-cubes";
export plots="intermediates/beam-plots";
export prods="products";
export spis="products/spi-fitting";

echo -e "\n############################################################"
echo "Make these directories";
echo -e "############################################################\n"
mkdir -p $orig_cubes $sel_cubes $conv_cubes $plots $prods $spis;



for s in $stokes;
	do
		echo "make cubes from ALL the output images: ${s^^}";
		
		images=$(ls ../*-[0-9][0-9][0-9][0-9]-$s-image.fits)
		fitstool.py --stack=$orig_cubes/${s,,}-cube.fits:FREQ $(echo $images);

		if [[ ${s,,} = "i" ]]; then
			images=$(ls ../*-[0-9][0-9][0-9][0-9]-$s-model.fits);
			fitstool.py --stack=$orig_cubes/${s,,}-model-cube.fits:FREQ $(echo $images);
			images=$(ls ../*-[0-9][0-9][0-9][0-9]-$s-residual.fits);
			fitstool.py --stack=$orig_cubes/${s,,}-residual-cube.fits:FREQ $(echo $images);
		fi
	done;


echo -e "\n############################################################"
echo "Plot the beams to identify which should be flagged";
echo -e "############################################################\n"
python plotting_bmaj_bmin.py -c $orig_cubes/i-cube.fits -o $plots/orig-bmaj-bmin-i;


echo -e "\n############################################################"
echo "copy relevant channels' images to this folder for IQUV";
echo -e "############################################################\n"
for n in ${sel[@]};
	do
		cp ../*-$n-{I,Q,U}-*image* .;
	done


echo -e "\n############################################################"
echo "copy I MFS image reference image here";
echo -e "############################################################\n"
cp ../*MFS-I-image.fits i-mfs.fits
cp ../*MFS-Q-image.fits q-mfs.fits
cp ../*MFS-U-image.fits u-mfs.fits


echo -e "\n############################################################"
echo "Save the names of the selected images. Simpleton workaround for using cubes in the scap.py script :("
echo -e "############################################################\n"
ls *-[0-9][0-9][0-9][0-9]*-image.fits > selected-freq-images.txt

#Replacing all begins of strings here with ../
sed -i 's/^/\.\.\//g' selected-freq-images.txt


echo -e "\n############################################################"
echo "write out the selected freqs into a single file for easier work. This is for use in the RM synth for wavelength":
echo -e "############################################################\n"
for im in *-[0-9][0-9][0-9][0-9]*-I-image.fits;
	do
		fitsheader -k CRVAL3 $im |  grep -i CRVAL3 >> frequencies.txt;
	done;


echo -e "\n############################################################"
echo "Cleanup freqs file by replacing CRVAL3 = and all spaces after it with emptiness";
echo -e "############################################################\n"
sed -i "s/CRVAL3  =\ \+//g" frequencies.txt


echo -e "\n############################################################"
echo "Selections steps"
echo -e "############################################################\n"
for s in $stokes;
	do
		echo "Make the selection cubes: ${s^^}";
		images=$(ls *-[0-9][0-9][0-9][0-9]-$s-image.fits);
		fitstool.py --stack=$sel_cubes/${s,,}-image-cube.fits:FREQ $(echo $images);
		
		echo "Convolve the cubes to the same resolution";
		spimple-imconv -image $sel_cubes/${s,,}-image-cube.fits -o $conv_cubes/${s,,}-conv-image-cube ;
	done


echo -e "\n############################################################"
echo "Plot the beams of selected channels to see if they make sense";
echo -e "############################################################\n"
python plotting_bmaj_bmin.py -c $sel_cubes/i-image-cube.fits -o $plots/selected-bmaj-bmin-i;



echo -e "\n############################################################"
echo "Renaming output file from spimple because the naming here is weird";
echo -e "############################################################\n"
rename.ul -- ".convolved.fits" ".fits" $conv_cubes/* ;



echo -e "\n############################################################"
echo "Just check if the beam sizes are the same";
echo -e "############################################################\n"
python plotting_bmaj_bmin.py -c $conv_cubes/*-conv-image-cube.fits -o $plots/conv-bmamin;


echo -e "\n############################################################"
echo "Delete the copied image files";
echo -e "############################################################\n"
rm ./*-[0-9][0-9][0-9][0-9]-*image.fits



echo -e "\n############################################################"
echo "Generate various interesting LoS above some certain threshold";
echo -e "############################################################\n"

#what to name the stuff: for the scrapper!
data_suffix="circle-t20";


# I change region size from 5pix to 3 pixels
python qu_pol/scrap.py -rs 3 -t $data_suffix -f selected-freq-images.txt --threshold 20 --output-dir $prods/scrap-outputs -wcs-ref i-mfs.fits



echo -e "\n############################################################"
echo "Edit the reg file in a way that it can be loaded into CARTA"
echo -e "############################################################\n"

sed -i "s/text=.*//g" $prods/scrap-outputs/regions/beacons*.reg



echo -e "\n############################################################"
echo "Perfrom RM synthesis for various lines of sight generated from previous step and plot the output";
echo "For pictor I set the maximum depth to 400, depth step 1 looks smoothest, niter from 500-1000 looks similar"
echo -e "############################################################\n"
python qu_pol/rm_synthesis.py -id $prods/scrap-outputs/*$data_suffix -od $prods/rm-plots -md 400 --depth-step 1


echo -e "\n############################################################"
echo "Generate interactive plots for the various LoS";
echo -e "############################################################\n"
python qu_pol/bokeh/plot_bk.py -id $prods/scrap-outputs/*$data_suffix --yaml qu_pol/bokeh/plots.yml -od $prods/bokeh-plots


echo -e "\n############################################################"
echo "Do some RM maps, fpol maps and other maps";
echo "Using my mask here, Don't know where yours is but if this step fails, check on that";
echo "Default maximum depth and number of iterations same as that of previous"
echo -e "############################################################\n"
python qu_pol/pica_rm-x2.py -q $conv_cubes/q-conv-image-cube.fits -u $conv_cubes/u-conv-image-cube.fits -i $conv_cubes/i-conv-image-cube.fits -ncore 120 -o $prods/initial -mask $mask_dir/true_mask.fits -f frequencies.txt 



# Doing some SPI maps

echo -e "\n############################################################"
echo "Copy I selected models and residuals";
echo -e "############################################################\n" 

# For selection with LS using or patterns
# https://unix.stackexchange.com/questions/50220/using-or-patterns-in-shell-wildcards


for n in ${sel[@]};
	do
		cp ../*-$n-I-{residual,model}* .;
	done


echo -e "\n############################################################"
echo "Get their wsums and store";
echo -e "############################################################\n" 

# Get wsums for the selected images with commas
# echo $(fitsheader *-model.fits | grep -i wsum | sed s"/WSCVWSUM=\s*//g") | sed "s/ /,/g"

fitsheader *I-model.fits | grep -i wsum | sed s"/WSCVWSUM=\s*//g" > wsums.txt



echo -e "\n############################################################"
echo "stack I residuals and models";
echo -e "############################################################\n"


images=$(ls *-[0-9][0-9][0-9][0-9]-*residual.fits);
fitstool.py --stack $sel_cubes/i-residuals.fits:FREQ $(echo $images);

images=$(ls *-[0-9][0-9][0-9][0-9]-*model.fits);
fitstool.py --stack $sel_cubes/i-models.fits:FREQ $(echo $images);

echo "Rename the convolved images";
# Adding || tru so that the error here does not fail the entire program
# see: https://stackoverflow.com/questions/11231937/bash-ignoring-error-for-a-particular-command
rename.ul -- ".convolved.fits" ".fits" $conv_cubes/* || true;


echo -e "\n############################################################"
echo "Delete the copied models and residuals";
echo -e "############################################################\n" 
rm ./*-{model,residual}.fits ;



echo -e "\n############################################################"
echo "Do the SPI fitting";
echo -e "############################################################\n" 

echo "Normalize the wsums by the largest values";

# # Doing this with a quick python script because, wll I can :) and store in this variable
wsums=$(python -c "import numpy as np; wsums = np.loadtxt('wsums.txt'); wsums = np.round(wsums/wsums.max(), 4); print(*wsums)")

# cw - channel weights, th-rms threshold factor, acr - add conv residuals, bm -  beam model
spimple-spifit -model $sel_cubes/i-models.fits -residual $sel_cubes/i-residuals.fits -o $spis/alpha-diff-reso -th 10 -nthreads 32 -pb-min 0.15 -cw $wsums -acr -bm JimBeam -band l --products aeikb


echo -e "\n############################################################"
echo "Generate images for the respective lobes";
echo -e "############################################################\n"


fitstool.py --prod $sel_cubes/i-image-cube.fits $mask_dir/east-lobe.fits -o ./east-lobe-cube.fits
fitstool.py --prod $sel_cubes/i-image-cube.fits $mask_dir/west-lobe.fits -o ./west-lobe-cube.fits


echo -e "\n############################################################"
echo "Make some other plots for paper";
echo -e "############################################################\n" 

# python qu_pol/test_paper_plots.py --input-maps $prods/initial -rim i-mfs.fits --cube-name $conv_cubes/*-conv-image-cube.fits --mask-name $mask_dir/true_mask.fits -elm $mask_dir/east-lobe.fits -wlm $mask_dir/west-lobe.fits -o $prods/some-plots

python qu_pol/test_paper_plots.py

echo -e "\n############################################################"
echo "                    Done                     "
echo -e "**************************************************************"