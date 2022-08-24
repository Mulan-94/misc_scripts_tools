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
echo -e "******************************************************";



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


export orig_cubes="intermediates/original-cubes";
export sel_cubes="intermediates/selection-cubes";
export conv_cubes="intermediates/conv-selection-cubes";
export plots="intermediates/beam-plots";
export prods="products";
export spis="products/spi-fitting";
export mask_dir=$HOME/pica/reduction/experiments/emancipation/masks;

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

		echo "Plot the beams to identify which should be flagged";
		python plotting_bmaj_bmin.py -c $orig_cubes/${s,,}-cube.fits -o $plots/orig-bmaj-bmin-$s
	done;


echo -e "\n############################################################"
echo "copy relevant channels' images to this folder for IQUV";
echo -e "############################################################\n"
for n in ${sel[@]};
	do
		cp ../*-[0-9][0-9]$n-*-*image* .;
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
ls *-[0-9][0-9][0-9][0-9]*-image.fits >> selected-freq-images.txt

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

echo "Renaming output file from spimple because the naming here is weird";
rename.ul -- ".convolved.fits" ".fits" $conv_cubes/* ;

echo "Just check if the beam sizes are the same";
python plotting_bmaj_bmin.py -c $conv_cubes/*-conv-image-cube.fits -o $plots/conv-bmamin;


echo -e "\n############################################################"
echo "Delete the copied image files";
echo -e "############################################################\n"
rm *-[0-9][0-9][0-9][0-9]-*image.fits



echo -e "\n############################################################"
echo "Generate various interesting LoS above some certain threshold";
echo -e "############################################################\n"

#what to name the stuff
data_suffix="circle-t0.05";

# Appropriate 20-best, 18-better, 16-more-data
python qu_pol/scrap.py -rs 5 -t $data_suffix -f selected-freq-images.txt --threshold 0.05 --output-dir $prods/scrap-outputs -wcs-ref i-mfs.fits --regions-threshold 16



echo -e "\n############################################################"
echo "Perfrom RM synthesis for various lines of sight generated from previous step and plot the output";
echo -e "############################################################\n"
python qu_pol/rm_synthesis.py -id $prods/scrap-outputs/*$data_suffix -od $prods/rm-plots -md 1200


echo -e "\n############################################################"
echo "Generate interactive plots for the various LoS";
echo -e "############################################################\n"
python qu_pol/bokeh/plot_bk.py -id $prods/scrap-outputs/*$data_suffix --yaml qu_pol/bokeh/plots.yml -od $prods/bokeh-plots


echo -e "\n############################################################"
echo "Do some RM maps, fpol maps and other maps";
echo "Using my mask here, Don't know where yours is but if this step fails, check on that";
echo -e "############################################################\n"
python qu_pol/pica_rm-x2.py -q $conv_cubes/q-conv-image-cube.fits -u $conv_cubes/u-conv-image-cube.fits -i $conv_cubes/i-conv-image-cube.fits -ncore 120 -o $prods/initial -mask $mask_dir/true_mask.fits -f frequencies.txt 



# Doing some SPI maps

echo -e "\n############################################################"
echo "Copy I selected models and residuals";
echo -e "############################################################\n" 

# For selection with LS using or patterns
# https://unix.stackexchange.com/questions/50220/using-or-patterns-in-shell-wildcards

# copy I residuals and models of the selected channels
cp ../*-00{03,04,10,11,12,13,14,15,16,17,18,19,20,42,43,44,45,46,47,48,49,50,51,52,53,54,55,60,72,73,74}-I-{residual,model}.fits .


echo -e "\n############################################################"
echo "Get their wsums and store";
echo -e "############################################################\n" 

# Get wsums for the selected images with commas
# echo $(fitsheader *-model.fits | grep -i wsum | sed s"/WSCVWSUM=\s*//g") | sed "s/ /,/g"

fitsheader *-model.fits | grep -i wsum | sed s"/WSCVWSUM=\s*//g" >> wsums.txt



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
rm *-model.fits *-residual.fits ;



echo -e "\n############################################################"
echo "Do the SPI fitting";
echo -e "############################################################\n" 

echo "Normalize the wsums by the largest values";

# # Doing this with a quick python script because, wll I can :) and store in this variable
wsums=$(python -c "import numpy as np; wsums = np.loadtxt('wsums.txt'); wsums = np.round(wsums/wsums.max(), 4); print(*wsums)")

# cw - channel weights, th-rms threshold factor, acr - add conv residuals, bm -  beam model
spimple-spifit -model $sel_cubes/i-models.fits -residual $sel_cubes/i-residuals.fits -o $spis/alpha-diff-reso -th 10 -nthreads 32 -pb-min 0.15 -cw $wsums -acr -bm JimBeam -band l --products aeikb


echo -e "\n############################################################"
echo "Genertate images for the respective lobes";
echo -e "############################################################\n"

for im in "iquv"; do
	fitstool.py --prod $sel_cubes/$im-image-cube.fits $mask_dir/true_mask.fits -o $sel_cubes/$im-mxd-image-cube.fits;
	done


echo -e "\n############################################################"
echo "Make some other plots for paper";
echo -e "############################################################\n" 

python qu_pol/test_paper_plots.py --input-maps $prods/initial -rim i-mfs.fits --cube-name $conv_cubes/*-conv-image-cube.fits --mask-name $mask_dir/true_mask.fits -elm $mask_dir/east-lobe.fits -wlm $mask_dir/west-lobe.fits -o $prods/some-plots


echo -e "\n############################################################"
echo "                    Done                     "
echo -e "**************************************************************"