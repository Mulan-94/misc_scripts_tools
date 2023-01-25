#!/bin/bash
set -e


checkScriptsExist(){
    echo -e "\n############################################################"
    echo -e "checking if the required scripts exist"
    echo -e "############################################################\n"

    # REference: https://devconnected.com/how-to-check-if-file-or-directory-exists-in-bash/

    if [[ ! -f plotting_bmaj_bmin.py ]]
    then
        echo "plotting_bmaj_bmin.py FILE does not exist. Creating"
        ln -s $HOME/git_repos/misc_scripts_n_tools/fits_related/plotting_bmaj_bmin.py
    fi

    if [[ ! -d qu_pol ]]
    then
        echo "qu_pol DIR does not exist. Creating"
        ln -s $HOME/git_repos/misc_scripts_n_tools/qu_pol
    fi

    
    echo -e "\n############################################################"
    echo "get the selected channels. Should be stored in a file called selected-channels"
    echo -e "############################################################\n"

    # copy my channel selections here
    if [[ ! -f selected-channels.txt ]]
    then
    	cp ../selected-channels.txt .
    fi

    return 0
}


initialiseEnvVarsForThisScript(){
    echo -e "\n############################################################"
    echo "We initialise some of the environment variables in env-vars"
    echo -e "\n############################################################"
    if [[ ! -f env-vars ]]
    then
    	echo "env-vars does not exist. Creating"
    	ln -s $HOME/git_repos/misc_scripts_n_tools/env-vars
    fi

    source env-vars

    echo -e "Setting up variables, and the selection of channels"

    export stokes="I Q U"
    export sel=($(echo $(cat selected-channels.txt)))

    return 0
}


initialiseMaskDir(){
    echo -e "\n############################################################"
    echo -e "Reading the mask dir"
    echo -e "############################################################\n"
    if [[ $1 = "" ]];
    then
    	echo "Mask dir not specified, I will use the default one";
    	export mask_dir=$HOME/pica/reduction/experiments/emancipation/masks;
    	echo "Mask dir:  $mask_dir";
    else
    	echo "Mask dir:  $1";
    	export mask_dir=$1;
    fi

    return 0 
}


makeDirs(){
    echo -e "\n############################################################"
    echo "Make these directories"
    echo -e "############################################################\n"
    mkdir -p $orig_cubes $sel_cubes $conv_cubes $plots $prods $spis $imgs
    return 0
}


copyAndStackRequiredImages(){

    echo -e "\n############################################################"
    echo "Getting staarted"
    echo -e "############################################################\n"
    for s in $stokes
    	do
    		echo "make cubes from ALL the output images: ${s^^}"
            
    		images=$(ls ../*-[0-9][0-9][0-9][0-9]-$s-image.fits)
    		fitstool.py --stack=$orig_cubes/${s,,}-cube.fits:FREQ $(echo $images)

    		if [[ ${s,,} = "i" ]]
            then
    			images=$(ls ../*-[0-9][0-9][0-9][0-9]-$s-model.fits)
    			fitstool.py --stack=$orig_cubes/${s,,}-model-cube.fits:FREQ $(echo $images)
    			images=$(ls ../*-[0-9][0-9][0-9][0-9]-$s-residual.fits)
    			fitstool.py --stack=$orig_cubes/${s,,}-residual-cube.fits:FREQ $(echo $images)
    		fi
    	done

    echo -e "\n############################################################"
    echo "Plot the beams to identify which should be flagged"
    echo -e "############################################################\n"
    python plotting_bmaj_bmin.py -c $orig_cubes/i-cube.fits -o $plots/orig-bmaj-bmin-i


    echo -e "\n############################################################"
    echo "copy relevant channels' images to this folder for IQUV"
    echo -e "############################################################\n"
    for n in ${sel[@]}
    	do
    		cp ../*-$n-{I,Q,U}-*image* $imgs
    	done


    echo -e "\n############################################################"
    echo "copy I MFS image reference image here"
    echo -e "############################################################\n"
    cp ../*MFS-I-image.fits i-mfs.fits
    cp ../*MFS-Q-image.fits q-mfs.fits
    cp ../*MFS-U-image.fits u-mfs.fits


    return 0
}


generateUsefulFiles(){
    # generate files containing
    # 1. selected images' names
    # 2. Frequencies of those images
    echo -e "\n############################################################"
    echo "Save the names of the selected images. Simpleton workaround for using cubes in the scap.py script :("
    echo -e "############################################################\n"
    ls $imgs/*-[0-9][0-9][0-9][0-9]*-image.fits > selected-freq-images.txt

    #Replacing all begins of strings here with ../
    sed -i 's/^/\.\.\//g' selected-freq-images.txt


    echo -e "\n############################################################"
    echo "write out the selected freqs into a single file for easier work. This is for use in the RM synth for wavelength":
    echo -e "############################################################\n"
    for im in $imgs/*-[0-9][0-9][0-9][0-9]*-I-image.fits
    	do
    		fitsheader -k CRVAL3 $im |  grep -i CRVAL3 >> frequencies.txt
    	done


    echo -e "\n############################################################"
    echo "Cleanup freqs file by replacing CRVAL3 = and all spaces after it with emptiness"
    echo -e "############################################################\n"
    sed -i "s/CRVAL3  =\ \+//g" frequencies.txt

    return 0
}


convolveCubesToSameResolution(){
    echo -e "\n############################################################"
    echo "Selections steps"
    echo -e "############################################################\n"
    for s in $stokes
        do
            echo "Make the selection cubes: ${s^^}"
            images=$(ls -v $imgs/*-[0-9][0-9][0-9][0-9]-$s-image.fits)
            fitstool.py --stack=$sel_cubes/${s,,}-image-cube.fits:FREQ $(echo $images)
            
            echo "Convolve the cubes to the same resolution"
            spimple-imconv -image $sel_cubes/${s,,}-image-cube.fits -o $conv_cubes/${s,,}-conv-image-cube 
        done


    echo -e "\n############################################################"
    echo "Plot the beams of selected channels to see if they make sense"
    echo -e "############################################################\n"
    python plotting_bmaj_bmin.py -c $sel_cubes/i-image-cube.fits -o $plots/selected-bmaj-bmin-i



    echo -e "\n############################################################"
    echo "Renaming output file from spimple because the naming here is weird"
    echo -e "############################################################\n"
    rename.ul -- ".convolved.fits" ".fits" $conv_cubes/* 



    echo -e "\n############################################################"
    echo "Just check if the beam sizes are the same"
    echo -e "############################################################\n"
    python plotting_bmaj_bmin.py -c $conv_cubes/*-conv-image-cube.fits -o $plots/conv-bmamin

    return 0
}


convolveSingleImagesForScrappy(){
    #requirement: use images with same conv?
    mkdir -p convim
    for im in $(ls -v $imgs/*.fits)
    do
        echo "Convolving channelised images manually for scrappy"
        spimple-imconv -image $im -pp 5.20e-03 2.86e-03 8.56e+01 -o convim/$(basename $im)
    done
    rm convim/*.clean_psf*
    rename.ul ".convolved.fits" "" convim/*
    rm -r $imgs
    mv convim $imgs

    return 0
}


runScrappy(){
    # Run the scrappy package
    # -----------------------
    # arg1: int
    #     Threshold to be used. Default 800
    # arg2:
    #     Region size to be used. Default 3
    # arg3: str
    #     Name of output directory for use. Default scrapy-out
    # arg4: bool
    #   Generate bokeh plots

    local thresh=$([ ! -z "$1" ] && echo $1 || echo 10)
    local regsize=$([ ! -z "$2" ] && echo $2 || echo 3)
    local scout=$([ ! -z "$3" ] && echo $3 || echo "scrappy-out")
    local bokeh=$([ ! -z "$4" ] && echo $4 || echo false)
    
    export PYTHONPATH=$PYTHONPATH:$(readlink -f qu_pol/scrappy/)

    python qu_pol/scrappy/scrap/scrappy.py -rs $regsize -idir $imgs \
        --threshold $thresh -odir $scout -ref-image i-mfs.fits -nri i-mfs.fits

    scout=$scout-s$regsize

    echo -e "\n############################################################"
    echo "Edit the reg file in a way that it can be loaded into CARTA"
    echo -e "############################################################\n"

    cp $scout/regions/*valid.reg $scout/regions/beacons.reg
    sed -i "s/text=.*//g" $scout/regions/beacons.reg


    echo -e "\n############################################################"
    echo "Perfrom RM synthesis for various lines of sight generated from \
        previous step and plot the output"
    echo "For pictor I set the maximum depth to 400, depth step 1 looks \
        smoothest, niter from 500-1000 looks similar"
    echo -e "############################################################\n"
    python qu_pol/scrappy/rmsynthesis/rm_synthesis.py -id $scout/los-data \
        -od $scout/los-rm-data -md 400 --depth-step 1


    echo -e "\n############################################################"
    echo "Generate interactive plots for the various LoS"
    echo -e "############################################################\n"
    # python qu_pol/scrappy/bokeh/plot_bk.py -id $scout/los-data \
    #   --yaml qu_pol/bokeh/plots.yml -od $scout/bokeh-plots

    return 0
}


generateRmMap(){
    echo -e "\n############################################################"
    echo "Do some RM maps, fpol maps and other maps for each pixel"
    echo "Using my mask here, Don't know where yours is but if this step \
        fails, check on that"
    echo "Default maximum depth and number of iterations same as that of previous"
    echo -e "############################################################\n"
    python qu_pol/pica_rm-x2.py -q $conv_cubes/q-conv-image-cube.fits \
        -u $conv_cubes/u-conv-image-cube.fits \
        -i $conv_cubes/i-conv-image-cube.fits -ncore 120 \
        -o $prods/initial -mask $mask_dir/true_mask.fits -f frequencies.txt

    return 0
}


generateSpiMap(){
    # Doing some SPI maps

    echo -e "\n############################################################"
    echo "Copy I selected models and residuals"
    echo -e "############################################################\n" 

    # For selection with LS using or patterns
    # https://unix.stackexchange.com/questions/50220/using-or-patterns-in-shell-wildcards


    for n in ${sel[@]}
    	do
    		cp ../*-$n-I-{residual,model}* .
    	done


    echo -e "\n############################################################"
    echo "Get their wsums and store"
    echo -e "############################################################\n" 

    # Get wsums for the selected images with commas
    fitsheader *I-model.fits | grep -i wsum | sed s"/WSCVWSUM=\s*//g" > wsums.txt



    echo -e "\n############################################################"
    echo "stack I residuals and models"
    echo -e "############################################################\n"


    images=$(ls *-[0-9][0-9][0-9][0-9]-*residual.fits)
    fitstool.py --stack $sel_cubes/i-residuals.fits:FREQ $(echo $images)

    images=$(ls *-[0-9][0-9][0-9][0-9]-*model.fits)
    fitstool.py --stack $sel_cubes/i-models.fits:FREQ $(echo $images)

    echo "Rename the convolved images"
    # Adding || tru so that the error here does not fail the entire program
    # see: https://stackoverflow.com/questions/11231937/bash-ignoring-error-for-a-particular-command
    rename.ul -- ".convolved.fits" ".fits" $conv_cubes/* || true


    echo -e "\n############################################################"
    echo "Delete the copied models and residuals"
    echo -e "############################################################\n" 
    rm ./*-{model,residual}.fits 



    echo -e "\n############################################################"
    echo "Do the SPI fitting"
    echo -e "############################################################\n" 

    echo "Normalize the wsums by the largest values"

    # # Doing this with a quick python script because,
    # # wll I can :) and store in this variable
    wsums=$(python -c "import numpy as np; wsums = np.loadtxt('wsums.txt'); wsums = np.round(wsums/wsums.max(), 4); print(*wsums)")

    freqs=$(cat frequencies.txt)

    # cw - channel weights, th-rms threshold factor, 
    # acr - add conv residuals, bm -  beam model
    spimple-spifit -model $sel_cubes/i-models.fits \
        -residual $sel_cubes/i-residuals.fits -o $spis/spi-map -th 10 \
        -nthreads 32 -pb-min 0.15 -cw $wsums -acr -bm JimBeam -band l \
        --products aeikb -cf $freqs


    return 0
}


makeLobeCubes(){
    echo -e "\n############################################################"
    echo "Generate images for the respective lobes"
    echo -e "############################################################\n"


    fitstool.py --prod $conv_cubes/i-conv-image-cube.fits $mask_dir/east-lobe.fits \
        -o ./east-lobe-cube.fits
    fitstool.py --prod $conv_cubes/i-conv-image-cube.fits $mask_dir/west-lobe.fits \
        -o ./west-lobe-cube.fits

    return 0
}


makePlotsforPaper(){
    echo -e "\n############################################################"
    echo "Make some other plots for paper"
    echo -e "############################################################\n" 

    python qu_pol/test_paper_plots.py
}


welcome(){
    echo -e "******************************************************"
    echo -e "   Welcome to MOST THINgs RM with Lexy"
    echo -e "******************************************************"
    echo -e "\n Place this script where the products will be placed"
    echo -e "This script requires some preliminary setup as follows"
    echo -e "In this same directory"
    echo -e "\n  1. Access to qu_pol from misc_tools_n_scripts/qu_pol"
    echo -e "  2. Access to plotting_bmaj_bmin.py from misc_tools_n_scripts/fits_related/"
    echo -e "  3. Images being worked on should be in this dirs' parent directory '../'\n"
    echo -e "  NOTE: PLEASE SUPPLY WHERE TO FIND YOUR MASSKSKSSSS!!!!!"
    echo -e "******************************************************"


    echo "To be run as:"
    echo -e "\t ./run_most_rm_things location/of/mask/dir"
    echo "e.g"
    echo -e "\t ./run_most_rm_things.sh /home/andati/pica/reduction/experiments\
        /emancipation/masks-572"
    echo -e "\t ./run_most_rm_things.sh /home/andati/pica/reduction/experiments\
        /emancipation/masks"

}


main(){
    # This function takes in the mask dir which is passed appropriately

    checkScriptsExist
    initialiseEnvVarsForThisScript
    
    # This function takes in one argument
    initialiseMaskDir $1
    
    makeDirs
    
    copyAndStackRequiredImages
    generateUsefulFiles
    convolveCubesToSameResolution
    convolveSingleImagesForScrappy

    # runScrappy(thresh, regsize, outname, bokeh_plots?)
    # running the Normal stuff
    runScrappy 10 3 $scout false

    generateRmMap
    generateSpiMap
    makeLobeCubes
    makePlotsforPaper

    return 0
}


testing(){

    initialiseEnvVarsForThisScript

    # This function takes in one argument
    initialiseMaskDir $1

    # # running the high snr stuff
    # # runScrappy 800 3 $scout-hi-snr false

    # makeLobeCubes

    makeLobeCubes
    makePlotsforPaper

    return 0
}


if [[ $1 = '-h' ]]
then
    welcome
else
    # Run this main function
    main $1

    # testing $1
fi
