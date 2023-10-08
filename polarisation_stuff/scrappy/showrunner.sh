#check on the use of masks
# # masks used are:
# 1. full mask
# 2. pica-region
# 3. hostpot-regions
# 4. lobe-regions
# 5. 

set -e

welcome(){
    help="
| ===========================================================================\n|\n\
| \t    Welcome to showrunner \n\
| \t'Together we run the show' ~ Kat DeLuna :) \n|\n\
| =========================================================================== \n|\n\
| Software used in this script requires Python > 3.8 \n\
| Please ensure they are installed \n\
| Recommend running this in a virtual environment \n|\n\
| --------------------------------------------------------------------------- \n\
| \tInstructions \n\
| \t1) Running everything at once  \n\
| ---------------------------------------------------------------------------\n\
\t1. Place this script in a sub-dir where your channelised images are \n\
  \ti.e. the channelised images should be at '../' with respect to where\n\
     \tthis script is \n\
\t2. Make script executable as: 'chmod +x showrunner.sh' \n\
\t3. Run the script as follows: \n\
\t\t ./run_most_rm_things location/of/mask/dir \n\
\te.g \n \
\t\t ./run_most_rm_things.sh /home/andati/pica/reduction/experiments/emancipation/masks-572\n\n\
| ---------------------------------------------------------------------------\n\
"

    echo -e $help

}

installRequiredSoftware(){
    pip install spimple \
        Owlcat \
    pip install -e ~/git_repos/misc_scripts_n_tools/qu_pol/scrappy/
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

    if [[ -z $mask_dir ]]
    then
        export mask_dir=masks
        mkdir -p $mask_dir
        echo -e "Created mask directory: $mask_dir"
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



selectGoodChannels(){
    
    # 1. Create my channel selections here
    if [[ ! -f selected-channels.txt ]]
    then
    	echo -e "\n############################################################"
        echo "Autoselect some valid channels. Should be stored in a file called selected-channels"
        echo -e "############################################################\n"
        sc-beam-plot --output ./ --prefix 00 --threshold 0.5 --auto-select ../
    fi

    mapfile -t sel < selected-channels.txt; export sel;

    
    # 2. copy those selected channels' images to some other directory
    echo -e "\n############################################################"
    echo "copy relevant channels' images to this folder for IQUV"
    echo -e "############################################################\n"
    for n in ${sel[@]}
    	do
    		cp ../*-$n-{I,Q,U}-*image* $imgs
    	done

    
    # 3. generate frequency file for the selected images if its not there already
    if [[ ! -f frequencies.txt ]]
    then
        echo -e "\n############################################################"
        echo "write out the selected freqs into a single file for easier work. This is for use in the RM synth for wavelength":
        echo -e "############################################################\n"
        for im in $imgs/*-[0-9][0-9][0-9][0-9]*-I-image.fits
            do
                # ensure that this is being appended Lexy !!!!!!
                fitsheader -k CRVAL3 $im |  grep -i CRVAL3 >> frequencies.txt
            done


        echo -e "\n############################################################"
        echo "Cleanup freqs file by replacing CRVAL3 = and all spaces after it with emptiness"
        echo -e "############################################################\n"
        sed -i "s/CRVAL3  =\ \+//g" frequencies.txt
    fi


    
    # 4. save the names of the selected images in a file
    echo -e "\n############################################################"
    echo "Save the names of the selected images. Simpleton workaround for using cubes in the scap.py script :("
    echo -e "############################################################\n"
    ls $imgs/*-[0-9][0-9][0-9][0-9]*-image.fits > selected-freq-images.txt

    #Replacing all begins of strings here with ../
    sed -i 's/^/\.\.\//g' selected-freq-images.txt

    return 0
}


stackAllImages(){
    for s in $stokes
    do
        echo "make cubes from ALL the output images: ${s^^}"
        
        images=$(ls -v ../*-[0-9][0-9][0-9][0-9]-$s-image.fits)
        fitstool.py --stack=$orig_cubes/${s,,}-cube.fits:FREQ $(echo $images)

        if [[ ${s,,} = "i" ]]
        then
            images=$(ls -v ../*-[0-9][0-9][0-9][0-9]-$s-model.fits)
            fitstool.py --stack=$orig_cubes/${s,,}-model-cube.fits:FREQ $(echo $images)
            images=$(ls -v ../*-[0-9][0-9][0-9][0-9]-$s-residual.fits)
            fitstool.py --stack=$orig_cubes/${s,,}-residual-cube.fits:FREQ $(echo $images)
        fi
    done

    return 0
}


selectedChannels_Stack(){
    # optional argument syntax: variable=${read_arg_in_position:-default_value}
    local indir=${1:-$imgs}
    local outdir=${2:-$sel_cubes}

    # print function information
    echo -e "############################################################\n"
    echo "Running function: $FUNCNAME($@)"
    echo -e "############################################################\n"
    
    for s in $stokes
        do
            echo "Make the selection cubes: ${s^^}"
            images=$(ls -v $indir/*-[0-9][0-9][0-9][0-9]-$s-image.fits)
            fitstool.py --stack=$outdir/${s,,}-image-cube.fits:FREQ $(echo $images)
            echo "Stored at: $outdir"
        done
    return 0
}


selectedChannels_CubeConvolve(){

    mapfile -t beam_dims < beam-dims.txt
    
    echo -e "\n############################################################"
    echo "Convolve the cubes to the same resolution"
    echo -e "############################################################\n"
    for s in $stokes
        do        
            spimple-imconv -image $sel_cubes/${s,,}-image-cube.fits -o $conv_cubes/${s,,}-conv-image-cube -pp ${beam_dims[@]}
        done


    echo -e "\n############################################################"
    echo "Renaming output file from spimple because the naming here is weird"
    echo -e "############################################################\n"
    rename.ul -- ".convolved.fits" ".fits" $conv_cubes/* || true


    echo -e "\n############################################################"
    echo "Just check if the beam sizes are the same"
    echo -e "############################################################\n"
    # sc-beam-plot --output ./ --threshold 0.5  $imgs

    return 0

}



selectedChannels_SinglesConvolve(){

    if [[ ! -f beam-dims.txt ]]
    then
        # read beam dimensions of the first available channel
        fitsheader $imgs/$(ls -v relevant-images/ | head -1) | grep -i "bmaj \|bmin \|bpa " > beam-dims.txt
        sed -i "s/.*=\s*//g" beam-dims.txt

    fi
    # beam_dims=$(python -c "import numpy as np; cat = np.loadtxt('beam-dims.txt'); print(*cat)")
    mapfile -t beam_dims < beam-dims.txt

    convim=$imgs-conv
    mkdir -p $convim


    echo "Convolving channelised images individually"
    for im in $(ls -v $imgs/*.fits)
    do
        spimple-imconv -image $im -pp ${beam_dims[@]} -o $convim/$(basename $im)
    done

    rm $convim/*.clean_psf*
    rename.ul ".convolved.fits" "" $convim/*

    # stack them
    selectedChannels_Stack $convim $sel_cubes


    echo -e "\n############################################################"
    echo "Just check if the beam sizes are the same"
    echo -e "############################################################\n"
    sc-beam-plot --output ./ --prefix 01 $imgs

    return 0
}


copyMfsImages(){
    echo -e "\n############################################################"
    echo "copy I MFS image reference image here"
    echo -e "############################################################\n"
    
    cp ../*MFS-I-image.fits i-mfs.fits
    cp ../*MFS-Q-image.fits q-mfs.fits
    cp ../*MFS-U-image.fits u-mfs.fits

}


generateSpiMap(){
    # Doing some SPI maps

    if [[ ! -f wsums.txt ]]
    then
        echo -e "\n############################################################"
        echo "Get their wsums and store"
        echo -e "############################################################\n" 

        # Get wsums for the selected images with commas
        fitsheader *I-model.fits | grep -i wsum | sed s"/WSCVWSUM=\s*//g" > wsums.txt
    fi

    types=("residual", "model")
    for im in ${types[@]}
        do  
            images=$(ls -v ../*-[0-9][0-9][0-9][0-9]-I-$im.fits)
            fitstool.py --stack=$sel_cubes/${s,,}-cube.fits:FREQ $(echo $images)
        done


    # echo "Rename the convolved images"
    # # Adding || tru so that the error here does not fail the entire program
    # # see: https://stackoverflow.com/questions/11231937/bash-ignoring-error-for-a-particular-command
    # rename.ul -- ".convolved.fits" ".fits" $conv_cubes/* || true


    echo -e "\n############################################################"
    echo "Do the SPI fitting"
    echo -e "############################################################\n" 

    echo "Normalize the wsums by the largest values"

    # # Doing this with a quick python script because,
    # # well I can :) and store in this variable
    wsums=$(python -c "import numpy as np; wsums = np.loadtxt('wsums.txt'); wsums = np.round(wsums/wsums.max(), 4); print(*wsums)")
    maparray -t freqs < frequencies.txt

    # cw - channel weights, th-rms threshold factor, 
    # acr - add conv residuals, bm -  beam model
    spimple-spifit -model $sel_cubes/i-models.fits \
        -residual $sel_cubes/i-residuals.fits -o $spis/spi-map -th 10 \
        -nthreads 32 -pb-min 0.15 -cw $wsums -acr -bm JimBeam -band l \
        --products aeikb -cf $freqs


    return 0
}



generateRmMap(){
    # arg1: str
    #   name: prefix
    #   Prefix prepended to the outputs from this script. 
    #   May include the dump dir. Default is 'initial'.
    
    # local prefix=${1:-"$prods/initial"}
    # local mask=${2:-"$mask_dir/true_mask.fits"}
    local args=${1:-"-h"}

    # print function information
    echo "Running function: $FUNCNAME($@)"

    echo -e "\n############################################################"
    echo "Do some RM maps, fpol maps and other maps for each pixel"
    echo "Using my mask here, Don't know where yours is but if this step \
        fails, check on that"
    echo "Default maximum depth and number of iterations same as that of previous"
    echo -e "############################################################\n"
    sc-rmmap $(echo $args)

    return 0
}


makeMasks(){
    # Automatically create masks
    # arg1: str
    #     Other arguments to be passed to simple mask

    local args=${1:="-h"}


    echo -e "\n############################################################"
    echo "Running function: $FUNCNAME($args)"
    echo -e "\n############################################################"

    # Make mask for my source
    sc-houdini $(echo $args)
    
    return 0
}



runScrappy(){
    # Run the scrappy package
    # -----------------------
    # arg1: int
    #     name: thresh
    #     Threshold to be used. Default 800
    # arg2:
    #     name: regsize
    #     Region size to be used. Default 3
    # arg3: str
    #     name: scout
    #     Name of output directory for use. Default scrapy-out
    # arg4: str
    #     name: mask
    #     Which mask to use for this run
    # arg5: bool
    #     name: bokeh
    #     Whether or not to Generate bokeh plots


    # for scrappy
    # I change region size from 5pix to 3 pixels.
    # Set minimum noise * threshold above which to generate regions

    local thresh=${1:-10}
    local regsize=${2:-3}
    local scout=${3:-"$prods/scrappy-out"}
    local mask=${4:-"$mask_dir/true_mask.fits"}
    local bokeh=${5:-false}
    local others=${6:-""}

    # print function information
    echo "Running function: $FUNCNAME($@)";
    
    sc-los -rs $regsize -idir $imgs \
        --threshold $thresh -odir $scout -ref-image i-mfs.fits -nri i-mfs.fits \
        -m $mask $(echo $others)

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
    sc-losrm -id $scout/los-data \
        -od $scout/los-rm-data -md 400 --depth-step 1

    if [[ bokeh ]]
    then
        echo -e "\n############################################################"
        echo "Generate interactive plots for the various LoS"
        echo -e "############################################################\n"
        sc-bokehplot -id $scout/los-data \
          --yaml qu_pol/bokeh/plots.yml -od $scout/bokeh-plots
    fi

    return 0
}


main(){
    installRequiredSoftware
    
    initialiseEnvVarsForThisScript
    makeDirs
    selectGoodChannels
    stackAllImages

    selectedChannels_Stack
    selectedChannels_CubeConvolve
    # selectedChannels_SinglesConvolve

    copyMfsImages
    
    makeMasks "
        i-mfs.fits
        -o $mask_dir/true_mask.fits
        -above 4e-3
        -rb $HOME/pica/reduction/experiments/emancipation/masks/important_regions/pica_region-for-mask.reg
        "

    # # pass all the arguments between the quotes to this function
    generateRmMap "
        -q $conv_cubes/q-conv-image-cube.fits
        -u $conv_cubes/u-conv-image-cube.fits
        -i $conv_cubes/i-conv-image-cube.fits -ncore 45 -md 200
        -o $prods/initial -mask $mask_dir/true_mask.fits -f frequencies.txt "

    runScrappy 50 3 $prods/scrap-outputs $mask_dir/true_mask.fits true

    # generateSpiMap

    return 0
}


if [[ $1 = '-h' ]]
then
    welcome
else
    # Run this main function
    main $1

fi
