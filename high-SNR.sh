set -e

# # # echo -e "\n############################################################"
# # # echo "copy relevant channels' images to this folder for IQUV";
# # # echo -e "############################################################\n"
# # # for n in ${sel[@]};
# # # 	do
# # # 		cp ../*-$n-{I,Q,U}-*image* $imgs;
# # # 	done

echo -e "\n############################################################"
echo "Generate various interesting LoS above some certain threshold";
echo -e "############################################################\n"

export PYTHONPATH=$PYTHONPATH:$(readlink -f qu_pol/scrappy/)

#what to name the stuff: for the scrapper!
export regsize=3;
scout=$scout-hi-SNR;
thresh=800;

python qu_pol/scrappy/scrap/scrappy.py -rs $regsize -idir $imgs --threshold $thresh -odir $scout -ref-image i-mfs.fits -nri i-mfs.fits

scout=$scout-s$regsize;


# # # echo -e "\n############################################################"
# # # echo "Renaming regions to match their associated remaining numbers";
# # # echo -e "############################################################\n"
# # # cd $scout/los-data/
# # # regnum=1
# # # for n in $(ls * | sort -V);
# # #     do
# # #         echo "$n >> reg_$regnum.npz";
# # #         mv $n reg_$regnum.npz || true;
# # #         regnum=$(( regnum + 1 ));
# # #     done
# # # cd $OLDPWD;


 # # # echo -e "\n############################################################"
 # # # echo "Delete the copied image files";
 # # # echo -e "############################################################\n"
 # # # # rm ./*-[0-9][0-9][0-9][0-9]-*image.fits
 # # # rm -r $imgs


echo -e "\n############################################################"
echo "Edit the reg file in a way that it can be loaded into CARTA"
echo -e "############################################################\n"

cp $scout/regions/*valid.reg $scout/regions/beacons.reg
sed -i "s/text=.*//g" $scout/regions/beacons.reg


echo -e "\n############################################################"
echo "Perfrom RM synthesis for various lines of sight generated from previous step and plot the output";
echo "For pictor I set the maximum depth to 400, depth step 1 looks smoothest, niter from 500-1000 looks similar"
echo -e "############################################################\n"
python qu_pol/scrappy/rmsynthesis/rm_synthesis.py -id $scout/los-data -od $scout/los-rm-data -md 400 --depth-step 1


echo -e "\n############################################################"
echo "Generate interactive plots for the various LoS";
echo -e "############################################################\n"
python qu_pol/scrappy/bokeh/plot_bk.py -id $scout/los-data --yaml qu_pol/bokeh/plots.yml -od $scout/bokeh-plots
