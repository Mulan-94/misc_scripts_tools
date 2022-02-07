#! /bin/python3
import argparse
import os
import logging
import numpy as np
import operator
from functools import reduce
from itertools import chain, zip_longest
from astropy.io import fits
from ipdb import set_trace

logging.basicConfig(format="%(name)s: %(levelname)s: %(message)s")
snitch = logging.getLogger("sade")
snitch.setLevel(logging.INFO)


def get_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""FITS mazematics :)\n==================\nFor example
        python fits_maths.py -ims 64-chans/*00*Q*image* -o image_sum.fits -ops "+" """
        )

    parser.add_argument("-overwrite", action="store_true", dest="overwrite",
        help="""Whether or not to overwrite the output images""")

    reqs = parser.add_argument_group("Required arguments")

    reqs.add_argument("-ops", "--operators", dest="operators", type=str,
        nargs="+", metavar="", required=True,
        help="""space separated list of operators IN THE ORDER in which you
        want them to occur. The available ones are + - / * ^ for sum,
        subtraction, division, multiplication and power respectively. Accompany
        power with the actual power. e.g. ^2 is square, ^0.5 is sqrt and so
        forth.""")

    reqs.add_argument("-ims", "--images", dest="images", required=True, metavar="",
        nargs="+", type=str, action="append",
        help="""Groups of images to plot. Do space separated list per group. If
        there are multiple groups, specify this argument multiple times. i.e
        -ims a b c d -ims z x y spaces separating each group. e.g if we want
        image a+b, c+d, e+f etc, set this as a,b  c,d  e,f""")
    
    reqs.add_argument("-o", "--output", dest="outputs", metavar="", nargs="+",
        default=[None], help="""Name of the output fits file.""")
    return parser


def read_input_image_header(im_name):
    """
    Parameters
    ----------
    im_name: :obj:`string`
        Image name

    Output
    ------
    info: dict
        Dictionary containing center frequency, frequency delta and image wsum
    """
    snitch.info(f"Reading image: {im_name} header")
    with fits.open(im_name, readonly=True) as hdu_list:
        # print(f"There are:{len(hdu_list)} HDUs in this image")
        data = hdu_list[0].data.squeeze()
    return data


def gen_fits_file_from_template(template_fits, new_data, out_fits, overwrite=False):
    with fits.open(template_fits, mode="readonly") as temp_hdu_list:
        temp_hdu, = temp_hdu_list
        #update with the new data
        if temp_hdu.data.ndim == 4:
            temp_hdu.data[0,0] = new_data
        elif temp_hdu.data.ndim == 3:
            temp_hdu.data[0] = new_data
        elif temp_hdu.data.ndim == 2:
            temp_hdu.data = new_data
        temp_hdu_list.writeto(out_fits, overwrite=overwrite)
    
    snitch.info("=================================")
    snitch.info(f"New file written to: {out_fits}")
    snitch.info("=================================")
    return


def operate(inp, op):
    ops = "+ - / * ^".split()
    funcs = [
        operator.add, operator.sub, operator.truediv,
        operator.mul, operator.pow
        ]
    op_func = {k:v for k, v in zip(ops, funcs)}

    if "^" in op:
        # get the power
        inp = op_func["^"](np.array(inp), float(op.split("^")[-1]))
    else:
        inp = reduce(op_func[op], inp)

    return inp

def main():
    ps = get_arguments().parse_args()

    # form image name input groups
    inp_groups =  {}
    for idx, imag in enumerate(ps.images):
        inp_groups[idx] = ps.images[idx]
    
    # ie the numbers to be operated on
    if len(inp_groups)>1:
        """
        nth row will represent the nth element in each group
        a,b,c   d,e,f   g,h,i 
        is translated to ->
            a b c
            d e f
            g h i
        
        places things to be operated together in one group
        this is the same case as when there's only a single group
        """
        nu_array = np.array(list(inp_groups.values())).T
      
        inp_groups = {r: list(nu_array[r, :]) for r in range(nu_array.shape[0])}


    if len(ps.outputs)>1:
        outputs = ps.outputs
    elif len(ps.outputs)==1:
        outputs = ps.outputs * len(inp_groups)
    else:
        outputs = ["group_out"] * len(inp_groups)

    for (group_idx, group), out_name in zip_longest(inp_groups.items(), outputs):
        # read the images and get their data
        # store the data in an ordered list
        inp_groups[group_idx] = [read_input_image_header(grp) for grp in group]

        # operate on the data
        # return the output
        # number of oupts should be the same as number of keys
        for op in ps.operators[0].split():
            inp_groups[group_idx] = operate(inp_groups[group_idx], op)

        # write the output to a new output fits file
        # close
        out_name =  f"{out_name}_{group_idx}.fits"

        gen_fits_file_from_template(
            group[0],  inp_groups[group_idx], out_name,
            overwrite=ps.overwrite)

    snitch.info("Done calculating")


if __name__ == "__main__":
    main()