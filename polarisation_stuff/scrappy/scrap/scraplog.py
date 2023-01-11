from utils import configure_logger
from utils.genutils import make_out_dir

from scrap.arguments import parser 


# appedning the region size to the directory
# this is the default REGSIZE from scrappy
opts = parser().parse_args()
odir = opts.odir  
odir += "-s30" if opts.reg_size is None else f"-s{opts.reg_size}"
snitch = configure_logger(odir)