from utils import configure_logger
from utils.genutils import make_out_dir

from scrap.arguments import parser 


# appedning the region size to the directory
# this is the default REGSIZE from scrappy
opts = parser().parse_args()
odir = opts.odir

if opts.debug:
    snitch = configure_logger(odir, level=0)
else:
    snitch = configure_logger(odir)

for key, value in opts._get_kwargs():
    snitch.warning(f"{key:17}:  {value}")