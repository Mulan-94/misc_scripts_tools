import os
import json
import yaml
import numpy as np
from argparse import ArgumentParser

from glob import glob
from bokeh.io import save, output_file
from bokeh.embed import json_item, components
from bokeh.layouts import row, gridplot, layout, column
from bokeh.models import (ColumnDataSource, Whisker, Line, Circle, Range1d,
        LinearAxis, DataRange1d, Panel, Tabs, Legend, LegendItem,
        HoverTool)
from bokeh.plotting import figure
from ipdb import set_trace

AMP_COLOUR = "#21317B"
IMAG_COLOUR = "#202221"
ANGLE_COLOUR = PHASE_COLOUR = "#00712a"
REAL_COLOUR = "#f26521"
CIRCLE_SIZE = 7
ALPHA = 1

class Pol():
    def __init__(self, i, q, u, i_err, q_err, u_err, freqs, **kwargs):
        self.i = i
        self.q = q
        self.u = u
        self.i_err = i_err
        self.q_err = q_err
        self.u_err = u_err
        self.freqs = freqs
        self.lpol = kwargs.get("lpol", None)

    @staticmethod
    def polarisation_angle(q, u):
        return 0.5 *np.arctan2(u, q)

    @staticmethod
    def polarisation_angle_error(p, q, u, q_err, u_err):
        "See A.11 Brentjens"
        err = (np.square(u*q_err) + np.square(q*u_err)) / (4 * p**4)
        return np.sqrt(err)

    @staticmethod
    def power_fn(q, u):
        return np.abs(q + 1j*u)
    
    @staticmethod
    def power_error(p, q, u, q_err, u_err):
        res = np.square((q/p)) * np.square(q_err) + np.square((u/p)) * np.square(u_err)
        return np.sqrt(res)

    @staticmethod
    def linear_polzn(q, u):
        return q + 1j*u
    
    @staticmethod
    def fractional_pol(i, q, u):
        fpol = Pol.power_fn(q, u)/i
        return fpol
    
    @staticmethod
    def fractional_pol_error(fpol, i, p, i_err, p_err):
        """
        fpol:
            Fractional polarisation
        i:
            Total intensity
        p:
            Complex linear polaraised power
        (i|p)_err
            Error in I and p
        """
        res = np.abs(fpol) * np.sqrt(np.square((p_err/p)) + np.square((i_err/i)))
        return res

    @staticmethod
    def lambda_squared(freqs):
        c = 3e8
        return (c/freqs)**2
    
    @staticmethod
    def amplitude(inp):
        return np.abs(inp)

    @staticmethod
    def real(inp):
        return np.real(inp)
    
    @staticmethod
    def imaginary(inp):
        return np.imag(inp)
    
    @staticmethod
    def phase(inp):
        return np.angle(inp, deg=True)



    def generate_data(self):
        datas = {}
        datas["power"] = Pol.power_fn(self.q, self.u)
        datas["fpol"] = Pol.fractional_pol(self.i, self.q, self.u)
        datas["lpol"] = self.lpol
        datas["angle"] = np.unwrap(Pol.polarisation_angle(self.q, self.u),
                                    period=np.pi, discont=np.pi/2)
        datas["angle"] = np.rad2deg(datas["angle"])

        datas["stokes"] = self.q + 1j*self.u
        datas["lambda_sq"] = Pol.lambda_squared(self.freqs)
        
        datas["angle_err"] = Pol.polarisation_angle_error(
                datas["power"], self.q, self.u, self.q_err, self.u_err)
        datas["power_err"] = Pol.power_error(
                datas["power"], self.q, self.u, self.q_err, self.u_err)
        datas["fpol_err"] = Pol.fractional_pol_error(
                datas["fpol"], self.i, datas["power"], self.i_err,
                datas["power_err"])

        datas["q_err"] = self.q_err
        datas["u_err"] = self.u_err
        return datas


def read_yaml(fname):
    with open(fname, "r") as fil:
        datu = yaml.safe_load(fil)
    return datu


def read_data(fname):
    with np.load(fname, allow_pickle=True) as data:
        # these frequencies are in Hz
        datas = {k: v for k, v in data.items()}
    return datas


def error_margins(base, y, err):
    """
    base:
        The x-axis
    y:
        The y-axis
    err:
        Error to be appended on the y-axis
    """
    ebars = Whisker(source=ColumnDataSource(data=dict(base=base, 
                    upper=y + err, lower=y - err)),
                    line_cap="round", line_color="red",
                    line_alpha=ALPHA, lower_head=None, upper_head=None,
                    line_width=0.5, base="base", upper="upper", lower="lower")
    return ebars


def make_plot(indata, meta_title, meta_data):
    """
    indata:
        dict containing the data
    meta_title
        title of the plot of the data being worked on
    meta_data
        Parameters concerning that plot
    """
    glyphs = {"Circle": Circle, "Line": Line}

    #which data to get from the data products
    mapping = {_: "lpol" for _ in ["lpol", "stokes"]}
    mapping.update({_: _ for _ in ["fdirty", "fclean", "rmtf"]})
    mapping["fpol"] = "fpol"
    mapping["angle"] = "angle"
    
    fig = figure(
        plot_width=800, plot_height=500, sizing_mode="stretch_both",
        # tooltips=[]
        )

    fig.axis.axis_label_text_font = "monospace"
    fig.axis.axis_label_text_font_style = "normal"
    fig.axis. axis_label_text_font_size = "15px"
    fig.yaxis.axis_label=meta_data["y_label"]
    fig.xaxis.axis_label=meta_data["x_label"]

    legend_items = []
    for yaxis, params in meta_data["items"].items():

        # there's a shared x for all plots, but each plot can define its x
        # give precedence to x axis in individual plots rather than the general one
        xaxis = params["x"] if "x" in params else meta_data["x"]
        # if xdata is specified, it implies that it needs to be processed
        if "xdata" in params:
            xdata = indata[params["xdata"]].copy()
        else:
            xdata = indata[meta_data["x"]]


        if "ydata" in params:
            ydata = indata[params["ydata"]].copy()
        else:
            ydata = indata[mapping[meta_title]].copy()
        
        data_src = {
            xaxis: getattr(Pol, xaxis)(xdata) if hasattr(Pol, xaxis) else xdata,
            yaxis: getattr(Pol, yaxis)(ydata) if hasattr(Pol, yaxis) else ydata
            }

        cds = ColumnDataSource(data=data_src)

        if meta_data["glyph"].lower() == "circle":
            specs = dict(size=CIRCLE_SIZE, fill_color=globals()[params["color"]]
            )
        else:
            specs = dict(line_width=3)

        glyph = glyphs.get(meta_data["glyph"])(
            x=xaxis, y=yaxis,
            line_color=globals()[params["color"]], 
            **specs
            )
        rend = fig.add_glyph(cds, glyph, visible=params["visible"])

        if "y_error" in params:
            ebars = error_margins(data_src[xaxis], data_src[yaxis], indata[params["y_error"]])
            ebars.name="ebar_y"
            ebars.js_link("visible", rend, "visible")
            rend.js_link("visible", ebars, "visible")
            fig.add_layout(ebars)

        if "x_error" in params:
            ebars = error_margins(data_src[yaxis], data_src[xaxis], indata[params["x_error"]])
            ebars.name="ebar_x"
            ebars.js_link("visible", rend, "visible")
            rend.js_link("visible", ebars, "visible")
            fig.add_layout(ebars)

        legend_items.append((params["label"], [rend]))
    
    tooltips = HoverTool(tooltips=[
        (f"{xaxis}, {yaxis}", f"(@{xaxis}, $y")])
    fig.add_tools(tooltips)
    
    fig.add_layout(Legend(
        items=legend_items, click_policy="hide", label_text_font_size="15px"))
    fig.y_range.only_visible = True
    return fig


def write_data(model, name, o_file):
    # o_path = os.path.join(os.environ['CYGSERVER'], "plots", o_file)
    o_path = os.path.join(".", o_file)
    # I.e. if the file was a json file. I was using this when I wanted to load 
    # the plot in a div
    if ".html" not in o_file:
        with open(o_path, "w") as fn:
            json.dump(json_item(model, name), fn)
    else:
        output_file(o_path, title=name)
        save(model, filename=o_path)


def arg_parser():
    parser = ArgumentParser(
        usage="""
Example

python plot_bk.py -id IQU-regions-mpc-20-circle-sel-0.05 IQU-regions-mpc-20-correct-0.05
=================+++++++++++++++++++++++===========
""")
    parser.add_argument("-id", type=str, nargs="+", dest="i_dirs",
        help="Input directories")
    parser.add_argument("--yaml", type=str, nargs="+", dest="yamls",
        default=["plots.yml"],
        help="Yaml file containing the plots to be plotted.")
    parser.add_argument("-od", "--output-dir", type=str, dest="odir",
        default=None,
        help="Where to dump outputs")
    return parser

if __name__ == "__main__":

    opts = arg_parser().parse_args()
    print("Starting bokeh plots")

    for y_file in opts.yamls:
        yaml_plots = read_yaml(y_file)

        for i_dir in opts.i_dirs:
            files = glob(f"{i_dir}/*npz")
            if len(files) == 0:
                print("No LOS files found, skipping.")
            # for each LOS, read its data and plot it
            for los in files:
                print(f"LOS: {los}")
                reg = os.path.splitext(os.path.basename(los))[0]
                depth_dir = glob(os.path.dirname(los) + "-depths*")[0]

                # read line of sight data wavelengths
                los_data = read_data(los)

                gets ={g.lower(): los_data[g] for g in 
                    ['I', 'Q', 'U', 'I_err', 'Q_err', 'U_err', 'freqs', "lpol"]}
                
                for key, value in gets.items():
                    gets[key] = np.ma.masked_array(data=value, mask=los_data["mask"]).compressed()

                grps = {"grp1": [], "grp2": []}

                pol_data = Pol(**gets).generate_data()
                
                # read los data for depths
                try:
                    pol_data.update(read_data(os.path.join(depth_dir, f"{reg}.npz")))
                except FileNotFoundError:
                    print(f"Depth File for {reg} not found")
                    continue

                for plot, plot_params in yaml_plots.items():
        
                    sub = Panel(child=make_plot(pol_data, plot, plot_params), title=plot_params["title"])
                    if plot in ["fpol", "angle", "stokes", "lpol"]:
                        grps["grp1"].append(sub)
                    else:
                        grps["grp2"].append(sub)

                outp = gridplot(children=[Tabs(tabs=grp) for _, grp in grps.items()],
                                ncols=len(grps) if grps["grp2"] else 1,
                                sizing_mode="stretch_both", toolbar_location="left")
                
                #change to .json if you want a json output
                if opts.odir is not None:
                    o_dir = opts.odir
                else:
                    o_dir = i_dir + "-plots"
                if not os.path.isdir(o_dir):
                    os.mkdir(o_dir)
                o_file =os.path.join(o_dir, reg.replace("_", "") + ".html")
                print(f"Writing {o_file}")

                write_data(model=outp, name=reg, o_file=o_file)
                print("Done")
        
