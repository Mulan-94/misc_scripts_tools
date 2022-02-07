#!/bin/bash/python3.7

import argparse
import Tigger

from dataclasses import dataclass
from itertools import zip_longest
from math import ceil

class RegionProperties:
    specs = []
    pre_specs = []
    
    def __init__(self):
        pass
    
    def include(self, on=1):
        # ensure that this is always last thing ong the list
        self.pre_specs.insert(-1, "+" if on else "-")

    def coord_sys(self, coord="physical"):
        coords = [
            'image','linear','fk4','b1950','fk5','j2000','galactic',
            'ecliptic','icrs','physical','amplifier','detector','wcs',
            'wcsa-wcsz'
            ]
        if coord in coords:
            self.pre_specs.insert(0, f"{coord};")

    def text(self, msg=None, tangle=0):
        if msg is not None:
            self.specs.append(f"text={msg} textangle={tangle}")
    
    def color(self, col=None):
        if col is not None:
            self.specs.append(f"color={col}")

    def dash_list(self, ):
        pass

    def dash(self, on=1):
        if on:
            self.specs.append(f"dash={on}")

    def width(self, size=1):
        self.specs.append(f"width={size}")

    def font(self, family="times", size=10, weight="bold", italic=False):
        out = f'font="{family} {size} {weight}'
        self.specs.append(out + ' italic"' if italic else out +'"')

    def selectable(self, on=1):
        if on:
            self.specs.append(f"select={on}")

    def highlightable(self, on=1):
        if on:
            self.specs.append(f"highlite={on}")

    def fixed(self, on=1):
        if on:
            self.specs.append(f"fixed={on}")
    
    def editable(self, on=1):
        if on:
            self.specs.append(f"edit={on}")

    def rotatable(self, on=1):
        if on:
            self.specs.append(f"rotate={on}")

    def deletable(self, on=1):
        if on:
            self.specs.append(f"delete={on}")

    def zindex(self, idx="source"):
        self.specs.append(f"{idx}")

    def tag(self, *args):
        if len(args)>0:
            self.specs.append(" ".join([f"tag={{ {arg} }}" for arg in args]))

    def update_properties(self, kwargs):
        """
        kwargs needs to be a dictionary containing keys for the functions 
        and a list, tuple, iterable containing arguments for the specified 
        function
        """
        for func, params in kwargs.items():
            print(f"Updating {func} property")
            # ensure that the params are in list form
            params = [params] if not isinstance(params, list) else params
            eval("self."+func)(*params)
        print("Update finished")
        return

    @property
    def comments(self):
        return "{}{} # {}".format(' '.join(self.pre_specs),
                                  self.__repr__().lower(), ' '.join(self.specs))


class Circle(RegionProperties):
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.specs = []
        self.pre_specs = []

    def __repr__(self):
        return "{}({},{},{})".format(self.__class__.__name__, 
                                       self.x, self.y, self.radius)

    def fill(self, on):
        if on:
            self.specs.append(f"fill={on}")


class Ellipse(RegionProperties):
    def __init__(self, x, y, maj, minor, angle):
        self.x = x
        self.y = y
        self.maj = maj
        self.minor = minor
        self.angle = angle
        self.specs = []
        self.pre_specs = []

    def __repr__(self):
        return f"{self.__class__.__name__.lower()}({x},{y},{maj},{minor},{angle})"

    def fill(self, on):
        if on:
            self.specs.append(f"fill={on}")


class Box(RegionProperties):
    def __init__(self, x, y, width, height, angle):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.angle = angle
        self.specs = []
        self.pre_specs = []

    def __repr__(self):
        return "{}({},{},{},{},{})".format(self.__class__.__name__, self.x,
            self.y, self.width, self.height, self.angle)

    def fill(self, on):
        if on:
            self.specs.append(f"fill={on}")


class Polygon(RegionProperties):
    def __init__(self, x1, y1, x2, y2, x3, y3, *args):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3
        self.args = args
        self.specs = []
        self.pre_specs = []
    
    def __repr__(self):
        return "{}({},{},{},{},{},{}, {','.join(self.args)})".format(
            self.__class__.__name__, self.x1, self.y1,
            self.x2, self.y2, self.x3, self.y3)

    def fill(self, on):
        if on:
            self.specs.append(f"fill={on}")


class Point(RegionProperties):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.specs = []
        self.pre_specs = []
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.x},{self.y})"


    def point(self, shape="cross", size=2):
        shapes = ["circle", "box", "diamond", "cross", "x", "arrow", "boxcircle"]
        if shape in shapes:
            self.specs.append(f"point={shape} {size}")


class Line(RegionProperties):
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.specs = []
        self.pre_specs = []

    def __repr__(self):
        return f"{self.__class__.__name__}({self.x1},{self.y1},{self.x2},{self.y2})"

    def line(self, arr_start=0, arr_end=0):
        self.specs.append(f"line={arr_start} {arr_end}")


class Vector(RegionProperties):
    def __init__(self, x1, y1, length, angle):  
        self.x1 = x1
        self.y1 = y1
        self.length = length
        self.angle = angle
        self.specs = []
        self.pre_specs = []

    def __repr__(self):
        return f"{self.__class__.__name__}({self.x1},{self.y1},{self.length},{self.angle})"

    def vector(self, on):
        if on:
            self.specs.append(f"vector={on}")


class Text(RegionProperties):
    def __init__(self, text, x, y):
        self.text = text
        self.x = x
        self.y = y
        self.specs = []
        self.pre_specs = []

    def __repr__(self):
        return f"{self.__class__.__name__}({self.text},{self.x},{self.y})"


class Annulus(RegionProperties):
    def __init__(self, x, y, inner):
        self.x = x
        self.y = y
        self.inner = inner
        self.specs = []
        self.pre_specs = []

    def __repr__(self):
        return f"{self.__class__.__name__}({self.x},{self.y},{self.inner})"


@dataclass
class Regions:
    circle = Circle
    ellipse = Ellipse
    box = Box
    polygon = Polygon
    point = Point
    line = Line
    vector = Vector
    text = Text
    annulus = Annulus


class LSM2Reg:
    regions = []
    def __init__(self, fname, out_name):
        self.fname = fname
        self.out_name = out_name

    def get_lsm_sources(self):
        print(f"Reading LSM file: {self.fname}")
        sources = Tigger.load(self.fname).sources
        return sources

    def src_to_reg(self, src_obj, shape="point", size=2):
        dec = "{}{}d{}m{:.3f}s".format(*src_obj.pos.dec_sdms())
        ra = "{}h{}m{:.3f}s".format(*src_obj.pos.ra_hms())
        if hasattr(Regions, shape):
            point_reg = getattr(Regions, shape)(ra, dec)
            point_reg.point(size=size)
            point_reg.tag(src_obj.name, src_obj.typecode)
            point_reg.width(3)
            # point_reg.zindex("source")
            return point_reg.comments
        else:
            raise("AttributeError")

    def normalizer(self, srcs):
        "I want to normalize the point sizes just"
        fluxes = [x.flux.I for x in srcs]
        drange = max(fluxes) - min(fluxes)
        # I intend the range of values to be from 1 to 10
        mrange = 10
        return mrange/drange

    def write_out(self):
        headers = [
            "# Region file format: DS9 version 4.0",
            'global color=yellow font="helvetica 10 normal roman" edit=1 move=1 delete=1 highlite=1 include=1 source=1',
            "physical"
        ]

        with open(self.out_name, "w") as ofile:
            ofile.writelines([x+"\n" for x in (headers + self.regions)])

        print(f"Region file -------> {self.out_name}\n")

    def generate_regions(self):
        sources = self.get_lsm_sources()
        norm = self.normalizer(sources)
        for src in sources:
            self.regions.append(self.src_to_reg(src, size=ceil(src.flux.I*norm)))
        self.write_out()



def arg_parser():
    parser = argparse.ArgumentParser(
        usage="%(prog)s ifile [ifile ...] -o [ofile ...]\n[] means optional",
        description="Convert LSM file to DS9 region file")
    parser.add_argument("ifile",  type=str, nargs="+",
        help="Input LSM file(s)")
    parser.add_argument("-o", "--output", dest="ofile", default=None, nargs="*",
        help="Name of the output file(s). Default will be the input name with a .reg extension")
    return parser




if __name__ == "__main__":
    parser = arg_parser().parse_args()

    if parser.ofile is None:
        parser.ofile = []
    
    for inp, oup in zip_longest(parser.ifile, parser.ofile):
        if oup is None:
            oup = f"{inp}.reg"
        lsm = LSM2Reg(inp, oup)
        lsm.generate_regions()

    