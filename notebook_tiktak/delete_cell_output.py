#! /bin/python3

import json
from glob import glob


def write_ipnb(inf, data):
    with open(inf, "w") as fil:
        json.dump(data, fil, sort_keys=False, indent=2)
    print("Output cells cleared for: {}".format(inf))

def read_ipnb(fname):
    with open(fname, "r") as js:
        data = json.load(js)
    return data

pynb_files = glob("**/*.ipynb", recursive=False)
print(f"Found {len(pynb_files)}  notebook files")

for fil in pynb_files:
    data = read_ipnb(fil)

    # #changing all the kernels to my kernel for my convenience
    data["metadata"]["kernelspec"].update({
       "display_name": "interferometry (local)",
       "language": "python",
       "name": "interferometry"
      })


    # for cell in data["cells"]:
    #     if "outputs" in cell:
    #         cell["outputs"] = []
    #     if "execution_count" in cell:
    #         cell["execution_count"] = None

    write_ipnb(fil, data)

print("Done clearing output cells")
print("--------------------------")