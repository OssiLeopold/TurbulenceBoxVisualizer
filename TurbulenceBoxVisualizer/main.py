# Import packages
import multiprocessing as mp                    # For multiprocessing
from multiprocessing import shared_memory
import analysator as pt
import numpy as np
from multiprocessing.resource_tracker import unregister

from animation_specs import AnimationSpecs      # Class for animation object
from animation_2D import Animation2D

# Set path to simulation bulkfiles
bulkpath = "/home/rxelmer/Documents/turso/bulks/sim21/"

# Enter number of frames to be animated. Define start frame if you want to start from some point.
start_frame = 0
end_frame = 20

# Define what animations are to be produced:
# Each animation has to be in the from of a list, e.g: ["<animation_type>", "<variable>", "<component>", "<animation_spesific>"]

# <animation_type>: 2D, fourier, sf, kurtosis or triple:
#   - 2D: 2 dimensional heat map of specified variable.
#   - fourier: fourier transform of given variable.
#   - sf: structure function of given variable.
#   - kurtosis: kurtosis of given variable.

# <variable>: "B", "v", "J", "rho"

# <component>: "x", "y", "z", "total" or "all" for vector variable and "pass" for scalar variable.

# <animation_spesific>:
#   - 2D and tirple: "unit" or "unitless"

#   - fourier:
#       - ["x", <0-1>] -> x (or y) states the direction along which you want the slice to be done.
#         <0-1> states the y (or x) coordinate where you want the slice to be done (0-1 -> y_min - y_max).
# 
#       - ["diag", <1 or 2>] -> if you want fourier in diagonal direction. Enter 1 if you want SW->NE.
#                   Enter 2 if you want NW->SW.
#
#       - ["trace", <0-1>, <0-1>] -> If you want trace PSD.
#          First number for x slice y-coord and second number for y slice x-coord.
#
#       - ["trace_diag"] -> for trace PSD for diag directions.
#
#   - sf: a list like [2,4,6...] which states the dl in cells for structure function.
#
#   - kurtosis: again a list like [2,4,6...].
 
name_beginning = "TurbulenceBoxPlots/sim21_anim/sim21"
filetype = ".mp4"

animations = [
              ("2D", "B", "x", "unitless"),("2D", "B", "y", "unitless"),("2D", "B", "z", "unitless"),("2D", "rho", "pass", "unit")
             ]

# Generate names for objects
names = []
for object in animations:
    if object[1] == "rho":
        names.append(f"{name_beginning}_{object[0]}_{object[1]}{filetype}")

    elif object[0] == "2D":
        names.append(f"{name_beginning}_{object[0]}_{object[1]}_{object[2]}_{object[3]}{filetype}")

    elif object[1] == "fourier":
        
        if object[3][0] == "x" or object[3][0] == "y":
            names.append(
                f"{name_beginning}_{object[0]}_{object[1]}_{object[2]}_{object[3][0]}_{object[3][1]}{filetype}")
        else:
            names.append(f"{name_beginning}_{object[0]}_{object[1]}_{object[2]}_{object[3][0]}{filetype}")
    
    else:
        names.append(f"{name_beginning}_{object[0]}_{object[1]}_{object[2]}{filetype}")

# Turn list into list of AnimationSpecs objects
for i, object in enumerate(animations):
    animations[i] = AnimationSpecs(
        animation_type = object[0], variable = object[1], component = object[2], 
        animation_specific = object[3], name = names[i], bulkpath=bulkpath
    )

def fetcher(variable_component):
    data = []
    if variable_component[0] == "rho" or variable_component[1] == "magnitude":
        for frame in range(end_frame - start_frame):
            vlsvobj = pt.vlsvfile.VlsvReader(object.bulkpath + f"bulk.{str(start_frame + frame).zfill(7)}.vlsv")
            data.append(
                np.array(vlsvobj.read_variable(variable_component[0], operator = variable_component[1]))[cellids[frame].argsort()])
    else:
        for frame in range(end_frame - start_frame):
            vlsvobj = pt.vlsvfile.VlsvReader(object.bulkpath + f"bulk.{str(start_frame + frame).zfill(7)}.vlsv")
            frame = np.array(vlsvobj.read_variable(variable_component[0], operator = variable_component[1]))[cellids[frame].argsort()]
            data.append(frame - np.mean(frame))

    data = np.array(data)

    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shm_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shm_array[:] = data[:]  # Copy data into shared memory
    block = {
        "name": shm.name,
        "shape": data.shape,
        "dtype": data.dtype,
        "shm": shm,
        "variable": variable_component[0],
        "component": variable_component[1]
    }
    unregister(shm._name, 'shared_memory')
    return block


def chooser(object):
    if object.animation_type == "2D":
        Animation2D(object)
"""     elif object.animation_type == "triple":
        animation_triple()
    elif object.animation_type == "fourier":
        animation_fourier()
    elif object.animation_type == "sf":
        animation_sf()
    elif object.animation_type == "kurtosis":
        animation_kurtosis() """

variables_to_be = [] # or not?
for object in animations:
    if object.animation_type == "2D" and object.unitless == True and (object.variable, "magnitude") not in variables_to_be and object.component != "pass":
        variables_to_be.append((object.variable, "magnitude"))
    if (object.variable, object.component) not in variables_to_be:
        variables_to_be.append((object.variable, object.component))

cellids = []
for i in range(end_frame - start_frame):
    vlsvobj = pt.vlsvfile.VlsvReader(object.bulkpath + f"bulk.{str(start_frame + i).zfill(7)}.vlsv")
    cellids.append(vlsvobj.read_variable("CellID"))

# Fetch all needed data into separate shared memory blocks
shared_blocks = []
with mp.Pool(len(variables_to_be)) as process:
    shared_blocks = process.map(fetcher, variables_to_be)

for object in animations:
    for block in shared_blocks:
        if object.variable == block["variable"] and object.component == block["component"]:
            object.memory_space = block["name"]
            object.shape = block["shape"]
            object.dtype = block["dtype"]
        if object.animation_type == "2D" and object.unitless == True and object.component != "pass":
            for block_norm in shared_blocks:
                if block_norm["variable"] == block["variable"] and block_norm["component"] == "magnitude":
                    object.memory_space_norm = block_norm["name"]

""" for object in animations:
    print(object.memory_space_norm)

for block in shared_blocks:
    print(block) """

# Launch a separate process for each AnimationSpecs object
with mp.Pool(len(animations)) as process:
    process.map(chooser, animations)

# Delete shared memory
for block in shared_blocks:
    block['shm'].close()
    block['shm'].unlink() 