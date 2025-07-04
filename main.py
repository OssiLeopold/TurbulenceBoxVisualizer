# Import packages
import multiprocessing as mp                    # For multiprocessing
from multiprocessing import shared_memory
import analysator as pt
import numpy as np
from multiprocessing.resource_tracker import unregister

from utils.animation_specs import AnimationSpecs      # Class for animation object
from utils.animation_2D import Animation2D
from utils.animation_triple import AnimationTriple
from utils.animation_fourier import AnimationFourier
from utils.animation_sf import AnimationSF
from utils.animation_kurtosis import AnimationKurtosis

# Set path to simulation bulkfiles
bulkpath = "/home/rxelmer/Documents/turso/bulks/sim22/"
vlsvobj = pt.vlsvfile.VlsvReader(bulkpath + "bulk.0000000.vlsv")
x_length = vlsvobj.read_parameter("xcells_ini")

# Enter number of frames to be animated. Define start frame if you want to start from some point.
start_frame = 283
end_frame = 283

# Define what animations are to be produced:
# Each animation has to be in the from of a list, e.g: ["<animation_type>", "<variable>", "<component>", "<animation_spesific>"]

# <animation_type>: 2D, fourier, sf, kurtosis or triple:
#   - 2D: 2 dimensional heat map of specified variable.
#   - fourier: fourier transform of given variable.
#   - sf: structure function of given variable.
#   - kurtosis: kurtosis of given variable.

# <variable>: "B", "v", "J", "rho"

# <component>: "x", "y", "z", or "total" for vector variable and "pass" for scalar variable.

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
 
name_beginning = "TurbulenceBoxPlots/sim22_anim/sim22"
filetype = ".mp4"

animations = [
            ("sf", "B", "y",[10,20,40,80,160,320]),("sf", "B", "x",[10,20,40,80,160,320]),
            ("kurtosis", "B", "y",[10,20,40,80,160,320]),("kurtosis", "B", "x",[10,20,40,80,160,320])
             ]

# Turn list into list of AnimationSpecs objects
def cfg_to_AnimationSpecs(animations):
    for i, object in enumerate(animations):
        animations[i] = AnimationSpecs(
            animation_type = object[0], variable = object[1], component = object[2], 
            animation_specific = object[3], bulkpath=bulkpath, filetype = filetype
        )
    return animations

# Generate names for objects
def namer(animations):
    for object in animations:
        if object.variable == "rho":
            name = f"{name_beginning}_{object.animation_type}_{object.variable_name}{filetype}"

        elif object.animation_type == "2D":
            name = f"{name_beginning}_{object.animation_type}_{object.variable_name}_{object.component}_{object.animation_spesific}{filetype}"

        elif object.animation_type == "triple":
            name = f"{name_beginning}_{object.animation_type}_{object.variable_name}_{object.animation_spesific}{filetype}"

        elif object.animation_type == "fourier":
            
            if object.fourier_direc == "x" or object.fourier_direc == "y":
                name = f"{name_beginning}_{object.animation_type}_{object.variable_name}_{object.component}_{object.fourier_direc}_{object.fourier_loc}{filetype}"
            else:
                name = f"{name_beginning}_{object.animation_type}_{object.variable_name}_{object.component}_{object.fourier_type}{filetype}"
            
        elif object.animation_type == "sf":
            name = f"{name_beginning}_{object.animation_type}_{object.variable_name}_{object.component}_{object.delta_ls[0]}-{object.delta_ls[-1]}{filetype}"
        
        else:
            name = f"{name_beginning}_{object.animation_type}_{object.variable_name}_{object.component}{filetype}"
        
        object.name = name

# Fetch all cellids of the bulkfiles
def cellids_fetcher(object):
    cellids = []
    for i in range(end_frame - start_frame + 1):
        vlsvobj = pt.vlsvfile.VlsvReader(object.bulkpath + f"bulk.{str(start_frame + i).zfill(7)}.vlsv")
        cellids.append(vlsvobj.read_variable("CellID"))
    return cellids

# Define which variables need to be fetched. Employ logic to avoid duplicates.
def variables_to_be(animations):
    variables_to_be = [] # or not?
    for object in animations:
        if object.animation_type == "triple":
            for component in ["x","y","z"]:
                if (object.variable, component) not in variables_to_be:
                    variables_to_be.append((object.variable, component))

        elif (object.variable, object.component) not in variables_to_be:
            variables_to_be.append((object.variable, object.component))
        
        if object.animation_type == "2D" and object.unitless == True and (object.variable, "magnitude") not in variables_to_be and object.component != "pass":
            variables_to_be.append((object.variable, "magnitude"))
        
        if object.animation_type == "triple" and object.unitless == True and (object.variable, "magnitude") not in variables_to_be:
            variables_to_be.append((object.variable, "magnitude"))

        if ("time", "pass") not in variables_to_be:
            variables_to_be.append(("time", "pass"))

    return variables_to_be

# Fetch data from vlsvfiles and place into shared memory
def fetcher(variable_component):
    if variable_component[0] == "rho" or variable_component[1] == "magnitude":
        data = np.empty((end_frame - start_frame + 1, x_length*x_length))
        for frame in range(end_frame - start_frame + 1):
            vlsvobj = pt.vlsvfile.VlsvReader(bulkpath + f"bulk.{str(start_frame + frame).zfill(7)}.vlsv")
            data[frame] = np.array(vlsvobj.read_variable(variable_component[0], operator = variable_component[1]))[cellids[frame].argsort()]

    elif variable_component[0] == "time":
        data = np.empty(end_frame - start_frame + 1)
        for frame in range(end_frame - start_frame + 1):
            vlsvobj = pt.vlsvfile.VlsvReader(bulkpath + f"bulk.{str(start_frame + frame).zfill(7)}.vlsv")
            data[frame] = vlsvobj.read_parameter("time")

    else:
        data = np.empty((end_frame - start_frame + 1, x_length*x_length))
        for frame in range(end_frame - start_frame + 1):
            vlsvobj = pt.vlsvfile.VlsvReader(bulkpath + f"bulk.{str(start_frame + frame).zfill(7)}.vlsv")
            frame_data = np.array(vlsvobj.read_variable(variable_component[0], operator = variable_component[1]))[cellids[frame].argsort()]
            data[frame] = frame_data - np.mean(frame_data)

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

# Include shared_memory adress and relevant data to animation objects
def mem_space_includer(animations, shared_blocks):
    for object in animations:
        for block in shared_blocks:
            if object.variable == block["variable"] and object.component == block["component"]:
                object.memory_space = block["name"]
                object.shape = block["shape"]
                object.dtype = block["dtype"]
            elif block["variable"] == "time":
                object.time = block["name"]
                object.time_shape = block["shape"]
                object.time_dtype = block["dtype"]
                
        if object.animation_type in ["2D", "triple"] and object.unitless == True:
            for block in shared_blocks:
                if block["variable"] == object.variable and block["component"] == "magnitude":
                    object.memory_space_norm = block["name"]

        if object.animation_type == "triple":
            for block in shared_blocks:
                if object.variable == block["variable"] and block["component"] in ["x","y","z"]:
                    object.memory_space[block["component"]] = block["name"]
                    object.shape[block["component"]] = block["shape"]
                    object.dtype = block["dtype"]



# Function for launching correct animation for each animation object
def chooser(object):
    if object.animation_type == "2D":
        Animation2D(object)
    elif object.animation_type == "triple":
        AnimationTriple(object)
    elif object.animation_type == "fourier":
        AnimationFourier(object)
    elif object.animation_type == "sf":
        AnimationSF(object)
    elif object.animation_type == "kurtosis":
        AnimationKurtosis(object)

if __name__ == "__main__":
    animations = cfg_to_AnimationSpecs(animations)
    namer(animations)

    global cellids
    cellids = cellids_fetcher(animations[0])

    variables = variables_to_be(animations)

    # Fetch all needed data into separate shared memory blocks
    shared_blocks = []
    with mp.Pool(len(variables)) as process:
        shared_blocks = process.map(fetcher, variables)

    mem_space_includer(animations, shared_blocks)

    # Debugging
    for object in animations:
        print(object.memory_space)
        print(object.time)

    """ for object in animations:
        print(object.memory_space_norm) """

    for block in shared_blocks:
        print(block)

    # Launch a separate process for each AnimationSpecs object
    with mp.Pool(len(animations)) as process:
        process.map(chooser, animations)

    # Delete shared memory
    for block in shared_blocks:
        block['shm'].close()
        block['shm'].unlink() 