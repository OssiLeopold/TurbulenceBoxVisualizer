# Import packages
import multiprocessing as mp                    # For multiprocessing
from multiprocessing import shared_memory
import analysator as pt
import numpy as np
from multiprocessing.resource_tracker import unregister
from configparser import ConfigParser
import ast

from utils.animation_specs import AnimationSpecs      # Class for animation object
from utils.animation_2D import Animation2D
from utils.animation_triple import AnimationTriple
from utils.animation_fourier import AnimationFourier
from utils.animation_sf import AnimationSF
from utils.animation_kurtosis import AnimationKurtosis
from utils.animation_rms import AnimationRMS

config = ConfigParser()
config.read(".TurbulenceBoxVisualizer.ini")

# Set path to simulation bulkfiles
bulkpath = config["paths"]["bulkpath"]
vlsvobj = pt.vlsvfile.VlsvReader(bulkpath + "bulk.0000000.vlsv")
x_length = vlsvobj.read_parameter("xcells_ini")

# Get start frame and end frame from config
start_frame = int(config["settings"]["start_frame"])
end_frame = int(config["settings"]["end_frame"])

animations = list(ast.literal_eval(config["settings"]["animations"]))

# Turn list into list of AnimationSpecs objects
def cfg_to_AnimationSpecs(animations):
    for i, object in enumerate(animations):
        animations[i] = AnimationSpecs(
            animation_type = object[0], variable = object[1],
            component = object[2], animation_specific = object[3]
        )
    return animations

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

        elif object.animation_type == "rms" and object.component == "pass":
            for component in ["x","y","z"]:
                if (object.variable, component) not in variables_to_be:
                    variables_to_be.append((object.variable, component))

        elif object.component == "perp":
            for component in ["x","y"]:
                if (object.variable, component) not in variables_to_be:
                    variables_to_be.append((object.variable, component))

        elif object.variable == "residual":
            for component in ["x","y","z"]:
                if ("vg_b_vol", component) not in variables_to_be:
                    variables_to_be.append(("vg_b_vol", component))
                if ("proton/vg_v", component) not in variables_to_be:
                    variables_to_be.append(("proton/vg_v", component))

            if ("proton/vg_rho", "pass") not in variables_to_be:
                variables_to_be.append(("proton/vg_rho", "pass"))
            
            if ("vg_j", "z") not in variables_to_be:
                variables_to_be.append(("vg_j", "z"))

            if ("vg_ttensor", "pass") not in variables_to_be:
                variables_to_be.append(("vg_ttensor", "pass"))

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
    if variable_component[0] == "proton/vg_rho" or variable_component[1] == "magnitude":
        data = np.empty((end_frame - start_frame + 1, x_length*x_length))
        for frame in range(end_frame - start_frame + 1):
            vlsvobj = pt.vlsvfile.VlsvReader(bulkpath + f"bulk.{str(start_frame + frame).zfill(7)}.vlsv")
            data[frame] = np.array(vlsvobj.read_variable(variable_component[0], operator = variable_component[1]))[cellids[frame].argsort()]

    elif variable_component[0] == "vg_ttensor":
        data = np.empty((end_frame - start_frame + 1, x_length*x_length,3))
        for frame in range(end_frame - start_frame + 1):
            vlsvobj = pt.vlsvfile.VlsvReader(bulkpath + f"bulk.{str(start_frame + frame).zfill(7)}.vlsv")
            data[frame] = np.diagonal(vlsvobj.read_variable(variable_component[0], operator = variable_component[1])[cellids[frame].argsort()], axis1=1, axis2=2)

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

        if object.component == "perp":
            for block in shared_blocks:
                if block["variable"] == object.variable and block["component"] in ["x","y"]:
                    object.memory_space[block["component"]] = block["name"]
                    object.shape[block["component"]] = block["shape"]
                    object.dtype = block["dtype"]

        if object.animation_type == "triple":
            for block in shared_blocks:
                if object.variable == block["variable"] and block["component"] in ["x","y","z"]:
                    object.memory_space[block["component"]] = block["name"]
                    object.shape[block["component"]] = block["shape"]
                    object.dtype = block["dtype"]

        if object.animation_type == "rms" and object.component == "pass":
            for block in shared_blocks:
                if object.variable == block["variable"] and block["component"] in ["x","y","z"]:
                    object.memory_space[block["component"]] = block["name"]
                    object.shape[block["component"]] = block["shape"]
                    object.dtype = block["dtype"]

        if object.variable == "residual":
            for block in shared_blocks:
                if "vg_b_vol" == block["variable"] and block["component"] in ["x", "y", "z"]:
                    object.memory_space[block["variable"] + block["component"]] = block["name"]
                    object.shape[block["variable"]] = block["shape"]
                    object.dtype = block["dtype"]
                if "proton/vg_v" == block["variable"] and block["component"] in ["x", "y", "z"]:
                    object.memory_space[block["variable"] + block["component"]] = block["name"]
                    object.shape[block["variable"]] = block["shape"]
                    object.dtype = block["dtype"]
                if "proton/vg_rho" == block["variable"]:
                    object.memory_space[block["variable"]] = block["name"]
                    object.shape[block["variable"]] = block["shape"]
                    object.dtype = block["dtype"]
                if "vg_ttensor" == block["variable"]:
                    object.memory_space[block["variable"]] = block["name"]
                    object.shape[block["variable"]] = block["shape"]
                    object.dtype = block["dtype"]
                if "vg_j" == block["variable"]:
                    object.memory_space[block["variable"]] = block["name"]
                    object.shape[block["variable"]] = block["shape"]
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
    elif object.animation_type == "rms":
        AnimationRMS(object)

if __name__ == "__main__":
    animations = cfg_to_AnimationSpecs(animations)

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
        print(object.memory_space_norm)"""

    for block in shared_blocks:
        print(block)

    # Launch a separate process for each AnimationSpecs object
    with mp.Pool(len(animations)) as process:
        process.map(chooser, animations)

    # Delete shared memory
    for block in shared_blocks:
        block['shm'].close()
        block['shm'].unlink() 