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
#from utils.animation_triple import AnimationTriple
#from utils.animation_fourier import AnimationFourier
#from utils.animation_sf import AnimationSF
#from utils.animation_kurtosis import AnimationKurtosis
#from utils.animation_rms import AnimationRMS

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
def config_to_AnimationSpecs(animations):
    animation_dict = {"B": [], "v": [], "J": [], "rho": []}
    for animation in animations:
        if animation[1] == "B":
            animation_dict["B"].append(animation)
        elif animation[1] == "v":
            animation_dict["v"].append(animation)
        elif animation[1] == "J":
            animation_dict["J"].append(animation)
        elif animation[1] == "rho":
            animation_dict["rho"].append(animation)
        else:
            print("A variable is specified incorrectly")

    animations_sorted = []
    for key in animation_dict.keys():
        animations_sorted.extend(animation_dict[key])

    for i, object in enumerate(animations_sorted):
        animations_sorted[i] = AnimationSpecs(
            animation_type = object[0], variable = object[1],
            component = object[2], animation_specific = object[3]
        )
    return animations_sorted

# Fetch all cellids of the bulkfiles
def cellids_fetcher(object):
    cellids = []
    for i in range(end_frame - start_frame + 1):
        vlsvobj = pt.vlsvfile.VlsvReader(object.bulkpath + f"bulk.{str(start_frame + i).zfill(7)}.vlsv")
        cellids.append(vlsvobj.read_variable("CellID"))
    return cellids

# Define which variables need to be fetched. Employ logic to avoid duplicates.
def needed_variables(object):
    variables_to_be = [] # or not?
    
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
    shared_blocks.append(block)

# Include shared_memory adress and relevant data to animation objects
def mem_space_includer(object, shared_blocks):
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
    animations = config_to_AnimationSpecs(animations)

    for animation in animations:
        print(animation.variable)

    global cellids
    cellids = cellids_fetcher(animations[0])

    global shared_blocks
    shared_blocks = []

    for i, animation in enumerate(animations):
        variables = needed_variables(animation)

        if len(shared_blocks) == 0:
            for variable in variables:
                fetcher(variable)
        else:
            for variable in variables:
                fetch = True
                for block in shared_blocks:
                    if variable == (block["variable"], block["component"]):
                        fetch = False
                    
                if fetch == True:
                    fetcher(variable)

        mem_space_includer(animation, shared_blocks)
        chooser(animation)
        if i+1 < len(animations) and animation.variable != animations[i+1].variable:
                new_shared_blocks = []
                for block in shared_blocks:
                    if block["variable"] != "time":
                        block['shm'].close()
                        block['shm'].unlink()
                    else:
                        new_shared_blocks.append(block) 
                shared_blocks = new_shared_blocks


    for block in shared_blocks:
        print(block)

    # Delete shared memory
    for block in shared_blocks:
        block['shm'].close()
        block['shm'].unlink() 