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
from utils.animation_reconnection import AnimationReconnection
from utils.plot_franci import PlotFranci
from utils.animation_sigma import AnimationSigma

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

# Define which variables need to be fetched.
def variables_to_be(animations):
    variables_to_be = set() # or not?
    for object in animations:
        if object.animation_type == "triple":
            for component in ["x","y","z"]:
                variables_to_be.add((object.variable, component))

        elif object.animation_type == "rms" and object.component == "pass":
            for component in ["x","y","z"]:
                variables_to_be.add((object.variable, component))

        elif object.component == "perp":
            for component in ["x","y"]:
                variables_to_be.add((object.variable, component))

        elif object.animation_type in ["franci", "sigma"]:
            for component in ["x","y","z"]:
                variables_to_be.add(("vg_b_vol", component))
                variables_to_be.add(("proton/vg_v", component))
            variables_to_be.add(("proton/vg_rho", "pass"))

            if object.animation_type == "franci":
                variables_to_be.add(("vg_j", "z"))
                variables_to_be.add(("vg_ttensor", "pass"))

        elif object.animation_type == "reconnection":
            for component in ["x","y"]:
                variables_to_be.add(("vg_b_vol", component))

            variables_to_be.add((object.variable, object.component))

        else:
            variables_to_be.add((object.variable, object.component))
        
        if object.animation_type == "2D" and object.unitless == True and object.component != "pass":
            variables_to_be.add((object.variable, "magnitude"))

        if object.animation_type == "triple" and object.unitless == True:
            variables_to_be.add((object.variable, "magnitude"))

        variables_to_be.add(("time", "pass"))

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
        "address": shm.name,
        "shape": data.shape,
        "dtype": data.dtype,
        "shm": shm,
    }
    unregister(shm._name, 'shared_memory')
    return (variable_component[0] + variable_component[1], block)

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
    elif object.animation_type == "reconnection":
        AnimationReconnection(object)
    elif object.animation_type == "franci":
        PlotFranci(object)
    elif object.animation_type == "sigma":
        AnimationSigma(object)

if __name__ == "__main__":
    animations = cfg_to_AnimationSpecs(animations)

    global cellids
    cellids = cellids_fetcher(animations[0])

    variables = variables_to_be(animations)

    # Fetch all needed data into separate shared memory blocks
    shared_blocks = []
    with mp.Pool(len(variables)) as process:
        shared_blocks = process.map(fetcher, variables)

    shared_blocks_dict = {}
    for i in range(len(shared_blocks)):
        shared_blocks_dict[shared_blocks[i][0]] = shared_blocks[i][1]

    # Include memory space addresses to animation objects
    for animation in animations:
        animation.memory_space = shared_blocks_dict

    # Launch a separate process for each AnimationSpecs object
    with mp.Pool(len(animations)) as process:
        process.map(chooser, animations)

    # Delete shared memory
    for block in shared_blocks:
        block[1]['shm'].close()
        block[1]['shm'].unlink()