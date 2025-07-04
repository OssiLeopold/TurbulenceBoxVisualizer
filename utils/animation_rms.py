import os
from configparser import ConfigParser
import analysator as pt
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from multiprocessing import shared_memory
from matplotlib import animation
from matplotlib.animation import FFMpegWriter

config = ConfigParser()
config.read(".TurbulenceBoxVisualizer.ini")

# Telling FFMpegWriter the location of FFMpeg
plt.rcParams['animation.ffmpeg_path'] = config["paths"]["ffmpeg_path"]

#enabling use of latex
os.environ['PATH']= config["paths"]["latex_path"] + os.environ['PATH'] 
os.environ['PTNOLATEX']='1'

class AnimationRMS():
    def __init__(self, object):
        self.object = object
        shm = shared_memory.SharedMemory(name=object.memory_space)
        self.data = np.ndarray(object.shape, dtype=object.dtype, buffer=shm.buf)

        shm_time = shared_memory.SharedMemory(name=object.time)
        self.time = np.ndarray(object.time_shape, dtype=object.time_dtype, buffer=shm_time.buf)

        self.vlsvobj = pt.vlsvfile.VlsvReader(object.bulkpath + "bulk.0000000.vlsv")
        self.x_length = int(self.vlsvobj.read_parameter("xcells_ini"))
        self.frames = len(self.data)

        fig, self.ax = plt.subplots()

        self.p = [self.ax.plot([],[])]
        self.rms = np.empty(self.frames)

        self.ax.set_xlim(0, self.time[-1])
        
        anim = animation.FuncAnimation(fig, self.update, frames = self.frames, interval = 20)
        
        writer = FFMpegWriter(fps=5)
        anim.save(object.name, writer=writer)
        plt.close()

    def update(self,frame):
        self.rms[frame] = np.sqrt(np.sum(self.data[frame]**2)/(self.x_length*self.x_length))
        
        self.p[0][0].set_data(self.time[:frame], self.rms[:frame])

