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
        if object.component in ["x","y","z","magnitude"]:
            self.animation_one()
        else:
            self.animation_all()

    def animation_one(self):
        shm = shared_memory.SharedMemory(name=self.object.memory_space)
        data = np.ndarray(self.object.shape, dtype=self.object.dtype, buffer=shm.buf)

        shm_time = shared_memory.SharedMemory(name=self.object.time)
        self.time = np.ndarray(self.object.time_shape, dtype=self.object.time_dtype, buffer=shm_time.buf)

        self.vlsvobj = pt.vlsvfile.VlsvReader(self.object.bulkpath + "bulk.0000000.vlsv")
        self.x_length = int(self.vlsvobj.read_parameter("xcells_ini"))
        self.frames = len(data)

        fig, self.ax = plt.subplots()

        label = f"${{{self.object.variable_name}}}_{{{self.object.component}}}$"

        self.p = [self.ax.plot([],[], label = r'{}'.format(label))]

        self.rms = np.sqrt(np.mean(data ** 2, axis=1) - np.mean(data, axis=1)**2)

        self.ax.set_xlim(0, self.time[-1])
        self.ax.set_ylim(min([min(self.rms), min(self.rms)])*0.9, max([max(self.rms), max(self.rms)])*1.1)
        self.ax.legend()
        
        anim = animation.FuncAnimation(fig, self.update_one, frames = self.frames + 1, interval = 20)
        
        writer = FFMpegWriter(fps=5)
        anim.save(self.object.name, writer=writer)
        plt.close()

    def update_one(self, frame):
        self.p[0][0].set_data(self.time[:frame], self.rms[:frame])

    def animation_all(self):
        shm_x = shared_memory.SharedMemory(name=self.object.memory_space["x"])
        shm_y = shared_memory.SharedMemory(name=self.object.memory_space["y"])
        shm_z = shared_memory.SharedMemory(name=self.object.memory_space["z"])
        data_x = np.ndarray(self.object.shape["x"], dtype=self.object.dtype, buffer=shm_x.buf)
        data_y = np.ndarray(self.object.shape["y"], dtype=self.object.dtype, buffer=shm_y.buf)
        data_z = np.ndarray(self.object.shape["z"], dtype=self.object.dtype, buffer=shm_z.buf)

        shm_time = shared_memory.SharedMemory(name=self.object.time)
        self.time = np.ndarray(self.object.time_shape, dtype=self.object.time_dtype, buffer=shm_time.buf)

        self.vlsvobj = pt.vlsvfile.VlsvReader(self.object.bulkpath + "bulk.0000000.vlsv")
        self.x_length = int(self.vlsvobj.read_parameter("xcells_ini"))
        self.frames = len(data_x)

        fig, self.ax = plt.subplots()

        perp_label = f"${{{self.object.variable_name}}}_{{\\perp}}$"
        par_label = f"${{{self.object.variable_name}}}_{{\\parallel}}$"

        self.p = [self.ax.plot([],[], label = r'{}'.format(perp_label)), self.ax.plot([],[], label = r'{}'.format(par_label))]

        self.rms_par = np.sqrt(np.mean(data_z ** 2, axis=1) - np.mean(data_z, axis=1)**2)

        perp = np.sqrt(data_x**2 + data_y**2)
        self.rms_perp = np.sqrt(np.mean(perp ** 2, axis=1) - np.mean(perp, axis=1)**2)

        self.ax.set_xlim(0, self.time[-1])
        self.ax.set_ylim(min([min(self.rms_perp), min(self.rms_par)])*0.9, max([max(self.rms_perp), max(self.rms_par)])*1.1)
        self.ax.legend()
        
        anim = animation.FuncAnimation(fig, self.update_all, frames = self.frames + 1, interval = 20)
        
        writer = FFMpegWriter(fps=5)
        anim.save(self.object.name, writer=writer)
        plt.close()

    def update_all(self,frame):
        self.p[0][0].set_data(self.time[:frame], self.rms_perp[:frame])
        self.p[1][0].set_data(self.time[:frame], self.rms_par[:frame])

