import os
from configparser import ConfigParser
import analysator as pt
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FFMpegWriter
import numpy as np
from scipy.stats import kurtosis
from multiprocessing import shared_memory
import numexpr as ne

config = ConfigParser()
config.read(".TurbulenceBoxVisualizer.ini")

plt.rcParams['animation.ffmpeg_path'] = config["paths"]["ffmpeg_path"]

os.environ['PATH']= config["paths"]["latex_path"] + os.environ['PATH'] 
os.environ['PTNOLATEX']='1'

class AnimationKurtosis():
    def __init__(self, object):
        self.object = object
        shm = shared_memory.SharedMemory(name=object.memory_space)
        self.data = np.ndarray(object.shape, dtype=object.dtype, buffer=shm.buf)

        self.vlsvobj = pt.vlsvfile.VlsvReader(object.bulkpath + "bulk.0000000.vlsv")
        self.x_length = int(self.vlsvobj.read_parameter("xcells_ini"))
        self.frames = len(self.data)

        self.slices_pos = [50,951,50]
        self.slice_n = len(self.slices_pos)

        shm_time = shared_memory.SharedMemory(name=object.time)
        self.time = np.ndarray(object.time_shape, dtype=object.time_dtype, buffer=shm_time.buf)

        fig, self.ax = plt.subplots()

        self.data_mesh_x = np.empty((self.frames, self.x_length, self.x_length))
        for i in range(self.frames):
                self.data_mesh_x[i] = self.data[i].reshape(-1, self.x_length)
        
        self.data_mesh_y = np.empty((self.frames, self.x_length, self.x_length))
        for i in range(self.frames):
                self.data_mesh_y[i] = self.data[i].reshape(-1, self.x_length).T

        self.ticks = []
        self.tick_labels = []
        for dl in self.object.delta_ls:
            self.ticks.append(1/dl)
            label = f"$1/{{{dl}}}$"
            self.tick_labels.append(r"{}".format(label))

        self.timelabel = self.ax.text(0.98, 1.02, "",transform=self.ax.transAxes)

        anim = animation.FuncAnimation(fig, self.update, frames = self.frames, interval = 20)
        
        writer = FFMpegWriter(fps=5)
        anim.save(object.name, writer=writer)
        plt.close()

    def update(self,frame):
        self.ax.clear()

        delta_array_container = np.empty((len(self.object.delta_ls), 2 * self.slice_n * self.x_length))
        
        for i, dl in enumerate(self.object.delta_ls):
            slices = np.empty((2 * self.slice_n, self.x_length))
            slices[:self.slice_n] = self.data_mesh_x[frame][self.slices_pos]
            slices[self.slice_n:] = self.data_mesh_y[frame][self.slices_pos]

            slices_shifted = np.roll(slices,dl)

            delta_array_container[i] = ne.evaluate('slices - slices_shifted').flatten()

        """ for i in range(4):    
            print(i,delta_array_container[i][-10:-1]) """
        
        kurtoi = []
        for i in range(len(self.object.delta_ls)):
            mean = np.mean(delta_array_container[i])
            SD = np.std(delta_array_container[i], mean=mean)
            delta_array_container[i] = (delta_array_container[i] - mean) / SD
            kurtoi.append(kurtosis(delta_array_container[i], fisher=False, bias =True) - 3)

        self.ax.plot(1 / np.array(self.object.delta_ls), kurtoi)
        
        self.ax.set_xscale("log")
        xlabel = f"$\\frac{{1}}{{\\Delta l}}$"
        ylabel = f"$K$"
        self.ax.set_xlabel(r'{}'.format(xlabel))
        self.ax.set_ylabel(r'{}'.format(ylabel))
        #self.ax.set_ylim(-2,4)

        self.ax.set_xticks(self.ticks)
        self.ax.set_xticklabels(self.tick_labels)

        self.timelabel.set_text(f"{self.time[frame]:.1f}s")

