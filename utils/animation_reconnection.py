import os
from configparser import ConfigParser
import analysator as pt
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import shared_memory
from matplotlib import animation
from matplotlib.animation import FFMpegWriter

config = ConfigParser()
config.read(".TurbulenceBoxVisualizer.ini")

#Telling FFMpegWriter the location of FFMpeg
plt.rcParams['animation.ffmpeg_path'] = config["paths"]["ffmpeg_path"]

#enabling use of latex
os.environ['PATH']= config["paths"]["latex_path"] + os.environ['PATH'] 
os.environ['PTNOLATEX']='1'

class AnimationReconnection():
    def __init__(self, object):
        self.object = object
        self.memory_space = object.memory_space

        shm_time = shared_memory.SharedMemory(name=self.memory_space["timepass"]["address"])
        self.time = np.ndarray(self.memory_space["timepass"]["shape"], dtype=self.memory_space["timepass"]["dtype"], buffer=shm_time.buf)

        self.vlsvobj = pt.vlsvfile.VlsvReader(object.bulkpath + "bulk.0000000.vlsv")
        self.cellids = self.vlsvobj.read_variable("CellID")

        prot_plas_freq = np.sqrt(1e6 * (1.602176634 * 10**(-19))**2 / (8.8541878128 * 10**(-12) * 1.67262192595 * 10**(-27)))
        dp = 299792458 / prot_plas_freq

        self.x_length = int(self.vlsvobj.read_parameter("xcells_ini"))
        coords = np.array(self.vlsvobj.get_cell_coordinates(np.sort(self.cellids))).T
        x = coords[0]
        y = coords[1]

        self.x_mesh = x.reshape(-1,self.x_length) / dp
        self.y_mesh = y.reshape(-1,self.x_length) / dp

        self.frames = len(self.time)

        if object.unitless == True:
            self.reconnection_unitless()
        else:
            self.reconnection_unit()

#def reconnection_unitless(self):




    def reconnection_unit(self):
        self.fig, self.ax = plt.subplots()

        mem_Bx = self.memory_space["vg_b_vol" + "x"]
        mem_By = self.memory_space["vg_b_vol" + "y"]

        shm_Bx = shared_memory.SharedMemory(name=mem_Bx["address"])
        shm_By = shared_memory.SharedMemory(name=mem_By["address"])
        
        B_x = np.ndarray(mem_Bx["shape"], dtype=mem_Bx["dtype"], buffer=shm_Bx.buf)
        B_y = np.ndarray(mem_By["shape"], dtype=mem_By["dtype"], buffer=shm_By.buf)

        if self.object.component == "perp":
            background = np.sqrt(B_x**2 + B_y**2)

        else:
            mem_background = self.object.memory_space[self.object.variable + self.object.component]
            shm_background = shared_memory.SharedMemory(name=mem_background["address"])
            background = np.ndarray(mem_background["shape"], dtype=mem_background["dtype"], buffer=shm_background.buf)

        self.background_mesh = background.reshape((self.frames, self.x_length, self.x_length))
        self.B_x_mesh = B_x.reshape((self.frames, self.x_length, self.x_length))
        self.B_y_mesh = B_y.reshape((self.frames, self.x_length, self.x_length))

        dx = np.diff(self.x_mesh[0])[0]            

        kx = 2*np.pi*np.fft.fftfreq(self.x_length, d=dx)
        ky = 2*np.pi*np.fft.fftfreq(self.x_length, d=dx)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2

        Bx_hat = np.fft.fft2(self.B_x_mesh, axes=(-2, -1))
        By_hat = np.fft.fft2(self.B_y_mesh, axes=(-2, -1))

        num = 1j * (KY*Bx_hat - KX*By_hat)
        Az_hat = np.empty((self.frames, self.x_length, self.x_length), dtype=complex)
        mask = K2 != 0
        for frame in range(self.frames):
            Az_hat[frame][mask] = -num[frame][mask] / K2[mask]
            Az_hat[frame][~mask] = 0

        self.Az = np.fft.ifft2(Az_hat, axes = (-2, -1)).real

        self.Min_A = round(min(self.Az.flatten()), 15)
        self.Max_A = round(max(self.Az.flatten()), 15)

        if abs(self.Min_A) > abs(self.Max_A):
            self.Max_A = -self.Min_A
        else:
            self.Min_A = -self.Max_A

        self.Min = round(min(background.flatten()), 15)
        self.Max = round(max(background.flatten()), 15)

        if abs(self.Min) > abs(self.Max):
            self.Max = -self.Min
        else:
            self.Min = -self.Max

        self.first = True
        anim = animation.FuncAnimation(self.fig, self.contour_update_unit, frames = self.frames, interval = 20)
        writer = FFMpegWriter(fps = 5)
        anim.save(self.object.name, writer = writer)

    def contour_update_unit(self, frame):
        self.ax.clear()
        pcm = self.ax.pcolormesh(self.x_mesh, self.y_mesh, self.background_mesh[frame], vmin = self.Min, vmax = self.Max,cmap = "bwr")
        if self.first == True:
            self.fig.colorbar(pcm, ax=self.ax)
        self.ax.contour(
            self.x_mesh, self.y_mesh, self.Az[frame], levels = 10, vmin = self.Min_A, vmax = self.Max_A)
        self.first = False