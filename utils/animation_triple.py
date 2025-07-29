import os
from configparser import ConfigParser
import analysator as pt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

class AnimationTriple():
    def __init__(self, object):
        self.object = object

        shm_x = shared_memory.SharedMemory(name=object.memory_space["x"])
        shm_y = shared_memory.SharedMemory(name=object.memory_space["y"])
        shm_z = shared_memory.SharedMemory(name=object.memory_space["z"])
        self.data_x = np.ndarray(object.shape["x"], dtype=object.dtype, buffer=shm_x.buf)
        self.data_y = np.ndarray(object.shape["y"], dtype=object.dtype, buffer=shm_y.buf)
        self.data_z = np.ndarray(object.shape["z"], dtype=object.dtype, buffer=shm_z.buf)
        
        shm_time = shared_memory.SharedMemory(name=object.time)
        self.time = np.ndarray(object.time_shape, dtype=object.time_dtype, buffer=shm_time.buf)

        prot_plas_freq = np.sqrt(1e6 * (1.602176634 * 10**(-19))**2 / (8.8541878128 * 10**(-12) * 1.67262192595 * 10**(-27)))
        dp = 299792458 / prot_plas_freq

        self.vlsvobj = pt.vlsvfile.VlsvReader(object.bulkpath + "bulk.0000000.vlsv")
        self.cellids = self.vlsvobj.read_variable("CellID")

        self.x_length = self.vlsvobj.read_parameter("xcells_ini")
        coords = np.array(self.vlsvobj.get_cell_coordinates(np.sort(self.cellids))).T
        x = coords[0]
        y = coords[1]
        self.x_mesh = x.reshape(-1,self.x_length) / dp
        self.y_mesh = y.reshape(-1,self.x_length) / dp
        self.frames = len(self.data_x)

        if object.unitless == True:
            self.animation_unitless()
        else:
            self.animation_unit()

    def animation_unitless(self):
        shm_norm = shared_memory.SharedMemory(name=self.object.memory_space_norm)
        mag = np.ndarray(self.object.shape["x"], dtype=self.object.dtype, buffer=shm_norm.buf)

        mag_average = np.mean(mag, axis=1).reshape((self.frames, 1))

        unitless_data_x = self.data_x / mag_average
        unitless_data_y = self.data_y / mag_average
        unitless_data_z = self.data_z / mag_average

        fig, self.axes = plt.subplots(1,3, figsize=(26,8))
        fig.tight_layout(pad=4.0)

        self.Min = round(min(np.array([unitless_data_x,unitless_data_y,unitless_data_z]).flatten()), 20)
        self.Max = round(max(np.array([unitless_data_x,unitless_data_y,unitless_data_z]).flatten()), 20)

        if abs(self.Min) > abs(self.Max):
            self.Max = -self.Min
        else:
            self.Min = -self.Max

        self.data_mesh_x = unitless_data_x.reshape((self.frames, self.x_length, self.x_length))
        self.data_mesh_y = unitless_data_y.reshape((self.frames, self.x_length, self.x_length))
        self.data_mesh_z = unitless_data_z.reshape((self.frames, self.x_length, self.x_length))

        self.p = [
            self.axes[0].pcolormesh(self.x_mesh, self.y_mesh, self.data_mesh_x[0], cmap = "bwr", vmin=self.Min, vmax=self.Max),
            self.axes[1].pcolormesh(self.x_mesh, self.y_mesh, self.data_mesh_y[0], cmap = "bwr", vmin=self.Min, vmax=self.Max),
            self.axes[2].pcolormesh(self.x_mesh, self.y_mesh, self.data_mesh_z[0], cmap = "bwr", vmin=self.Min, vmax=self.Max)
            ]
        cbar = fig.colorbar(self.p[2], ax=self.axes, fraction=0.02, pad=0.01)
        components = ["x","y","z"]

        for i in range(3):
            title = f"$\\frac{{{"\\delta " + self.object.variable_name + "_" + components[i]}}}{{\\langle {self.object.variable_name}\\rangle}}$"
            self.axes[i].set_title(r'{}'.format(title), fontsize=20)
            self.axes[i].set_xlabel(r'$x/d_p$',fontsize=12)
            self.axes[i].set_ylabel(r'$y/d_p$', rotation=0, fontsize=12)

        self.timelabel = self.axes[2].text(0.98, 1.02, "",transform=self.axes[2].transAxes, fontsize = 16)
        

        anim = animation.FuncAnimation(fig, self.unitless_update, frames = self.frames, interval = 20)

        writer = FFMpegWriter(fps=5)
        anim.save(self.object.name, writer = writer)
        plt.close()

    def unitless_update(self,frame):
        for i in range(3):
            self.p[i].remove()
        self.timelabel.set_text(f"{self.time[frame]:.1f}s")
        self.p = [
            self.axes[0].pcolormesh(self.x_mesh, self.y_mesh, self.data_mesh_x[frame], cmap = "bwr", vmin=self.Min, vmax=self.Max),
            self.axes[1].pcolormesh(self.x_mesh, self.y_mesh, self.data_mesh_y[frame], cmap = "bwr", vmin=self.Min, vmax=self.Max),
            self.axes[2].pcolormesh(self.x_mesh, self.y_mesh, self.data_mesh_z[frame], cmap = "bwr", vmin=self.Min, vmax=self.Max)
        ]
        return self.p

    def animation_unit(self):
        fig, self.axes = plt.subplots(1,3, figsize=(26,8))
        fig.tight_layout(pad=4.0)

        self.Min = np.min(np.array([self.data_x,self.data_y,self.data_z]).flatten()) / self.object.unit
        self.Max = np.max(np.array([self.data_x,self.data_y,self.data_z]).flatten()) / self.object.unit

        print(self.Min)
        print(self.Max)

        if abs(self.Min) > abs(self.Max):
            self.Max = -self.Min
        else:
            self.Min = -self.Max

        self.data_mesh_x = self.data_x.reshape((self.frames, self.x_length, self.x_length)) / self.object.unit
        self.data_mesh_y = self.data_y.reshape((self.frames, self.x_length, self.x_length)) / self.object.unit
        self.data_mesh_z = self.data_z.reshape((self.frames, self.x_length, self.x_length)) / self.object.unit

        self.p = [
            self.axes[0].pcolormesh(self.x_mesh, self.y_mesh, self.data_mesh_x[0], cmap = "bwr", vmin=self.Min, vmax=self.Max),
            self.axes[1].pcolormesh(self.x_mesh, self.y_mesh, self.data_mesh_y[0], cmap = "bwr", vmin=self.Min, vmax=self.Max),
            self.axes[2].pcolormesh(self.x_mesh, self.y_mesh, self.data_mesh_z[0], cmap = "bwr", vmin=self.Min, vmax=self.Max)
            ]
        cbar = fig.colorbar(self.p[2], ax=self.axes, fraction=0.02, pad=0.01)
        cbar.set_label(r'{}'.format(self.object.unit_name), rotation = 0, fontsize=12, va="top")
        cbar.ax.xaxis.set_label_position("top")
        components = ["x","y","z"]

        for i in range(3):
            title = f"$\\delta {self.object.variable_name}_{components[i]}$"
            self.axes[i].set_title(r'{}'.format(title), fontsize=20)
            self.axes[i].set_xlabel(r'$x/d_p$',fontsize=12)
            self.axes[i].set_ylabel(r'$y/d_p$', rotation=0, fontsize=12)
        
        self.timelabel = self.axes[2].text(0.98, 1.02, "",transform=self.axes[2].transAxes, fontsize = 16)

        anim = animation.FuncAnimation(fig, self.unit_update, frames = self.frames, interval = 20)

        writer = FFMpegWriter(fps=5)
        anim.save(self.object.name, writer = writer)
        plt.close()

    def unit_update(self,frame):
        for i in range(3):
            self.p[i].remove()
        self.timelabel.set_text(f"{self.time[frame]:.1f}s")
        self.p = [
            self.axes[0].pcolormesh(self.x_mesh, self.y_mesh, self.data_mesh_x[frame], cmap = "bwr", vmin=self.Min, vmax=self.Max),
            self.axes[1].pcolormesh(self.x_mesh, self.y_mesh, self.data_mesh_y[frame], cmap = "bwr", vmin=self.Min, vmax=self.Max),
            self.axes[2].pcolormesh(self.x_mesh, self.y_mesh, self.data_mesh_z[frame], cmap = "bwr", vmin=self.Min, vmax=self.Max)
        ]
        return self.p
    
















