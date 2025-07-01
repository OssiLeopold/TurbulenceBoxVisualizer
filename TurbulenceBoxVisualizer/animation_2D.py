import os
import analysator as pt
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import shared_memory
from matplotlib import animation
from matplotlib.animation import FFMpegWriter

# Telling FFMpegWriter the location of FFMpeg
plt.rcParams['animation.ffmpeg_path'] = "/home/rxelmer/Documents/turso/appl_local/ffmpeg/bin/ffmpeg"

#enabling use of latex
os.environ['PATH']='/home/rxelmer/Documents/turso/appl_local/tex-basic/texlive/2023/bin/x86_64-linux:'+ os.environ['PATH'] 
os.environ['PTNOLATEX']='1'

class Animation2D():
    def __init__(self, object):
        self.object = object
        shm = shared_memory.SharedMemory(name=object.memory_space)
        self.data = np.ndarray(object.shape, dtype=object.dtype, buffer=shm.buf)

        prot_plas_freq = np.sqrt(1e6 * (1.602176634 * 10**(-19))**2 / (8.8541878128 * 10**(-12) * 1.67262192595 * 10**(-27)))
        dp = 299792458 / prot_plas_freq

        self.vlsvobj = pt.vlsvfile.VlsvReader(object.bulkpath + "bulk.0000000.vlsv")
        self.cellids = self.vlsvobj.read_variable("CellID")

        self.x_length = self.vlsvobj.read_parameter("xcells_ini")
        x = np.array([self.vlsvobj.get_cell_coordinates(coord)[0] for coord in np.sort(self.cellids)])
        y = np.array([self.vlsvobj.get_cell_coordinates(coord)[1] for coord in np.sort(self.cellids)])
        self.x_mesh = x.reshape(-1,self.x_length) / dp
        self.y_mesh = y.reshape(-1,self.x_length) / dp

        self.frames = len(self.data)

        if object.unitless == True:
            self.animation_unitless()
        else:
            self.animation_unit()

    def animation_unitless(self):
        shm_norm = shared_memory.SharedMemory(name=self.object.memory_space_norm)
        mag = np.ndarray(self.object.shape, dtype=self.object.dtype, buffer=shm_norm.buf)
        mag_average = np.empty(self.frames)

        for i in range(self.frames):
            mag_average[i] = np.average(mag[i])

        unitless_data = np.empty(self.object.shape)
        for i in range(self.frames):
            unitless_data[i] = self.data[i] / mag_average[i]

        fig, self.ax = plt.subplots()

        self.Min = round(min(unitless_data.flatten()), 15)
        self.Max = round(max(unitless_data.flatten()), 15)

        if abs(self.Min) > abs(self.Max):
            self.Max = -self.Min
        else:
            self.Min = -self.Max

        self.data_mesh = []
        for i in range(self.frames):
            self.data_mesh.append(unitless_data[i].reshape(-1, self.x_length))

        self.p = [
            self.ax.pcolormesh(self.x_mesh, self.y_mesh, self.data_mesh[0], cmap = "bwr", vmin=self.Min, vmax=self.Max)]
        cbar = fig.colorbar(self.p[0])

        title = f"$\\frac{{{"\\delta " + self.object.variable_name + "_" + self.object.component}}}{{\\langle {self.object.variable_name}\\rangle}}$"
        self.ax.set_title(r'{}'.format(title), fontsize=16)
        self.ax.set_xlabel(r'$x/d_p$',fontsize=12)
        self.ax.set_ylabel(r'$y/d_p$', rotation=0, fontsize=12)
        

        anim = animation.FuncAnimation(fig, self.unitless_update, frames = self.frames, interval = 20)

        writer = FFMpegWriter(fps=5)
        anim.save(self.object.name, writer = writer)
        plt.close()

    def unitless_update(self,frame):
        self.p[0].remove()
        self.p = [
            self.ax.pcolormesh(self.x_mesh, self.y_mesh, self.data_mesh[frame], cmap = "bwr", vmin=self.Min, vmax=self.Max)]
        return self.p[0]

    def animation_unit(self):
        fig, self.ax = plt.subplots()

        self.Min = round(min(self.data.flatten())/self.object.unit, 10)
        self.Max = round(max(self.data.flatten())/self.object.unit, 10)

        if abs(self.Min) > abs(self.Max):
            self.Max = -self.Min
        else:
            self.Min = -self.Max

        self.data_mesh = []
        for i in range(self.frames):
            self.data_mesh.append(self.data[i].reshape(-1, self.x_length))

        self.p = [
            self.ax.pcolormesh(self.x_mesh, self.y_mesh, self.data_mesh[0]/self.object.unit, cmap = "bwr", vmin=self.Min, vmax=self.Max)]
        cbar = fig.colorbar(self.p[0])
        
        if self.object.component != "pass":
            self.ax.set_title(r'$\delta {}$'.format(self.object.variable_name + "_" + self.object.component), fontsize=16)
        else:
            self.ax.set_title(r'$\delta {}$'.format(self.object.variable_name), fontsize=16)
        cbar.set_label(r'{}'.format(self.object.unit_name), rotation = 0, fontsize=12, va="top")
        self.ax.set_xlabel(r'$x/d_p$',fontsize=12)
        self.ax.set_ylabel(r'$y/d_p$', rotation=0, fontsize=12)
        

        anim = animation.FuncAnimation(fig, self.unit_update, frames = self.frames, interval = 20)

        writer = FFMpegWriter(fps=5)
        anim.save(self.object.name, writer = writer)
        plt.close()

    def unit_update(self,frame):
        self.p[0].remove()
        self.p = [
            self.ax.pcolormesh(self.x_mesh, self.y_mesh, self.data_mesh[frame]/self.object.unit, cmap = "bwr", vmin=self.Min, vmax=self.Max)]
        return self.p[0]