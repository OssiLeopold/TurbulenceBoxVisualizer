import os
from configparser import ConfigParser
import analysator as pt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy as sp
from multiprocessing import shared_memory
from matplotlib.colors import LogNorm
from matplotlib import animation
from matplotlib.animation import FFMpegWriter

config = ConfigParser()
config.read(".TurbulenceBoxVisualizer.ini")

# Telling FFMpegWriter the location of FFMpeg
plt.rcParams['animation.ffmpeg_path'] = config["paths"]["ffmpeg_path"]

#enabling use of latex
os.environ['PATH']= config["paths"]["latex_path"] + os.environ['PATH'] 
os.environ['PTNOLATEX']='1'

class AnimationFourier():
    def __init__(self, object):
        self.object = object
        
        if object.component in ["x","y","z"]:
            shm = shared_memory.SharedMemory(name=object.memory_space)
            self.data = np.ndarray(object.shape, dtype=object.dtype, buffer=shm.buf)
            self.frames = len(self.data)
        elif object.component == "perp":
            shm_x = shared_memory.SharedMemory(name=object.memory_space["x"])
            shm_y = shared_memory.SharedMemory(name=object.memory_space["y"])
            self.data_x = np.ndarray(object.shape["x"], dtype=object.dtype, buffer=shm_x.buf)
            self.data_y = np.ndarray(object.shape["y"], dtype=object.dtype, buffer=shm_y.buf)
            self.frames = len(self.data_x)
        
        shm_time = shared_memory.SharedMemory(name=object.time)
        self.time = np.ndarray(object.time_shape, dtype=object.time_dtype, buffer=shm_time.buf)
        
        self.vlsvobj = pt.vlsvfile.VlsvReader(object.bulkpath + "bulk.0000211.vlsv")
        self.cellids = self.vlsvobj.read_variable("CellID")

        self.x_length = int(self.vlsvobj.read_parameter("xcells_ini"))
        self.x = np.array(self.vlsvobj.get_cell_coordinates(np.sort(self.cellids))).T[0]
        
        if object.fourier_type == "principle":
            self.animation_principle()
        elif object.fourier_type == "trace":
            self.animation_trace()
        elif object.fourier_type == "diag":
            self.animation_diag()
        elif object.fourier_type == "trace_diag":
            self.animation_trace_diag()
        elif object.fourier_type == "1D":
            self.animation_1D_PSD()
        elif object.fourier_type == "2D":
            self.animation_2D_PSD()

    def animation_1D_PSD(self):
        fig, self.ax = plt.subplots()

        # Reshape raw data into mesh
        data_x_mesh = self.data_x.reshape((self.frames, self.x_length, self.x_length))
        data_y_mesh = self.data_y.reshape((self.frames, self.x_length, self.x_length))

        # Fourier transfrom meshshes
        data_x_mesh_ft = np.abs(sp.fft.fft2(data_x_mesh, workers = 8, axes=(-2, -1)))
        data_y_mesh_ft = np.abs(sp.fft.fft2(data_y_mesh, workers = 8, axes=(-2, -1)))

        # |F_perp|**2 = |F_x|**2 + |F_y|**2
        PSD_2D_perp = data_x_mesh_ft**2 + data_y_mesh_ft**2

        del data_x_mesh_ft, data_y_mesh_ft
        
        nbins = 500
        dx = np.diff(self.x[0:self.x_length])[0]

        k_xy = 2 * np.pi * sp.fft.fftfreq(self.x_length, dx)
        KX, KY = np.meshgrid(k_xy, k_xy)
        K_perp = np.sqrt(KX**2 + KY**2)

        del k_xy, KX, KY

        k_bin_edges = np.linspace(0, np.max(K_perp), num = nbins + 1) 
        bin_idx = np.digitize(K_perp.ravel(), k_bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, nbins - 1)

        self.PSD_1D_perp = np.empty((self.frames, nbins))
        
        for frame in range(self.frames):
            self.PSD_1D_perp[frame] = np.bincount(bin_idx, weights = PSD_2D_perp[frame].ravel(), minlength=nbins)

        del PSD_2D_perp

        self.PSD_1D_perp *= (dx * dx) / (nbins * nbins)

        Min = min(self.PSD_1D_perp.flatten())
        Max = max(self.PSD_1D_perp.flatten())

        self.p = [self.ax.plot([], [])]

        prot_plas_freq = np.sqrt(1e6 * (1.602176634 * 10**(-19))**2 / (8.8541878128 * 10**(-12) * 1.67262192595 * 10**(-27)))
        self.dp = 299792458 / prot_plas_freq
        self.k_vals = 0.5 * (k_bin_edges[1:] + k_bin_edges[:-1]) * self.dp
        a = Max * (10**(-6) * self.dp)**2
        b = Max * (10**(-6) * self.dp)**(5/3)
        c = Max * (10**(-6) * self.dp)**3

        self.p.append(self.ax.plot(self.k_vals, a * (self.k_vals)**(-2), label = "k**(-2)"))
        self.p.append(self.ax.plot(self.k_vals, b * (self.k_vals)**(-5/3), label = "k**(-5/3)"))
        self.p.append(self.ax.plot(self.k_vals, c * (self.k_vals)**(-3), label = "k**(-3)"))

        self.ax.set_xscale("log")
        self.ax.set_yscale("log")

        self.ax.set_ylim(1e-10, Max*2)
        self.ax.set_xlim(self.k_vals[0]*0.9,self.k_vals[-1])

        xlabel = f"$k_{{\\perp}}d_p$"
        self.ax.set_xlabel(r"{}".format(xlabel))
        ylabel = f"$P(k_{{\\perp}})$"
        self.ax.set_ylabel(r"{}".format(ylabel))
        self.ax.legend()

        self.timelabel = self.ax.text(0.98, 1.02, "", transform=self.ax.transAxes)

        anim = animation.FuncAnimation(fig, self.update_1D_PSD, frames = self.frames, interval = 20)
        
        writer = FFMpegWriter(fps=5)
        anim.save(self.object.name, writer = writer)
        plt.close()

    def update_1D_PSD(self, frame):
        self.p[0][0].set_data(self.k_vals, self.PSD_1D_perp[frame])
        self.timelabel.set_text(f"{self.time[frame]:.1f}s")
        return self.p

    def animation_2D_PSD(self):
        fig, self.ax = plt.subplots()

        # Reshape raw data into mesh
        data_x_mesh = self.data_x.reshape((self.frames, self.x_length, self.x_length))
        data_y_mesh = self.data_y.reshape((self.frames, self.x_length, self.x_length))

        # Fourier transfrom meshshes
        data_x_mesh_ft = np.abs(sp.fft.fftshift(sp.fft.fft2(data_x_mesh, workers = 8, axes=(-2, -1)), axes = (-2,-1)))
        data_y_mesh_ft = np.abs(sp.fft.fftshift(sp.fft.fft2(data_y_mesh, workers = 8, axes=(-2, -1)), axes = (-2,-1)))

        # |F_perp|**2 = |F_x|**2 + |F_y|**2
        self.PSD_2D_perp = data_x_mesh_ft**2 + data_y_mesh_ft**2

        del data_x_mesh_ft, data_y_mesh_ft

        self.Min = np.min(self.PSD_2D_perp.ravel())
        self.Max = np.max(self.PSD_2D_perp.ravel())

        dx = np.diff(self.x[0:self.x_length])[0]

        k_xy = 2 * np.pi * sp.fft.fftshift(sp.fft.fftfreq(self.x_length, dx))
        self.KX, self.KY = np.meshgrid(k_xy, k_xy)

        self.p = [self.ax.pcolormesh(self.KX, self.KY, self.PSD_2D_perp[0], norm=LogNorm(vmin=1e-9, vmax=self.Max))]

        cbar = fig.colorbar(self.p[0])
        self.ax.set_xlim(-10e-6, 10e-6)
        self.ax.set_ylim(-10e-6, 10e-6)

        x_label = f"$k_{{x}}$"
        y_label = f"$k_{{y}}$"
        self.ax.set_xlabel(r'{}'.format(x_label))
        self.ax.set_ylabel(r'{}'.format(y_label))

        self.timelabel = self.ax.text(0.98, 1.02, "", transform=self.ax.transAxes)

        anim = animation.FuncAnimation(fig, self.update_2D_PSD, frames = self.frames, interval = 20)
        
        writer = FFMpegWriter(fps=5)
        anim.save(self.object.name, writer = writer)
        plt.close()

    def update_2D_PSD(self, frame):
        self.p[0].remove()
        self.p[0] = self.ax.pcolormesh(self.KX, self.KY, self.PSD_2D_perp[frame], norm=LogNorm(vmin=1e-9, vmax=self.Max))
        return self.p

    def animation_principle(self):
        fig, self.ax = plt.subplots()
        self.data_mesh = np.empty((self.frames, self.x_length, self.x_length))

        if self.object.fourier_direc == "x":
            for i in range(self.frames):
                self.data_mesh[i] = self.data[i].reshape(-1, self.x_length)
        elif self.object.fourier_direc == "y":
            for i in range(self.frames):
                self.data_mesh[i] = self.data[i].reshape(-1, self.x_length).T
        
        # Fourier transrom data_mesh at specified location
        self.data_mesh_ft = np.empty((self.frames, self.x_length//2))
        for i in range(self.frames):
            self.data_mesh_ft[i] = np.abs(sp.fft.fft(self.data_mesh[i][int(self.x_length * self.object.fourier_loc)])[:self.x_length//2])

        print(self.data_mesh_ft[0])

        self.spatial_freq = sp.fft.fftfreq(self.x_length, np.diff(self.x[0:self.x_length])[0])[:self.x_length//2]
        
        self.Min = round(min(self.data_mesh_ft.flatten()))
        self.Max = round(max(self.data_mesh_ft.flatten()))

        self.p = [self.ax.plot(2*np.pi * self.spatial_freq, self.data_mesh_ft[0])]

        a = self.Max * (10**(-6))**2
        b = self.Max * (10**(-6))**(5/3)

        spatial_freq_for_curve = np.delete(self.spatial_freq,0)
        self.p.append(self.ax.plot(2*np.pi * spatial_freq_for_curve[:self.x_length//2-1], a * (2*np.pi*spatial_freq_for_curve[:self.x_length//2-1])**(-2), label = "k**(-2)"))
        self.p.append(self.ax.plot(2*np.pi * spatial_freq_for_curve[:self.x_length//2-1], b * (2*np.pi*spatial_freq_for_curve[:self.x_length//2-1])**(-5/3), label = "k**(-5/3)"))

        self.ax.set_xlabel(r'$k$')
        ylabel = f"$\\frac{{{self.object.unit_name}^2}}{{k}}$"
        self.ax.set_ylabel(r'{}'.format(ylabel))

        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.grid(axis="y")
        self.timelabel = self.ax.text(0.98, 1.02, "",transform=self.ax.transAxes)

        self.ax.legend()

        anim = animation.FuncAnimation(fig, self.update_principle, frames = self.frames, interval = 20)
        
        writer = FFMpegWriter(fps=5)
        anim.save(self.object.name, writer = writer)
        plt.close()

    def update_principle(self,frame):
        self.p[0][0].set_data(2*np.pi * self.spatial_freq, self.data_mesh_ft[frame])
        self.timelabel.set_text(f"{self.time[frame]:.1f}s")
        return self.p
    
    def animation_trace(self):
        fig, self.ax = plt.subplots()
        self.data_mesh_x = np.empty((self.frames, self.x_length, self.x_length))
        self.data_mesh_y = np.empty((self.frames, self.x_length, self.x_length))

        for i in range(self.frames):
            self.data_mesh_x[i] = self.data[i].reshape(-1, self.x_length)

        for i in range(self.frames):
            self.data_mesh_y[i] = self.data[i].reshape(-1, self.x_length).T
        
        # Fourier transrom entire data_mesh
        self.data_mesh_x_ft = np.empty((self.frames, self.x_length//2), dtype="float64")
        self.data_mesh_y_ft = np.empty((self.frames, self.x_length//2), dtype="float64")
        self.data_mesh_trace = np.empty((self.frames, self.x_length//2), dtype="float64")

        for i in range(self.frames):
            self.data_mesh_x_ft[i] = np.abs(sp.fft.fft(self.data_mesh_x[i][int(self.x_length * self.object.fourier_loc_x)])[:self.x_length//2])

        for i in range(self.frames):
            self.data_mesh_y_ft[i] = np.abs(sp.fft.fft(self.data_mesh_y[i][int(self.x_length * self.object.fourier_loc_y)])[:self.x_length//2])

        self.data_mesh_trace = self.data_mesh_x_ft + self.data_mesh_y_ft

        self.spatial_freq = sp.fft.fftfreq(self.x_length, np.diff(self.x[0:self.x_length])[0])[:self.x_length//2]
        

        self.Min = round(min(self.data_mesh_trace.flatten()),15)
        self.Max = round(max(self.data_mesh_trace.flatten()),15)

        self.p = [self.ax.plot(2*np.pi * self.spatial_freq, self.data_mesh_trace[0])]

        a = self.Max * (10**(-6))**2
        b = self.Max * (10**(-6))**(5/3)

        spatial_freq_for_curve = np.delete(self.spatial_freq,0)
        self.p.append(self.ax.plot(2*np.pi * spatial_freq_for_curve[:self.x_length//2-1], a * (2*np.pi*spatial_freq_for_curve[:self.x_length//2-1])**(-2), label = "k**(-2)"))
        self.p.append(self.ax.plot(2*np.pi * spatial_freq_for_curve[:self.x_length//2-1], b * (2*np.pi*spatial_freq_for_curve[:self.x_length//2-1])**(-5/3), label = "k**(-5/3)"))

        self.ax.set_xlabel(r'$k$')
        ylabel = f"$\\frac{{{self.object.unit_name}^2}}{{k}}$"
        self.ax.set_ylabel(r'{}'.format(ylabel), rotation = 0)

        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.grid(axis="y")
        self.timelabel = self.ax.text(0.98, 1.02, "",transform=self.ax.transAxes)

        self.ax.legend()

        anim = animation.FuncAnimation(fig, self.update_trace, frames = self.frames, interval = 20)
        
        writer = FFMpegWriter(fps=5)
        anim.save(self.object.name, writer = writer)
        plt.close()

    def update_trace(self,frame):
        self.p[0][0].set_data(2*np.pi * self.spatial_freq, self.data_mesh_trace[frame])
        self.timelabel.set_text(f"{self.time[frame]:.1f}s")
        return self.p
    
    def animation_diag(self):
        fig, self.ax = plt.subplots()
        self.data_mesh = np.empty((self.frames, self.x_length, self.x_length))

        for i in range(self.frames):
            self.data_mesh[i] = self.data[i].reshape(-1, self.x_length)
        
        self.diag_mesh = np.empty((self.frames, self.x_length))
        
        if self.object.fourier_direc == 1:
            for i in range(self.frames):
                for j in range(self.x_length):
                    self.diag_mesh[i][j] = self.data_mesh[i][j][j]
        elif self.object.fourier_direc == 2:
            for i in range(self.frames):
                for j in range(self.x_length):
                    self.diag_mesh[i][j] = self.data_mesh[i][-j-1][j]

        self.data_mesh_ft = np.empty((self.frames, self.x_length//2), dtype="float64")
        for i in range(self.frames):
            self.data_mesh_ft[i] = np.abs(sp.fft.fft(self.diag_mesh[i])[:self.x_length//2])

        self.spatial_freq = sp.fft.fftfreq(self.x_length, np.sqrt(2) * np.diff(self.x[0:self.x_length])[0])[:self.x_length//2]

        self.Min = round(min(self.data_mesh_ft.flatten()))
        self.Max = round(max(self.data_mesh_ft.flatten()))

        self.p = [self.ax.plot(2*np.pi * self.spatial_freq, self.data_mesh_ft[0])]

        a = self.Max * (10**(-6))**2
        b = self.Max * (10**(-6))**(5/3)

        spatial_freq_for_curve = np.delete(self.spatial_freq,0)
        self.p.append(self.ax.plot(2*np.pi * spatial_freq_for_curve[:self.x_length//2-1], a * (2*np.pi*spatial_freq_for_curve[:self.x_length//2-1])**(-2), label = "k**(-2)"))
        self.p.append(self.ax.plot(2*np.pi * spatial_freq_for_curve[:self.x_length//2-1], b * (2*np.pi*spatial_freq_for_curve[:self.x_length//2-1])**(-5/3), label = "k**(-5/3)"))

        self.ax.set_xlabel(r'$k$')
        ylabel = f"$\\frac{{{self.object.unit_name}^2}}{{k}}$"
        self.ax.set_ylabel(r'{}'.format(ylabel))

        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.grid(axis="y")
        self.timelabel = self.ax.text(0.98, 1.02, "",transform=self.ax.transAxes)

        self.ax.legend()

        anim = animation.FuncAnimation(fig, self.update_diag, frames = self.frames, interval = 20)
        
        writer = FFMpegWriter(fps=5)
        anim.save(self.object.name, writer = writer)
        plt.close()

    def update_diag(self,frame):
        self.p[0][0].set_data(2*np.pi * self.spatial_freq, self.data_mesh_ft[frame])
        self.timelabel.set_text(f"{self.time[frame]:.1f}s")
        return self.p
    
    def animation_trace_diag(self):
        fig, self.ax = plt.subplots()
        self.data_mesh = np.empty((self.frames, self.x_length, self.x_length))

        for i in range(self.frames):
            self.data_mesh[i] = self.data[i].reshape(-1, self.x_length)
        
        self.diag_mesh_1 = np.empty((self.frames, self.x_length))
        self.diag_mesh_2 = np.empty((self.frames, self.x_length))
        
        for i in range(self.frames):
            for j in range(self.x_length):
                self.diag_mesh_1[i][j] = self.data_mesh[i][j][j]

        for i in range(self.frames):
            for j in range(self.x_length):
                self.diag_mesh_2[i][j] = self.data_mesh[i][-j-1][j]



        self.diag_mesh_1_ft = np.empty((self.frames, self.x_length//2), dtype="float64")
        for i in range(self.frames):
            self.diag_mesh_1_ft[i] = np.abs(sp.fft.fft(self.diag_mesh_1[i])[:self.x_length//2])

        self.diag_mesh_2_ft = np.empty((self.frames, self.x_length//2), dtype="float64")
        for i in range(self.frames):
            self.diag_mesh_2_ft[i] = np.abs(sp.fft.fft(self.diag_mesh_2[i])[:self.x_length//2])

        self.trace_mesh = self.diag_mesh_1_ft + self.diag_mesh_2_ft

        self.spatial_freq = sp.fft.fftfreq(self.x_length, np.sqrt(2) * np.diff(self.x[0:self.x_length])[0])[:self.x_length//2]

        self.Min = round(min(self.trace_mesh.flatten()))
        self.Max = round(max(self.trace_mesh.flatten()))

        self.p = [self.ax.plot(2*np.pi * self.spatial_freq, self.trace_mesh[0])]

        a = self.Max * (10**(-6))**2
        b = self.Max * (10**(-6))**(5/3)

        spatial_freq_for_curve = np.delete(self.spatial_freq,0)
        self.p.append(self.ax.plot(2*np.pi * spatial_freq_for_curve[:self.x_length//2-1], a * (2*np.pi*spatial_freq_for_curve[:self.x_length//2-1])**(-2), label = "k**(-2)"))
        self.p.append(self.ax.plot(2*np.pi * spatial_freq_for_curve[:self.x_length//2-1], b * (2*np.pi*spatial_freq_for_curve[:self.x_length//2-1])**(-5/3), label = "k**(-5/3)"))

        self.ax.set_xlabel(r'$k$')
        ylabel = f"$\\frac{{{self.object.unit_name}^2}}{{k}}$"
        self.ax.set_ylabel(r'{}'.format(ylabel))

        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.grid(axis="y")
        self.timelabel = self.ax.text(0.98, 1.02, "",transform=self.ax.transAxes)

        self.ax.legend()

        anim = animation.FuncAnimation(fig, self.update_trace_diag, frames = self.frames, interval = 20)
        
        writer = FFMpegWriter(fps=5)
        anim.save(self.object.name, writer = writer)
        plt.close()

    def update_trace_diag(self,frame):
        self.p[0][0].set_data(2*np.pi * self.spatial_freq, self.trace_mesh[frame])
        self.timelabel.set_text(f"{self.time[frame]:.1f}s")
        return self.p