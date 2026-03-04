import os
from configparser import ConfigParser
import analysator as pt
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from multiprocessing import shared_memory
from matplotlib.colors import LogNorm
from matplotlib import animation
from matplotlib.animation import FFMpegWriter
from matplotlib.widgets import SpanSelector

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
        self.memory_space = object.memory_space
        
        shm_time = shared_memory.SharedMemory(name=self.memory_space["timepass"]["address"])
        self.time = np.ndarray(self.memory_space["timepass"]["shape"], dtype=self.memory_space["timepass"]["dtype"], buffer=shm_time.buf)
        self.frames = len(self.time)
        
        self.vlsvobj = pt.vlsvfile.VlsvReader(object.bulkpath + "bulk.0000000.vlsv")
        self.cellids = self.vlsvobj.read_variable("CellID")

        self.x_length = int(self.vlsvobj.read_parameter("xcells_ini"))
        self.x = np.array(self.vlsvobj.get_cell_coordinates(np.sort(self.cellids))).T[0]
        
        if object.fourier_type == "1D":
            self.animation_1D_PSD()
        elif object.fourier_type == "2D":
            self.animation_2D_PSD()
        elif object.fourier_type == "window":
            self.window()

    def animation_1D_PSD(self):
        fig, self.ax = plt.subplots()

        mem_x = self.memory_space[self.object.variable + "x"]
        mem_y = self.memory_space[self.object.variable + "y"]

        shm_x = shared_memory.SharedMemory(name=mem_x["address"])
        shm_y = shared_memory.SharedMemory(name=mem_y["address"])
        
        data_x = np.ndarray(mem_x["shape"], dtype=mem_x["dtype"], buffer=shm_x.buf)
        data_y = np.ndarray(mem_y["shape"], dtype=mem_y["dtype"], buffer=shm_y.buf)

        # Reshape raw data into mesh
        data_x_mesh = data_x.reshape((self.frames, self.x_length, self.x_length))
        data_y_mesh = data_y.reshape((self.frames, self.x_length, self.x_length))

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
        K_perp = K_perp.flatten()

        del k_xy, KX, KY

        k_min = min(K_perp)
        k_max = max(K_perp)
        kbins = np.arange(k_min, k_max, k_max/512)
        print(len(kbins))

        self.PSD_1D_perp = np.empty((self.frames, len(kbins)-1))
        
        #for frame in range(self.frames):
        #    self.PSD_1D_perp[frame] = np.bincount(bin_idx, weights = PSD_2D_perp[frame].ravel(), minlength=nbins)

        for frame in range(self.frames):
            value = PSD_2D_perp[frame].flatten()
            self.PSD_1D_perp[frame],_,_ = sp.stats.binned_statistic(K_perp, value, statistic="mean",bins=kbins)
            self.PSD_1D_perp[frame] *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

        del PSD_2D_perp

        print(self.PSD_1D_perp)

        Min = min(self.PSD_1D_perp.flatten())
        Max = max(self.PSD_1D_perp.flatten())

        self.p = [self.ax.plot([], [])]

        prot_plas_freq = np.sqrt(1e6 * (1.602176634 * 10**(-19))**2 / (8.8541878128 * 10**(-12) * 1.67262192595 * 10**(-27)))
        self.dp = 299792458 / prot_plas_freq
        self.k_vals = 0.5 * (kbins[1:] + kbins[:-1]) * self.dp
        """ a = Max * (10**(-6) * self.dp)**2
        b = Max * (10**(-6) * self.dp)**(5/3)
        c = Max * (10**(-6) * self.dp)**3

        self.p.append(self.ax.plot(self.k_vals, a * (self.k_vals)**(-2), label = "k**(-2)"))
        self.p.append(self.ax.plot(self.k_vals, b * (self.k_vals)**(-5/3), label = "k**(-5/3)"))
        self.p.append(self.ax.plot(self.k_vals, c * (self.k_vals)**(-3), label = "k**(-3)"))
 """
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")

        self.ax.set_ylim(Min, Max)
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

        mem_x = self.memory_space[self.object.variable + "x"]
        mem_y = self.memory_space[self.object.variable + "y"]

        shm_x = shared_memory.SharedMemory(name=mem_x["address"])
        shm_y = shared_memory.SharedMemory(name=mem_y["address"])
        
        data_x = np.ndarray(mem_x["shape"], dtype=mem_x["dtype"], buffer=shm_x.buf)
        data_y = np.ndarray(mem_y["shape"], dtype=mem_y["dtype"], buffer=shm_y.buf)

        # Reshape raw data into mesh
        data_x_mesh = data_x.reshape((self.frames, self.x_length, self.x_length))
        data_y_mesh = data_y.reshape((self.frames, self.x_length, self.x_length))

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

    def window(self):
        self.fig, self.ax = plt.subplots(1,2, figsize=(12,5))

        mem_x = self.memory_space[self.object.variable + "x"]
        mem_y = self.memory_space[self.object.variable + "y"]

        shm_x = shared_memory.SharedMemory(name=mem_x["address"])
        shm_y = shared_memory.SharedMemory(name=mem_y["address"])
        
        data_x = np.ndarray(mem_x["shape"], dtype=mem_x["dtype"], buffer=shm_x.buf)
        data_y = np.ndarray(mem_y["shape"], dtype=mem_y["dtype"], buffer=shm_y.buf)

        # Reshape raw data into mesh
        data_x_mesh = data_x.reshape((self.frames, self.x_length, self.x_length))
        data_y_mesh = data_y.reshape((self.frames, self.x_length, self.x_length))

        # Fourier transfrom meshshes
        data_x_mesh_ft = np.abs(sp.fft.fftshift(sp.fft.fft2(data_x_mesh, workers = 8, axes=(-2, -1)), axes = (-2,-1)))
        data_y_mesh_ft = np.abs(sp.fft.fftshift(sp.fft.fft2(data_y_mesh, workers = 8, axes=(-2, -1)), axes = (-2,-1)))

        # |F_perp|**2 = |F_x|**2 + |F_y|**2
        PSD_2D_perp = data_x_mesh_ft**2 + data_y_mesh_ft**2

        del data_x_mesh_ft, data_y_mesh_ft
        
        dx = np.diff(self.x[0:self.x_length])[0]

        k_xy = 2*np.pi * sp.fft.fftshift(sp.fft.fftfreq(self.x_length, dx))
        KX, KY = np.meshgrid(k_xy, k_xy)
        K_perp = np.sqrt(KX**2 + KY**2)
        K_perp = K_perp.flatten()

        print(K_perp)
        print(len(K_perp))

        del k_xy, KX, KY

        k_min = sorted(K_perp)[1]
        k_max = sorted(K_perp)[-1]

        print(k_min)
        print(k_max)

        #kbins = np.concatenate([np.array([0]), np.logspace(np.log10(k_min*0.1), np.log10(k_max), num=31)])
        kbins = np.logspace(np.log10(k_min), np.log10(k_max), num=32, base=10)
        print(kbins)

        #kbins = np.arange(k_min, k_max, 16*k_max/self.x_length)

        print(kbins[0])
        print(kbins[-1])

        print(len(kbins))

        PSD_1D_perp_raw = np.empty((self.frames, len(kbins)-1))

        for frame in range(self.frames):
            value = PSD_2D_perp[frame].flatten()
            PSD_1D_perp_raw[frame],_,_ = sp.stats.binned_statistic(K_perp, value, statistic="sum", bins=kbins)
            zeros = PSD_1D_perp_raw[frame] == 0.0
            non_zeros = zeros == False
            PSD_1D_perp_raw[frame][0:sum(non_zeros)] = PSD_1D_perp_raw[frame][non_zeros]

        self.PSD_1D_perp = PSD_1D_perp_raw[:,0:sum(non_zeros)]

        del PSD_2D_perp

        print(self.PSD_1D_perp)

        Min = min(self.PSD_1D_perp.flatten())
        Max = max(self.PSD_1D_perp.flatten())

        prot_plas_freq = np.sqrt(5e6 * (1.602176634 * 10**(-19))**2 / (8.8541878128 * 10**(-12) * 1.67262192595 * 10**(-27)))
        self.dp = 299792458 / prot_plas_freq
        self.k_vals = 0.5 * (kbins[1:] + kbins[:-1]) * self.dp
        self.k_vals = self.k_vals[non_zeros]

        delta_k = [2,3]
        self.gradients = np.empty((self.frames, len(delta_k), len(self.k_vals)))

        for i in range(self.frames):
            for j, dk in enumerate(delta_k):
                for k, k_val in enumerate(self.k_vals):
                    window = (self.k_vals >= k_val) & (self.k_vals <= k_val * dk)
                    k_window = self.k_vals[window]
                    PSD_window = self.PSD_1D_perp[i][window]
                    if len(k_window) > 1:
                        parameters, covariance = sp.optimize.curve_fit(self.fit, np.log10(k_window), np.log10(PSD_window), maxfev=10000)
                        #gradient = (np.log10(PSD_window[-1]) - np.log10(PSD_window[0])) / (np.log10(k_window[-1]) - np.log10(k_window[0]))
                        self.gradients[i][j][k] = parameters[1]
                    else:
                        self.gradients[i][j][k] = 0
        
        self.p = [self.ax[0].plot([1],[1]), self.ax[1].plot([1],[1], label=f"{delta_k[0]}"), self.ax[1].plot([1],[1], label=f"{delta_k[1]}")]
 
        self.ax[1].set_ylim(-3,1)
        self.ax[1].set_xscale("log")
        self.ax[1].set_xlim(self.k_vals[0]*0.9,self.k_vals[-1])
        self.ax[1].grid(True, which='both', linestyle='--', alpha=0.4)
        self.ax[1].legend()

        self.ax[0].set_xscale("log")
        self.ax[0].set_yscale("log")

        self.ax[0].set_ylim(1e-15,Max)
        self.ax[0].set_xlim(self.k_vals[0]*0.9,self.k_vals[-1])
        self.ax[0].grid(True, which='both', linestyle='--', alpha=0.4)

        xlabel = f"$k_{{\\perp}}d_p$"
        self.ax[0].set_xlabel(r"{}".format(xlabel))
        ylabel = f"$P(k_{{\\perp}})$"
        self.ax[0].set_ylabel(r"{}".format(ylabel))


        anim = animation.FuncAnimation(self.fig, self.update_window, self.frames, interval=20)
        writer = FFMpegWriter(fps=5)
        anim.save("window_test.mp4", writer=writer)

        #self.fig.savefig(f"mahti_sim001_window_100.jpg")

    def fit(self, log_k, A, B):
        return A + B * log_k    

    def update_window(self, frame):
        self.p[0][0].set_data(self.k_vals, self.PSD_1D_perp[frame])
        self.p[1][0].set_data(self.k_vals, self.gradients[frame][0])
        self.p[2][0].set_data(self.k_vals, self.gradients[frame][1])

        return self.p