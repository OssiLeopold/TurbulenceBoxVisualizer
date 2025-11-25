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

MU_0 = 4 * np.pi * 10**(-7)
M_P = 1.67262192595 * 10**(-27)

class AnimationSigma():
    def __init__(self, object):
        self.object = object
        memory_space = object.memory_space
        
        shm_time = shared_memory.SharedMemory(name=memory_space["timepass"]["address"])
        self.time = np.ndarray(memory_space["timepass"]["shape"], dtype=memory_space["timepass"]["dtype"], buffer=shm_time.buf)
        self.frames = len(self.time)
        
        self.vlsvobj = pt.vlsvfile.VlsvReader(object.bulkpath + "bulk.0000000.vlsv")
        self.cellids = self.vlsvobj.read_variable("CellID")

        prot_plas_freq = np.sqrt(1e6 * (1.602176634 * 10**(-19))**2 / (8.8541878128 * 10**(-12) * 1.67262192595 * 10**(-27)))
        dp = 299792458 / prot_plas_freq

        self.x_length = int(self.vlsvobj.read_parameter("xcells_ini"))
        coords = np.array(self.vlsvobj.get_cell_coordinates(np.sort(self.cellids))).T
        self.x = coords[0]
        self.y = coords[1]
        self.x_mesh = self.x.reshape(-1,self.x_length) / dp
        self.y_mesh = self.y.reshape(-1,self.x_length) / dp
        
        mem_rho= memory_space["proton/vg_rho" + "pass"]
        mem_Bx = memory_space["vg_b_vol" + "x"]
        mem_By = memory_space["vg_b_vol" + "y"]
        mem_Bz = memory_space["vg_b_vol" + "z"]
        mem_vx = memory_space["proton/vg_v" + "x"]
        mem_vy = memory_space["proton/vg_v" + "y"]
        mem_vz = memory_space["proton/vg_v" + "z"]

        shm_rho = shared_memory.SharedMemory(name=mem_rho["address"])
        shm_Bx = shared_memory.SharedMemory(name=mem_Bx["address"])
        shm_By = shared_memory.SharedMemory(name=mem_By["address"])
        shm_Bz = shared_memory.SharedMemory(name=mem_Bz["address"])
        shm_vx = shared_memory.SharedMemory(name=mem_vx["address"])
        shm_vy = shared_memory.SharedMemory(name=mem_vy["address"])
        shm_vz = shared_memory.SharedMemory(name=mem_vz["address"])

        rho = np.ndarray(mem_rho["shape"], dtype = mem_rho["dtype"], buffer = shm_rho.buf) * M_P
        self.Bx = np.ndarray(mem_Bx["shape"], dtype = mem_Bx["dtype"], buffer = shm_Bx.buf)
        self.By = np.ndarray(mem_By["shape"], dtype = mem_By["dtype"], buffer = shm_By.buf)
        self.Bz = np.ndarray(mem_Bz["shape"], dtype = mem_Bz["dtype"], buffer = shm_Bz.buf)
        self.bx = self.Bx / np.sqrt(MU_0 * rho)
        self.by = self.By / np.sqrt(MU_0 * rho)
        self.bz = self.Bz / np.sqrt(MU_0 * rho)
        self.vx = np.ndarray(mem_vx["shape"], dtype = mem_vx["dtype"], buffer = shm_vx.buf)
        self.vy = np.ndarray(mem_vy["shape"], dtype = mem_vy["dtype"], buffer = shm_vy.buf)
        self.vz = np.ndarray(mem_vz["shape"], dtype = mem_vz["dtype"], buffer = shm_vz.buf)

        if object.animation_specific == "2D":
            self.animation_2D()
        elif object.animation_specific == "fourier":
            self.animation_fourier()

    def animation_2D(self):
        sigma_c = 2 * (self.vx*self.bx + self.vy*self.by + self.vz*self.bz) / (self.vx**2+self.vy**2+self.vz**2 + self.bx**2+self.by**2+self.bz**2)
        sigma_r = (self.vx**2+self.vy**2+self.vz**2 - self.bx**2-self.by**2-self.bz**2) / (self.vx**2+self.vy**2+self.vz**2 + self.bx**2+self.by**2+self.bz**2)
        self.sigma_c = sigma_c.reshape((self.frames, self.x_length, self.x_length))
        self.sigma_r = sigma_r.reshape((self.frames, self.x_length, self.x_length))

        del self.bx, self.by, self.bz

        dx = np.diff(self.x[0:self.x_length])[0]

        kx = 2*np.pi*np.fft.fftfreq(self.x_length, d=dx)
        ky = 2*np.pi*np.fft.fftfreq(self.x_length, d=dx)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2

        Bx_mesh = self.Bx.reshape((self.frames, self.x_length, self.x_length))
        By_mesh = self.By.reshape((self.frames, self.x_length, self.x_length))
        Bz_mesh = self.Bz.reshape((self.frames, self.x_length, self.x_length))

        Bx_hat = np.fft.fft2(Bx_mesh, axes=(-2, -1))
        By_hat = np.fft.fft2(By_mesh, axes=(-2, -1))

        num = 1j * (KY*Bx_hat - KX*By_hat)
        Az_hat = np.empty((self.frames, self.x_length, self.x_length), dtype=complex)
        mask = K2 != 0
        for frame in range(self.frames):
            Az_hat[frame][mask] = -num[frame][mask] / K2[mask]
            Az_hat[frame][~mask] = 0

        Az = np.fft.ifft2(Az_hat, axes = (-2, -1)).real
        self.sigma_m = Az * Bz_mesh

        self.Min_c = round(np.min(self.sigma_c.ravel()), 10)
        self.Max_c = round(np.max(self.sigma_c.ravel()), 10)

        if (np.abs(self.Min_c) > np.abs(self.Max_c)):
            self.Max_c = - self.Min_c
        else:
            self.Min_c = - self.Max_c
        
        self.Min_r = round(np.min(self.sigma_r.ravel()), 10)
        self.Max_r = round(np.max(self.sigma_r.ravel()), 10)

        if (np.abs(self.Min_r) > np.abs(self.Max_r)):
            self.Max_r = - self.Min_r
        else:
            self.Min_r = - self.Max_r

        self.Min_m = np.min(self.sigma_m.ravel())
        self.Max_m = np.max(self.sigma_m.ravel())

        if (np.abs(self.Min_m) > np.abs(self.Max_m)):
            self.Max_m = - self.Min_m
        else:
            self.Min_m = - self.Max_m

        fig, self.axes = plt.subplots(1, 3, figsize = (26, 8))
        
        self.p = [
            self.axes[0].pcolormesh(self.x_mesh, self.y_mesh, self.sigma_c[0], cmap = "bwr", vmin = self.Min_c, vmax = self.Max_c),
            self.axes[1].pcolormesh(self.x_mesh, self.y_mesh, self.sigma_r[0], cmap = "bwr", vmin = self.Min_r, vmax = self.Max_r),
            self.axes[2].pcolormesh(self.x_mesh, self.y_mesh, self.sigma_m[0], cmap = "bwr", vmin = self.Min_m, vmax = self.Max_m)
        ]

        anim = animation.FuncAnimation(fig, self.update_2D, self.frames, interval = 20)
        writer = FFMpegWriter(fps=5)
        anim.save("Animations/test/sigma_trial.mp4", writer=writer)
        plt.close()

    def update_2D(self, frame):
        for i in range(3):
            self.p[i].remove()
        #self.timelabel.set_text(f"{self.time[frame]:.1f}s")
        self.p = [
            self.axes[0].pcolormesh(self.x_mesh, self.y_mesh, self.sigma_c[frame], cmap = "bwr", vmin=self.Min_c, vmax=self.Max_c),
            self.axes[1].pcolormesh(self.x_mesh, self.y_mesh, self.sigma_r[frame], cmap = "bwr", vmin=self.Min_r, vmax=self.Max_r),
            self.axes[2].pcolormesh(self.x_mesh, self.y_mesh, self.sigma_m[frame], cmap = "bwr", vmin=self.Min_m, vmax=self.Max_m)
        ]
        return self.p

    def animation_fourier(self):
        bx_mesh = self.bx.reshape((self.frames, self.x_length, self.x_length))
        by_mesh = self.by.reshape((self.frames, self.x_length, self.x_length))
        Bz_mesh = self.bz.reshape((self.frames, self.x_length, self.x_length))

        vx_mesh = self.vx.reshape((self.frames, self.x_length, self.x_length))
        vy_mesh = self.vy.reshape((self.frames, self.x_length, self.x_length))
        vz_mesh = self.vz.reshape((self.frames, self.x_length, self.x_length))

        z_p_x = vx_mesh + bx_mesh
        z_p_y = vy_mesh + by_mesh
        z_p_z = vz_mesh + Bz_mesh

        z_m_x = vx_mesh - bx_mesh
        z_m_y = vy_mesh - by_mesh
        z_m_z = vz_mesh - Bz_mesh

        z_p_x_ft = sp.fft.fft2(z_p_x, workers = 8, axes=(-2, -1))
        z_p_y_ft = sp.fft.fft2(z_p_y, workers = 8, axes=(-2, -1))
        z_p_z_ft = sp.fft.fft2(z_p_z, workers = 8, axes=(-2, -1))

        z_m_x_ft = sp.fft.fft2(z_m_x, workers = 8, axes=(-2, -1))
        z_m_y_ft = sp.fft.fft2(z_m_y, workers = 8, axes=(-2, -1))
        z_m_z_ft = sp.fft.fft2(z_m_z, workers = 8, axes=(-2, -1))

        del z_p_x, z_p_y, z_p_z, z_m_x, z_m_y, z_m_z

        E_p = 1/4*(z_p_x_ft*np.conj(z_p_x_ft)+z_p_x_ft*np.conj(z_p_x_ft)+z_p_x_ft*np.conj(z_p_x_ft))
        E_m = 1/4*(z_m_x_ft*np.conj(z_m_x_ft)+z_m_y_ft*np.conj(z_m_y_ft)+z_m_z_ft*np.conj(z_m_z_ft))

        sigma_r_ft_2D = (z_p_x_ft*np.conj(z_m_x_ft) + z_p_y_ft*np.conj(z_m_y_ft) + z_p_z_ft*np.conj(z_m_z_ft)) / (E_p + E_m)
        sigma_c_ft_2D = 2 * (E_p - E_m) / (E_p + E_m)

        del E_p, E_m
        
        sigma_r_ft_2D = sigma_r_ft_2D.astype(np.float64)
        sigma_c_ft_2D = sigma_c_ft_2D.astype(np.float64)

        del z_p_x_ft, z_p_y_ft, z_p_z_ft, z_m_x_ft, z_m_y_ft, z_m_z_ft
        
        nbins = 500
        dx = np.diff(self.x[0:self.x_length])[0]

        k_xy = 2 * np.pi * sp.fft.fftfreq(self.x_length, dx)
        KX, KY = np.meshgrid(k_xy, k_xy)
        K_perp = np.sqrt(KX**2 + KY**2)

        del k_xy, KX, KY

        k_bin_edges = np.linspace(0, np.max(K_perp), num = nbins + 1) 
        bin_idx = np.digitize(K_perp.ravel(), k_bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, nbins - 1)

        self.sigma_r_ft_1D = np.empty((self.frames, nbins))
        self.sigma_c_ft_1D = np.empty((self.frames, nbins))
        
        for frame in range(self.frames):
            self.sigma_r_ft_1D[frame] = np.bincount(bin_idx, weights = sigma_r_ft_2D[frame].ravel(), minlength=nbins)
            self.sigma_c_ft_1D[frame] = np.bincount(bin_idx, weights = sigma_c_ft_2D[frame].ravel(), minlength=nbins)

        del sigma_r_ft_2D, sigma_c_ft_2D

        #self.sigma_r_ft_1D *= (dx*dx) / (nbins*nbins)
        #self.sigma_c_ft_1D *= (dx*dx) / (nbins*nbins)

        Min_r = min(self.sigma_r_ft_1D.flatten())
        Max_r = max(self.sigma_r_ft_1D.flatten())
        Min_c = min(self.sigma_c_ft_1D.flatten())
        Max_c = max(self.sigma_c_ft_1D.flatten())

        fig, self.axes = plt.subplots(1,2, figsize=(16,8))

        self.p = [self.axes[0].plot([], []), self.axes[1].plot([],[])]

        prot_plas_freq = np.sqrt(1e6 * (1.602176634 * 10**(-19))**2 / (8.8541878128 * 10**(-12) * 1.67262192595 * 10**(-27)))
        self.dp = 299792458 / prot_plas_freq
        self.k_vals = 0.5 * (k_bin_edges[1:] + k_bin_edges[:-1]) * self.dp

        self.axes[0].set_xscale("log")
        #self.axes[0].set_yscale("log")
        self.axes[1].set_xscale("log")
        #self.axes[1].set_yscale("log")

        self.axes[0].set_ylim(Min_r, Max_r)
        self.axes[0].set_xlim(self.k_vals[0],self.k_vals[-1])
        self.axes[1].set_ylim(Min_c, Max_c)
        self.axes[1].set_xlim(self.k_vals[0],self.k_vals[-1])

        """         xlabel = f"$k_{{\\perp}}d_p$"
        self.ax.set_xlabel(r"{}".format(xlabel))
        ylabel = f"$P(k_{{\\perp}})$"
        self.ax.set_ylabel(r"{}".format(ylabel))
        self.ax.legend() """

        #self.timelabel = self.ax.text(0.98, 1.02, "", transform=self.ax.transAxes)

        anim = animation.FuncAnimation(fig, self.update_fourier, frames = self.frames, interval = 20)
        
        writer = FFMpegWriter(fps=5)
        anim.save("Animations/test/sigma_fft_trial.mp4", writer = writer)
        plt.close()

    def update_fourier(self, frame):
        self.p[0][0].set_data(self.k_vals, self.sigma_r_ft_1D[frame])
        self.p[1][0].set_data(self.k_vals, self.sigma_c_ft_1D[frame])
        #self.timelabel.set_text(f"{self.time[frame]:.1f}s")
        return self.p
