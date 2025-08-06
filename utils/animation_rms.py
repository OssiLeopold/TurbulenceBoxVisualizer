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
        if object.variable == "residual":
            self.animation_residual()
        elif object.component in ["x","y","z","magnitude"]:
            self.animation_one()
        else:
            self.animation_all()

    def animation_residual(self):
        mu_0 = 4 * np.pi * 10**(-7)
        m_p = 1.67262192595 * 10**(-27)

        shm_time = shared_memory.SharedMemory(name=self.object.time)
        self.time = np.ndarray(self.object.time_shape, dtype=self.object.time_dtype, buffer=shm_time.buf)
        self.frames = len(self.time)

        shm_rho = shared_memory.SharedMemory(name=self.object.memory_space["proton/vg_rho"])
        shm_Bx = shared_memory.SharedMemory(name=self.object.memory_space["vg_b_vol" + "x"])
        shm_By = shared_memory.SharedMemory(name=self.object.memory_space["vg_b_vol" + "y"])
        shm_Bz = shared_memory.SharedMemory(name=self.object.memory_space["vg_b_vol" + "z"])
        rho = np.ndarray(self.object.shape["vg_b_vol"], dtype = self.object.dtype, buffer = shm_rho.buf) * m_p
        bx = np.ndarray(self.object.shape["vg_b_vol"], dtype = self.object.dtype, buffer = shm_Bx.buf) / np.sqrt(mu_0 * rho)
        by = np.ndarray(self.object.shape["vg_b_vol"], dtype = self.object.dtype, buffer = shm_By.buf) / np.sqrt(mu_0 * rho)
        bz = np.ndarray(self.object.shape["vg_b_vol"], dtype = self.object.dtype, buffer = shm_Bz.buf) / np.sqrt(mu_0 * rho)

        shm_vx = shared_memory.SharedMemory(name=self.object.memory_space["proton/vg_v" + "x"])
        shm_vy = shared_memory.SharedMemory(name=self.object.memory_space["proton/vg_v" + "y"])
        shm_vz = shared_memory.SharedMemory(name=self.object.memory_space["proton/vg_v" + "z"])
        vx = np.ndarray(self.object.shape["proton/vg_v"], dtype = self.object.dtype, buffer = shm_vx.buf)
        vy = np.ndarray(self.object.shape["proton/vg_v"], dtype = self.object.dtype, buffer = shm_vy.buf)
        vz = np.ndarray(self.object.shape["proton/vg_v"], dtype = self.object.dtype, buffer = shm_vz.buf)

        shm_J = shared_memory.SharedMemory(name=self.object.memory_space["vg_j"])
        J_z = np.ndarray(self.object.shape["vg_j"], dtype = self.object.dtype, buffer = shm_J.buf)

        shm_T = shared_memory.SharedMemory(name=self.object.memory_space["vg_ttensor"])
        T = np.ndarray(self.object.shape["vg_ttensor"], dtype = self.object.dtype, buffer = shm_T.buf)
        T = np.swapaxes(T,1,2)
        print(T[1])

        self.J_z_rms = np.sqrt(np.mean((J_z**2), axis=1) - np.mean((J_z), axis=1)**2)

        b_perp = np.sqrt(bx**2 + by**2)
        self.b_parr_rms = np.sqrt(np.mean((bz**2), axis=1) - np.mean((bz), axis=1)**2)
        self.b_perp_rms = np.sqrt(np.mean((b_perp**2), axis=1) - np.mean((b_perp), axis=1)**2)

        v_perp = np.sqrt(vx**2 + vy**2)
        self.v_parr_rms = np.sqrt(np.mean((vz**2), axis=1) - np.mean((vz), axis=1)**2)
        self.v_perp_rms = np.sqrt(np.mean((v_perp**2), axis=1) - np.mean((v_perp), axis=1)**2)
        
        self.T_perp = np.mean(1/2 * (T[:,0,:] + T[:,1,:]) / (500 * 1e3),axis=1) 
        self.T_parr = np.mean(T[:,2,:] / (500 * 1e3),axis=1) 
        self.A = np.mean(1/2 * (T[:,0,:] + T[:,1,:])/T[:,2,:],axis=1)

        self.sigma_c = np.mean((2 * (vx*bx + vy*by + vz*bz) / (vx**2+vy**2+vz**2 + bx**2+by**2+bz**2)), axis = 1)
        self.sigma_r = np.mean(((vx**2+vy**2+vz**2 - bx**2-by**2-bz**2) / (vx**2+vy**2+vz**2 + bx**2+by**2+bz**2)), axis = 1)

        fig, self.axes = plt.subplots(2,2, figsize = (10,10))
        self.axes = self.axes.flatten()

        J_z_rms_label = f"$J_\\parallel^{{rms}}$"
        b_perp_label = f"$b_\\perp^{{rms}}$"
        b_parr_label = f"$b_\\parallel{{rms}}$"
        v_perp_label = f"$v_\\perp^{{rms}}$"
        v_parr_label = f"$v_\\parallel^{{rms}}$"
        T_perp_label = f"$\\langle\\frac{{T_\\perp}}{{T_0}}\\rangle$"
        T_parr_label = f"$\\langle\\frac{{T_\\parallel}}{{T_0}}\\rangle$"
        A_label = f"$\\langle A_p\\rangle$"
        label_c = f"$\\langle\\sigma_c\\rangle$"
        label_r = f"$\\langle\\sigma_r\\rangle$"
        
        self.axes[0].plot(self.time, self.J_z_rms, label = r'{}'.format(J_z_rms_label))
        
        self.axes[1].plot(self.time, self.b_perp_rms, label = r'{}'.format(b_perp_label))
        self.axes[1].plot(self.time, self.b_parr_rms, label = r'{}'.format(b_parr_label))
        self.axes[1].plot(self.time, self.v_perp_rms, label = r'{}'.format(v_perp_label))
        self.axes[1].plot(self.time, self.v_parr_rms, label = r'{}'.format(v_parr_label))
        
        self.axes[2].plot(self.time, self.T_perp, label = r'{}'.format(T_perp_label))
        self.axes[2].plot(self.time, self.T_parr, label = r'{}'.format(T_parr_label))
        self.axes[2].plot(self.time, self.A, label = r'{}'.format(A_label))
        
        self.axes[3].plot(self.time, self.sigma_c, label = r'{}'.format(label_c))
        self.axes[3].plot(self.time, self.sigma_r, label = r'{}'.format(label_r))

        self.axes[0].legend()
        self.axes[1].legend()
        self.axes[2].legend()
        self.axes[3].legend()

        #anim = animation.FuncAnimation(fig, self.update_residual, frames = self.frames + 1, interval = 20)
        
        #writer = FFMpegWriter(fps=5)
        self.object.name = self.object.name[0:-3] + "jpg"

        fig.savefig(self.object.name)

        plt.close()

    def update_residual(self, frame):
        self.p[0][0].set_data(self.time[:frame], self.b_perp_rms[:frame])
        self.p[1][0].set_data(self.time[:frame], self.b_parr_rms[:frame])
        self.p[2][0].set_data(self.time[:frame], self.v_perp_rms[:frame])
        self.p[3][0].set_data(self.time[:frame], self.v_parr_rms[:frame])
        self.p[4][0].set_data(self.time[:frame], self.sigma_c[:frame])
        self.p[5][0].set_data(self.time[:frame], self.sigma_r[:frame])

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

