import os
from configparser import ConfigParser
import analysator as pt
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import shared_memory

config = ConfigParser()
config.read(".TurbulenceBoxVisualizer.ini")

# Telling FFMpegWriter the location of FFMpeg
plt.rcParams['animation.ffmpeg_path'] = config["paths"]["ffmpeg_path"]

#enabling use of latex
os.environ['PATH']= config["paths"]["latex_path"] + os.environ['PATH'] 
os.environ['PTNOLATEX']='1'

class PlotFranci():
    def __init__(self, object):
        memory_space = object.memory_space

        mu_0 = 4 * np.pi * 10**(-7)
        m_p = 1.67262192595 * 10**(-27)
        T_0 = 500 * 10**3

        shm_time = shared_memory.SharedMemory(name=memory_space["timepass"]["address"])
        time = np.ndarray(memory_space["timepass"]["shape"], dtype=memory_space["timepass"]["dtype"], buffer=shm_time.buf)
        frames = len(time)

        mem_rho= memory_space["proton/vg_rho" + "pass"]
        mem_Bx = memory_space["vg_b_vol" + "x"]
        mem_By = memory_space["vg_b_vol" + "y"]
        mem_Bz = memory_space["vg_b_vol" + "z"]
        mem_vx = memory_space["proton/vg_v" + "x"]
        mem_vy = memory_space["proton/vg_v" + "y"]
        mem_vz = memory_space["proton/vg_v" + "z"]
        mem_Jz = memory_space["vg_j" + "z"]
        mem_T = memory_space["vg_ttensor" + "pass"]

        shm_rho = shared_memory.SharedMemory(name=mem_rho["address"])
        shm_Bx = shared_memory.SharedMemory(name=mem_Bx["address"])
        shm_By = shared_memory.SharedMemory(name=mem_By["address"])
        shm_Bz = shared_memory.SharedMemory(name=mem_Bz["address"])
        shm_vx = shared_memory.SharedMemory(name=mem_vx["address"])
        shm_vy = shared_memory.SharedMemory(name=mem_vy["address"])
        shm_vz = shared_memory.SharedMemory(name=mem_vz["address"])
        shm_Jz = shared_memory.SharedMemory(name=mem_Jz["address"])
        shm_T = shared_memory.SharedMemory(name=mem_T["address"])

        rho = np.ndarray(mem_rho["shape"], dtype = mem_rho["dtype"], buffer = shm_rho.buf) * m_p
        bx = np.ndarray(mem_Bx["shape"], dtype = mem_Bx["dtype"], buffer = shm_Bx.buf) / np.sqrt(mu_0 * rho)
        by = np.ndarray(mem_By["shape"], dtype = mem_By["dtype"], buffer = shm_By.buf) / np.sqrt(mu_0 * rho)
        bz = np.ndarray(mem_Bz["shape"], dtype = mem_Bz["dtype"], buffer = shm_Bz.buf) / np.sqrt(mu_0 * rho)
        vx = np.ndarray(mem_vx["shape"], dtype = mem_vx["dtype"], buffer = shm_vx.buf)
        vy = np.ndarray(mem_vy["shape"], dtype = mem_vy["dtype"], buffer = shm_vy.buf)
        vz = np.ndarray(mem_vz["shape"], dtype = mem_vz["dtype"], buffer = shm_vz.buf)
        Jz = np.ndarray(mem_Jz["shape"], dtype = mem_Jz["dtype"], buffer = shm_Jz.buf)
        T = np.ndarray(mem_T["shape"], dtype = mem_T["dtype"], buffer = shm_T.buf)

        b_perp = np.sqrt(bx**2 + by**2)
        b_parr_rms = np.sqrt(np.mean((bz**2), axis=1) - np.mean((bz), axis=1)**2)
        b_perp_rms = np.sqrt(np.mean((b_perp**2), axis=1) - np.mean((b_perp), axis=1)**2)
        del b_perp

        v_perp = np.sqrt(vx**2 + vy**2)
        v_parr_rms = np.sqrt(np.mean((vz**2), axis=1) - np.mean((vz), axis=1)**2)
        v_perp_rms = np.sqrt(np.mean((v_perp**2), axis=1) - np.mean((v_perp), axis=1)**2)
        del v_perp

        Jz_rms = np.sqrt(np.mean((Jz**2), axis=1) - np.mean((Jz), axis=1)**2)
        
        T = np.swapaxes(T,1,2)
        T_perp = np.mean(1/2 * (T[:,0,:] + T[:,1,:]) / T_0,axis=1) 
        T_parr = np.mean(T[:,2,:] / T_0,axis=1) 
        A = np.mean(1/2 * (T[:,0,:] + T[:,1,:])/T[:,2,:],axis=1)

        sigma_c = np.mean((2 * (vx*bx + vy*by + vz*bz) / (vx**2+vy**2+vz**2 + bx**2+by**2+bz**2)), axis = 1)
        sigma_r = np.mean(((vx**2+vy**2+vz**2 - bx**2-by**2-bz**2) / (vx**2+vy**2+vz**2 + bx**2+by**2+bz**2)), axis = 1)

        fig, axes = plt.subplots(2,2, figsize = (10,10))
        axes = axes.flatten()

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
        
        axes[0].plot(time, Jz_rms, label = r'{}'.format(J_z_rms_label))
        
        axes[1].plot(time, b_perp_rms, label = r'{}'.format(b_perp_label))
        axes[1].plot(time, b_parr_rms, label = r'{}'.format(b_parr_label))
        axes[1].plot(time, v_perp_rms, label = r'{}'.format(v_perp_label))
        axes[1].plot(time, v_parr_rms, label = r'{}'.format(v_parr_label))
        
        axes[2].plot(time, T_perp, label = r'{}'.format(T_perp_label))
        axes[2].plot(time, T_parr, label = r'{}'.format(T_parr_label))
        axes[2].plot(time, A, label = r'{}'.format(A_label))
        
        axes[3].plot(time, sigma_c, label = r'{}'.format(label_c))
        axes[3].plot(time, sigma_r, label = r'{}'.format(label_r))

        axes[0].legend()
        axes[1].legend()
        axes[2].legend()
        axes[3].legend()

        object.name = object.name[0:-3] + "jpg"

        fig.savefig(object.name)

        plt.close()