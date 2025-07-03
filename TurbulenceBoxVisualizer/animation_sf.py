import analysator as pt
import matplotlib.pyplot as plt
import os
from matplotlib import animation
from matplotlib.animation import FFMpegWriter
import numpy as np
from multiprocessing import shared_memory
import seaborn as sns
import numexpr as ne

plt.rcParams['animation.ffmpeg_path'] = "/home/rxelmer/Documents/turso/appl_local/ffmpeg/bin/ffmpeg"

os.environ['PATH']='/home/rxelmer/Documents/turso/appl_local/tex-basic/texlive/2023/bin/x86_64-linux:'+ os.environ['PATH'] 
os.environ['PTNOLATEX']='1'

class AnimationSF():
    def __init__(self, object):
        self.object = object
        shm = shared_memory.SharedMemory(name=object.memory_space)
        self.data = np.ndarray(object.shape, dtype=object.dtype, buffer=shm.buf)

        shm_time = shared_memory.SharedMemory(name=object.time)
        self.time = np.ndarray(object.time_shape, dtype=object.time_dtype, buffer=shm_time.buf)

        self.vlsvobj = pt.vlsvfile.VlsvReader(object.bulkpath + "bulk.0000000.vlsv")
        self.x_length = int(self.vlsvobj.read_parameter("xcells_ini"))
        self.frames = len(self.data)

        self.data_mesh_x = np.empty((self.frames, self.x_length, self.x_length))
        for i in range(self.frames):
                self.data_mesh_x[i] = self.data[i].reshape(-1, self.x_length)
        
        self.data_mesh_y = np.empty((self.frames, self.x_length, self.x_length))
        for i in range(self.frames):
                self.data_mesh_y[i] = self.data[i].reshape(-1, self.x_length).T

        self.titles = []
        for dl in self.object.delta_ls:
            title = f"$\\Delta l={{{dl}}}\\ cells$"
            self.titles.append(r"{}".format(title))

        if len(self.object.delta_ls) <= 3:
            fig, self.axes = plt.subplots(1,len(self.object.delta_ls), figsize=(12,4))
        elif len(self.object.delta_ls) == 4:
            fig, self.axes = plt.subplots(2,2, figsize=(12,8))
        elif len(self.object.delta_ls) <= 6:
            fig, self.axes = plt.subplots(2,3, figsize=(14,8))
        elif len(self.object.delta_ls) <= 9:
            fig, self.axes = plt.subplots(3,3, figsize=(16,12))

        fig.tight_layout(pad=4.0)
        suptitle = f"${{{object.variable_name}}}_{{{object.component}}}$ structure function"
        fig.suptitle(r'{}'.format(suptitle))
        self.timelabel = fig.text(0.6, 0.98, 'Centered at top of figure', ha='center', va='top', fontsize = 16)
        
        self.axes = self.axes.flatten()

        for i in range(len(self.axes) - len(self.object.delta_ls)):
            fig.delaxes(self.axes[-i])
            self.axes = np.delete(self.axes,-i)

        anim = animation.FuncAnimation(fig, self.update, frames = self.frames, interval = 20)
        
        writer = FFMpegWriter(fps=5)
        anim.save(object.name, writer=writer)
        plt.close()

    def update(self,frame):
        for ax in self.axes:
            ax.clear()

        delta_array_container = np.empty((len(self.object.delta_ls), 18 * self.x_length))
        
        for i, dl in enumerate(self.object.delta_ls):
            index = 0
            for j in range(int(self.x_length/10), int(self.x_length/10*9)+1, int(self.x_length/10)):
                value_slice = self.data_mesh_x[frame][j]
                beginning = self.data_mesh_x[frame][j][:-dl]
                end = self.data_mesh_x[frame][j][-dl:]
                
                value_slice_shifted = np.empty(self.x_length)
                value_slice_shifted[dl:] = beginning
                value_slice_shifted[:dl] = end

                delta_array_container[i][index * self.x_length:((index+1)*self.x_length)] = ne.evaluate('value_slice - value_slice_shifted')
                
                value_slice = self.data_mesh_y[frame][j]
                beginning = self.data_mesh_y[frame][j][:-dl]
                end = self.data_mesh_y[frame][j][-dl:]
                
                value_slice_shifted = np.empty(self.x_length)
                value_slice_shifted[dl:] = beginning
                value_slice_shifted[:dl] = end

                delta_array_container[i][((index+1)*self.x_length):((index+2)*self.x_length)] = ne.evaluate('value_slice - value_slice_shifted')

                index += 2


        """ for i in range(4):    
            print(i,delta_array_container[i][-10:-1]) """

        for i, ax in enumerate(self.axes):
            mean = np.mean(delta_array_container[i])
            SD = np.std(delta_array_container[i])
            delta_array_container[i] = (delta_array_container[i] - mean) / SD

            ax.hist(delta_array_container[i], bins=50, density=True)
            #sns.kdeplot(delta_array_container[i], fill=True, ax=ax)

            mean = np.mean(delta_array_container[i])
            SD = np.std(delta_array_container[i])
            x = np.linspace(mean - 4*SD, mean + 4*SD, 1000)
            gaussian = (1 / (SD * np.sqrt(2 * np.pi))) * np.exp(- (x - mean)**2 / (2 * SD**2))

            ax.plot(x,gaussian)

        for i, ax in enumerate(self.axes):
            ax.set_title(self.titles[i])

            x_label = f"$(\\delta {{{self.object.variable_name}}}_{{{self.object.component}}}-\\mu)/\\sigma$"

            if len(self.axes) < 4:
                ax.set_xlabel(x_label)
            elif len(self.axes) == 4 and i > len(self.axes) - 3:
                ax.set_xlabel(x_label)
            elif len(self.axes) > 4 and i > len(self.axes) - 4:
                ax.set_xlabel(x_label)

            ax.set_xlim(-6,6)
            ax.set_yscale("log")
            ax.set_ylim(1e-3,1)

        self.timelabel.set_text(f"{self.time[frame]:.1f}s")

            