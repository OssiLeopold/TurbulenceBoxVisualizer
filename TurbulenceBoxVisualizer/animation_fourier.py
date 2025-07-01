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

class AnimationFourier():
    def __init__(self, object):
        self.object = object
        shm = shared_memory.SharedMemory(name=object.memory_space)
        self.data = np.ndarray(object.shape, dtype=object.dtype, buffer=shm.buf)