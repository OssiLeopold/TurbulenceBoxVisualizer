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

def animation_streamplot_unit(self):
    self.fig, self.ax = plt.subplots()
    shm_background = shared_memory.SharedMemory(name=self.object.memory_space["background"])
    shm_Bx = shared_memory.SharedMemory(name=self.object.memory_space["vg_b_volx"])
    shm_By = shared_memory.SharedMemory(name=self.object.memory_space["vg_b_voly"])
    
    background = np.ndarray(self.object.shape["background"], dtype=self.object.dtype, buffer=shm_background.buf)
    B_x = np.ndarray(self.object.shape["vg_b_volx"], dtype=self.object.dtype, buffer=shm_Bx.buf)
    B_y = np.ndarray(self.object.shape["vg_b_voly"], dtype=self.object.dtype, buffer=shm_By.buf)

    self.background_mesh = background.reshape((self.frames, self.x_length, self.x_length))
    self.B_x_mesh = B_x.reshape((self.frames, self.x_length, self.x_length))
    self.B_y_mesh = B_y.reshape((self.frames, self.x_length, self.x_length))

    if self.object.variable in ["J_vs_A", "E_vs_A"]:
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

    else:
        self.Min = round(min(background.flatten()), 15)
        self.Max = round(max(background.flatten()), 15)

        if abs(self.Min) > abs(self.Max):
            self.Max = -self.Min
        else:
            self.Min = -self.Max
        
        nx_s, ny_s = 5, 5
        sx = np.linspace(self.x_mesh[0][1], self.x_mesh.max(), num=nx_s)
        sy = np.linspace(self.x_mesh[0][1], self.x_mesh.max(), num=ny_s)
        SX, SY = np.meshgrid(sx, sy, indexing="xy")
        self.seeds = np.column_stack([SX.ravel(), SY.ravel()])

        self.first = True
        anim = animation.FuncAnimation(self.fig, self.streamplot_update_unit, frames = self.frames, interval = 20)
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

def streamplot_update_unit(self, frame):
    self.ax.clear()
    pcm = self.ax.pcolormesh(self.x_mesh, self.y_mesh, self.background_mesh[frame], vmin = self.Min, vmax = self.Max,cmap = "bwr")
    if self.first == True:
        self.fig.colorbar(pcm, ax=self.ax)
    self.ax.streamplot(
        self.x_mesh, self.y_mesh, self.B_x_mesh[frame], self.B_y_mesh[frame], start_points = self.seeds, broken_streamlines = False, linewidth = 0.6, arrowstyle = "-")
    self.first = False

def animation_streamplot_unit(self):
    self.fig, self.ax = plt.subplots()
    shm_background = shared_memory.SharedMemory(name=self.object.memory_space["background"])
    shm_Bx = shared_memory.SharedMemory(name=self.object.memory_space["vg_b_volx"])
    shm_By = shared_memory.SharedMemory(name=self.object.memory_space["vg_b_voly"])
    
    background = np.ndarray(self.object.shape["background"], dtype=self.object.dtype, buffer=shm_background.buf)
    B_x = np.ndarray(self.object.shape["vg_b_volx"], dtype=self.object.dtype, buffer=shm_Bx.buf)
    B_y = np.ndarray(self.object.shape["vg_b_voly"], dtype=self.object.dtype, buffer=shm_By.buf)

    self.background_mesh = background.reshape((self.frames, self.x_length, self.x_length))
    self.B_x_mesh = B_x.reshape((self.frames, self.x_length, self.x_length))
    self.B_y_mesh = B_y.reshape((self.frames, self.x_length, self.x_length))

    if self.object.variable in ["J_vs_A", "E_vs_A"]:
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

    else:
        self.Min = round(min(background.flatten()), 15)
        self.Max = round(max(background.flatten()), 15)

        if abs(self.Min) > abs(self.Max):
            self.Max = -self.Min
        else:
            self.Min = -self.Max
        
        nx_s, ny_s = 5, 5
        sx = np.linspace(self.x_mesh[0][1], self.x_mesh.max(), num=nx_s)
        sy = np.linspace(self.x_mesh[0][1], self.x_mesh.max(), num=ny_s)
        SX, SY = np.meshgrid(sx, sy, indexing="xy")
        self.seeds = np.column_stack([SX.ravel(), SY.ravel()])

        self.first = True
        anim = animation.FuncAnimation(self.fig, self.streamplot_update_unit, frames = self.frames, interval = 20)
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

def streamplot_update_unit(self, frame):
    self.ax.clear()
    pcm = self.ax.pcolormesh(self.x_mesh, self.y_mesh, self.background_mesh[frame], vmin = self.Min, vmax = self.Max,cmap = "bwr")
    if self.first == True:
        self.fig.colorbar(pcm, ax=self.ax)
    self.ax.streamplot(
        self.x_mesh, self.y_mesh, self.B_x_mesh[frame], self.B_y_mesh[frame], start_points = self.seeds, broken_streamlines = False, linewidth = 0.6, arrowstyle = "-")
    self.first = False