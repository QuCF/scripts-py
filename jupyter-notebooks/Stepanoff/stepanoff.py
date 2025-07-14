import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import h5py

from scipy.integrate import RK45
from scipy.integrate import solve_ivp
import random

import os

import pylib.mix as mix

from matplotlib import colors
colors_ = ["blue", "red", "green", "gray", "black"]


def reload():
    mix.reload_module(mix)
    return




# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
def f_init_1d(kappa, x):
    f_loc = lambda x1: np.exp(kappa * np.sin(x1/2.)**2) - 1
    y = f_loc(x) / f_loc(np.pi)
    return y


# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
def f_init_2d(kappa, x, Nx):
    g_cl = f_init_1d(kappa, x)

    y_2D = np.zeros((Nx, Nx))
    f_loc = lambda x1, x2: np.exp(kappa * np.sin((x1 + x2)/2.)**2) - 1
    for i_phi in range(Nx):
        for i_theta in range(Nx):
            y_2D[i_theta, i_phi] = f_loc(x[i_theta], x[i_phi]) * g_cl[i_phi]
    y_2D /= f_loc(np.pi, 0.)
    return y_2D


# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
def create_diag_matrix(t, N_half, alpha, name_matrix, flag_cut):
    if not flag_cut:
        N = 2 * N_half + 1
        c_arr = np.arange(-N_half, N_half+1)
    else:
        N = 2 * N_half
        c_arr = np.arange(-N_half, N_half)

    aa = np.linspace(0, 2.*np.pi, N)
    diag = np.zeros(N**2)
    count = -1

    if name_matrix == "M1":
        for i2 in range(N):
            for i1 in range(N):
                count += 1
                diag[count] = - (1. - alpha) * c_arr[i1] * (1. - np.cos(aa[i2]))

    if name_matrix == "M2":
        for i2 in range(N):
            for i1 in range(N):
                count += 1
                diag[count] = - alpha * c_arr[i2] * (1. - np.cos(aa[i1]))

    # --- reference unitary matrix ---
    Nsq = N**2
    diag_U_ref = np.zeros(Nsq, dtype = complex)
    for ii in range(Nsq):
        diag_U_ref[ii] = np.exp(-1j * t * diag[ii])
    return diag_U_ref


# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
def rearrange_into_2d(y, N):
    y_new = np.zeros((N, N), dtype = y.dtype)
    for ic in range(N):
        for ir in range(N):
            y_new[ir, ic] = y[ir + ic * N]
    return y_new


# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
def change_coord_to_orig(psi, Nx):
    y_orig = np.array(psi.real.transpose())
    y_new  = np.zeros((Nx, Nx))
    for ir in range(Nx):
        for ic in range(Nx):
            ic_new = ic + ir
            if ic_new >= Nx:
                ic_new = ic_new - Nx
            y_new[ir, ic_new] = y_orig[ir, ic]
    return y_new


# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
def computation_in_real_and_fourier(
        dt, Nt, x,
        diag_U_M1, diag_U_M2, init_2d,
        flag_plot = False
    ):
    # ----------------------------------------------------------
    def evolution_one_matrix(psi_init_x, id_axis):
        init_fft_2d = np.fft.fft(psi_init_x, axis = id_axis)
        init_fft_2d = np.fft.fftshift(init_fft_2d, axes = id_axis)
        init_fft_1d = init_fft_2d.flatten(order = "F")  # CHECK; 

        if id_axis == 0:
            psi_t_fourier_1D = diag_U_M1 * init_fft_1d
        else:
            psi_t_fourier_1D = diag_U_M2 * init_fft_1d

        psi_t_fourier_2D = rearrange_into_2d(psi_t_fourier_1D, Nx)
        psi_t_fourier_2D = np.fft.ifftshift(psi_t_fourier_2D, axes = id_axis)
        psi_t_x = np.fft.ifft(psi_t_fourier_2D, axis = id_axis)
        return psi_t_x
    # ----------------------------------------------------------
    Nx = len(x)

    # --- Computation ---
    psi_t_x = np.array(init_2d)
    for ii_t in range(Nt):
        # --- evolution in Fourier over theta ---
        psi_t_x = evolution_one_matrix(psi_t_x, 0)

        # --- evolution in Fourier over phi ---
        psi_t_x = evolution_one_matrix(psi_t_x, 1)

    # --- Return to the original coordinates ---
    y_mod = np.array(psi_t_x.real.transpose()) # the result in modified coordinates
    y_orig = change_coord_to_orig(psi_t_x, Nx)
    y_orig /= np.max(np.abs(y_orig)) # the normalized result in the original coordinates

    # --- Plotting ---
    if flag_plot:
        X, Y = np.meshgrid(x/np.pi, x/np.pi)
        fig1, axs = plt.subplots(1, 1, figsize=(10,8))
        plot_2D_subplot(
            axs, fig1, X, Y, y_orig,  
            str_title="$" + "t={:0.2f}".format(Nt*dt) + "$", 
            label_x = '$\\theta/\pi$', label_y = "$\phi/\pi$"
        )
        plt.tight_layout()
    return y_mod, y_orig


# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
def form_file_name(alpha, kappa, nx, prefix = ""):
    file_name = "{:s}Stepanoff_data_k{:0.1f}_a{:0.3f}_n{:d}.hdf5".format(prefix, kappa, alpha, nx)
    return file_name


# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
def load_data(full_name):
    print("Reading the coefficients from:\n " + full_name)
    with h5py.File(full_name, "r") as f:
        grp = f["basic"]

        nx = int(grp["nx"][()])
        N_data = int(grp["N_data"][()])
        alpha = grp["alpha"][()]
        kappa = grp["kappa"][()]
        theta = np.array(grp["theta"])

        fx_arr = []
        t_arr  = []
        for id_data in range(N_data):
            grp = f["data_{:d}".format(id_data)]
            fx = np.array(grp["fx"])
            t  = float(grp["t"][()])
            fx_arr.append(fx)
            t_arr.append(t)

    # --- Form grids ---
    Nx = 1 << nx
    x1, x2 = np.meshgrid(theta, theta)
    x = np.concatenate((x1[:, :, np.newaxis], x2[:, :, np.newaxis]), axis=2)
    x1 = x1 / np.pi
    x2 = x2 / np.pi
    x = np.reshape(x, (Nx ** 2, 2))
    return fx_arr, t_arr, Nx, nx, alpha, kappa, x1, x2, x


# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
def get_Dopri_classical_simulations(
        dt, Nt, x,
        alpha, kappa,
        path_save,
        prefix = "", flag_plot = True
    ):
    Nx = len(x)
    nx = int(np.log2(Nx))

    full_name = form_file_name(alpha, kappa, nx, prefix)

    print("--- Looking for the file: {:s} in the path {:s}".format(full_name, path_save))
    full_name = os.path.join(path_save, full_name)
    if os.path.isfile(full_name):
        print("--- Loading data ---")
        y_cl_dopri_arr, t_arr, _, _, _, _, _, _, _, = load_data(full_name)
    else:
        print("file is not found")

    # --- Find a necessary simulation ---
    t_ref = int(Nt * dt)
    t_arr_ = np.array(t_arr, dtype = int)
    pos_arr = np.where(t_arr_ == t_ref)[0][0]
    print()
    print("Take the Dopri simulation at t = {:d}".format(t_arr_[pos_arr]))
    y_cl_dopri_1d = y_cl_dopri_arr[pos_arr]

    # --- Normalize to 1 ---
    y_cl_dopri_1d /= np.max(np.abs(y_cl_dopri_1d))

    # --- from 1D to 2D ---
    y_cl_dopri_2d = np.reshape(y_cl_dopri_1d, (Nx, Nx)) 

    # --- Plotting ---
    if flag_plot:
        X, Y = np.meshgrid(x/np.pi, x/np.pi)
        fig1, axs = plt.subplots(1, 1, figsize=(10,8))
        plot_2D_subplot(
            axs, fig1, X, Y, y_cl_dopri_2d,  
            str_title="CL-real: $" + "t={:0.2f}".format(Nt*dt) + "$", 
            label_x = '$\\theta/\pi$', label_y = "$\phi/\pi$"
        )
        plt.tight_layout()

    return y_cl_dopri_2d



# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
def plot_2D(X, Y, f, str_title="", fontsize = 30,):
    fig = plt.figure(figsize=(11,10))
    ax = fig.add_subplot(111)

    f_max_loc = np.max(np.abs(f))

    cmap = plt.get_cmap("seismic")
    levels = np.linspace(-f_max_loc, f_max_loc, 101) 
    divnorm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    cs = ax.pcolormesh(X, Y, f, cmap=cmap, norm=divnorm, shading='gouraud') 
    cb = fig.colorbar(cs, ax = ax)
    cb.ax.ticklabel_format(style="scientific")
    cb.ax.tick_params(labelsize=fontsize)

    offset_text = cb.ax.yaxis.get_offset_text()
    offset_text.set_fontsize(fontsize) 
    offset_text.set_x(4) 

    ax.set_xlabel('$\\theta/\pi$', fontsize = fontsize)
    ax.set_ylabel("$\phi/\pi$", fontsize = fontsize)
    ax.set_title(str_title, fontsize = fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    # ax.grid()
    return


# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
def plot_2D_subplot(
        ax, fig, X, Y, f, 
        str_title="", 
        fontsize = 30, 
        label_x = '$\\theta\'/\pi$', label_y = "$\phi\'/\pi$",
        flag_x = True, flag_y = True
):
    f_max_loc = np.max(np.abs(f))
    f_min_loc = np.min(f)

    cmap = plt.get_cmap("seismic")
    levels = np.linspace(-f_max_loc, f_max_loc, 101) 
    # levels = np.linspace(f_min_loc, f_max_loc, 101) 
    divnorm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    cs = ax.pcolormesh(X, Y, f, cmap=cmap, norm=divnorm, shading='gouraud') 
    cb = fig.colorbar(cs, ax = ax)
    cb.ax.ticklabel_format(style="scientific")
    cb.ax.tick_params(labelsize=fontsize)

    offset_text = cb.ax.yaxis.get_offset_text()
    offset_text.set_fontsize(fontsize) 
    offset_text.set_x(4) 

    ax.set_xlabel(label_x, fontsize = fontsize)
    ax.set_ylabel(label_y, fontsize = fontsize)
    ax.set_title(str_title, fontsize = fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    # ax.grid()

    if not flag_x:
        ax.set_xticklabels([])
    if not flag_y:
        ax.set_yticklabels([])
    return


# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
def plot_init(kappa, x):
    Nx = len(x)
    y_2D = f_init_2d(kappa, x, Nx)

    # --- Plotting 2D ---
    X, Y = np.meshgrid(x/np.pi, x/np.pi)
    plot_2D(X, Y, y_2D, "initial conditions in modified coordinates")
    return