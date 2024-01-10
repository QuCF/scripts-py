import numpy as np
import importlib as imp
import h5py
# import pylib.Global_variables as GLO
import matplotlib.pyplot as plt
import sys
from numba import jit

import pylib.mix as mix
import pylib.qucf_oracle as qucf_o
import pylib.qucf_read as qucf_r
import pylib.qucf_matrix_D as qucf_m


def reload():
    mix.reload_module(mix)
    mix.reload_module(qucf_o)
    mix.reload_module(qucf_r)
    mix.reload_module(qucf_m)
    return


def plot_A_structure(
        A, label_A, file_save,
        flag_save = False, path_save = None, 
        fontsize = 24, cmap='bwr', marker_size = 4,
        text_coord_xy = [4, 4]
):
    def find_rows_columns(A):
        N = A.shape[0]
        rows_plot_1 = np.zeros(N*N)
        cols_plot_1 = np.zeros(N*N)
        N_nz_1 = 0
        for ir in range(N):
            for ic in range(N):
                if np.abs(A[ir, ic]) > 0:
                    N_nz_1 += 1
                    rows_plot_1[N_nz_1-1] = ir
                    cols_plot_1[N_nz_1-1] = ic
        rows_plot_1 = rows_plot_1[0:N_nz_1]
        cols_plot_1 = cols_plot_1[0:N_nz_1]
        return rows_plot_1, cols_plot_1

    N = A.shape[0]
    rows_A, cols_A = find_rows_columns(A)

    fig1 = plt.figure(figsize=(10,10))
    ax = fig1.add_subplot(111)
    ax.scatter(rows_A, cols_A, color="red", s = marker_size) 
    plt.xlim(0, N)
    plt.ylim(0, N)
    plt.gca().invert_yaxis()
    plt.xlabel(r'$\textbf{columns}$', fontsize = fontsize)
    plt.ylabel(r"$\textbf{rows}$", fontsize = fontsize)
    plt.grid()
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.text(
        int(text_coord_xy[0]), int(text_coord_xy[1]), 
        r'$\textbf{' + label_A + '}$', fontsize=fontsize
    )
    plt.show()
    if flag_save:
        if path_save is None:
            print("Error: a path for saving a figure is not given.")
            return
        plt.savefig(path_save + "/" + "{:s}.png".format(file_save))
    return