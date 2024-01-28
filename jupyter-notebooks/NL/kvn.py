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





# class KvN_:
#     Nx = None
#     Nt = None
#     x = None
#     t = None
#     dx = None
#     dt = None
#     F_ = None


#     def __init__(self, nx, nt, x_max, t_max, F_):
#         self.F_ = F_
#         self.Nx = 1<<nx
#         self.Nt = 1<<nt
#         self.x = np.linspace(-x_max, x_max, self.Nx)
#         self.t = np.linspace(0, t_max, self.Nt)
#         self.dx = np.diff(self.x)[0]
#         self.dt = np.diff(self.t)[0]
#         return


#     def construct_CD_matrix(self):
#         H_CD = np.zeros((self.Nx, self.Nx), dtype=complex)
#         for ii in range(1,self.Nx-1):
#             Fm = self.F_(self.x[ii-1]) 
#             Fc = self.F_(self.x[ii]) 
#             Fp = self.F_(self.x[ii+1]) 
#             H_CD[ii,ii-1] = - (Fm + Fc)
#             H_CD[ii,ii+1] =   (Fp + Fc)
#         H_CD = -1j/(4.*self.dx) * H_CD
#         H_CD[1,0] = 0.0
#         H_CD[self.Nx-2, self.Nx-1] = 0.0
#         return H_CD


#     def construct_UW_matrix(self):
#         H_UW = np.zeros((self.Nx, self.Nx), dtype=complex)
#         for ii in range(1,self.Nx-1):
#             Fc = self.F_(self.x[ii])

#             if Fc <= 0:
#                 Fp = self.F_(self.x[ii+1])
#                 H_UW[ii,ii]   = - 2 * Fc
#                 H_UW[ii,ii+1] = Fp + Fc
#             else:
#                 Fm = self.F_(self.x[ii-1])
#                 H_UW[ii,ii]   = 2 * Fc
#                 H_UW[ii,ii-1] = - (Fm + Fc)

#         H_UW = -1j/(2.*self.dx) * H_UW
#         # H_UW[self.Nx-2, self.Nx-1] = 0.0
#         return H_UW


#     def solve_using_matrix(self, A, psi_init):
#         psi_tx = np.zeros((self.Nt,self.Nx), dtype = complex)
#         psi_tx[0,:] = np.array(psi_init)

#         coef_dt = self.dt * (-1j)
#         for it in range(self.Nt-1):
#             Hpsi = A.dot(psi_tx[it])

#             for ix in range(1, self.Nx-1):
#                 psi_tx[it+1, ix] = psi_tx[it, ix] + coef_dt * Hpsi[ix]
#         return psi_tx


#     def compute_mean(self, psi_tx_matrix):
#         mean_t = np.zeros(self.Nt)
#         for it in range(self.Nt):
#             psi_t1   = psi_tx_matrix[it,:]
#             psi_t1_c = np.conjugate(psi_t1)
#             norm = psi_t1_c.dot(psi_t1) * self.dx
#             mean_t[it] = np.real(np.trapz(self.x*psi_t1_c*psi_t1, dx=self.dx) / norm)
#         return mean_t   


# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

def is_Hermitian(A, name):
    inds_nonzero = np.nonzero(np.transpose(np.conjugate(A)) - A)
    if np.size(inds_nonzero) == 0:
        print("the matrix {:s} is Hermitian".format(name))
    else:
        print("the matrix {:s} is non-Hermitian".format(name))
    return


def h_adj(AA):
    return np.transpose(np.conjugate(AA))
    

def get_herm_aherm_parts(B):
    Bh = (B + h_adj(B)) / 2.
    Ba = (B - h_adj(B)) / (2.j)
    # B_ch = Bh + 1j * Ba
    return Bh, Ba 


def plot_A_structure(
        A, name_A, label_A, 
        file_save, flag_save = False, path_save = None, 
        fontsize = 24, cmap='bwr', marker_size = 160,
        text_coord_label = [1, 14],
        text_coord_name_A = [12, 1],
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
    plt.xlim(0, N - 1)
    plt.ylim(0, N - 1)
    plt.gca().invert_yaxis()
    plt.xlabel(r'$\textbf{columns}$', fontsize = fontsize)
    plt.ylabel(r"$\textbf{rows}$", fontsize = fontsize)
    plt.grid()

    plt.yticks(np.arange(0, N, 3))
    plt.xticks(np.arange(0, N, 3))

    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.text(
        int(text_coord_label[0]), int(text_coord_label[1]), 
        r'$\textbf{' + label_A + '}$', fontsize=fontsize
    )
    ax.text(
        int(text_coord_name_A[0]), int(text_coord_name_A[1]), 
        r'$' + name_A + '$', fontsize=fontsize
    )
    plt.show()
    if flag_save:
        if path_save is None:
            print("Error: a path for saving a figure is not given.")
            return
        plt.savefig(path_save + "/" + "{:s}.png".format(file_save))
    return