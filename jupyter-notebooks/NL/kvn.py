import numpy as np
import importlib as imp
import h5py
# import pylib.Global_variables as GLO
import matplotlib.pyplot as plt
import sys
from numba import jit

from scipy.linalg import expm
from scipy.integrate import RK45

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


def is_Hermitian(A, name):
    return mix.is_Hermitian(A, name)


def h_adj(AA):
    return np.transpose(np.conjugate(AA))
    

def get_herm_aherm_parts(B):
    return mix.get_herm_aherm_parts(B)


def plot_A_structure(
        A, 
        name_A = "", label_A = "", 
        file_save = "", flag_save = False, path_save = None, 
        fontsize = 24, marker_size = 160,
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


# ---------------------------------------------------------------------------
def is_Hermitian(A, name):
    return mix.is_Hermitian(A, name)


# ---------------------------------------------------------------------------
def construct_CD_matrix_1D(x, F):
    Nx = len(x)
    dx = np.diff(x)[0]
    H_CD = np.zeros((Nx, Nx), dtype=complex)
    for ii in range(1,Nx-1):
        Fm = F(x[ii-1]) 
        Fc = F(x[ii]) 
        Fp = F(x[ii+1]) 
        H_CD[ii,ii-1] = - (Fm + Fc)
        H_CD[ii,ii+1] =   (Fp + Fc)
    H_CD = -1j/(4.*dx) * H_CD
    H_CD[1,0] = 0.0
    H_CD[Nx-2, Nx-1] = 0.0
    return H_CD


# ---------------------------------------------------------------------------
def construct_UW_matrix_1D(x, F):
    Nx = len(x)
    dx = np.diff(x)[0]
    H_UW = np.zeros((Nx, Nx), dtype=complex)

    # --- left ---
    Fc = F(x[0])
    if Fc <= 0:
        Fp = F(x[ii+1])
        H_UW[0,0] = - 2 * Fc
        H_UW[0,1] = Fp + Fc
    else:
        H_UW[0,0]   = 2 * Fc
    # --- bulk ---    
    for ii in range(1,Nx-1):
        Fc = F(x[ii])

        if Fc <= 0:
            Fp = F(x[ii+1])
            H_UW[ii,ii]   = - 2 * Fc
            H_UW[ii,ii+1] = Fp + Fc
        else:
            Fm = F(x[ii-1])
            H_UW[ii,ii]   = 2 * Fc
            H_UW[ii,ii-1] = - (Fm + Fc)
    # --- right ---
    Fc = F(x[Nx-1])
    if Fc <= 0:
        H_UW[Nx-1,Nx-1]   = - 2 * Fc
    else:
        Fm = F(x[Nx-2])
        H_UW[Nx-1,Nx-1]   = 2 * Fc
        H_UW[Nx-1,Nx-2] = - (Fm + Fc)
    # ---
    H_UW = -1j/(2.*dx) * H_UW
    # H_UW[Nx-2, Nx-1] = 0.0
    return H_UW


# ---------------------------------------------------------------------------
def solve_KvN_1D_using_Hamiltonian(t, Nx, psi_init, A):
    Nt = len(t)
    dt = np.diff(t)[0]

    psi_tx      = np.zeros((Nt,Nx), dtype = complex)
    psi_tx[0,:] = np.array(psi_init)

    coef_dt = dt * (-1j)
    for it in range(Nt-1):
        Hpsi = A.dot(psi_tx[it])

        for ix in range(1, Nx-1):
            psi_tx[it+1, ix] = psi_tx[it, ix] + coef_dt * Hpsi[ix]
    return psi_tx


# ---------------------------------------------------------------------------
def compute_mean_1D(x, Nt, psi_tx_matrix):
    # x_operator = np.diag(x)
    dx = np.diff(x)[0]
    mean_t = np.zeros(Nt)
    for it in range(Nt):
        psi_t1   = psi_tx_matrix[it,:]
        psi_t1_c = np.conjugate(psi_t1)
        norm = psi_t1_c.dot(psi_t1) * dx
        mean_t[it] = np.real(np.trapz(x*psi_t1_c*psi_t1, dx=dx) / norm)
    return mean_t


# ------------------------------------------------------------------------------------------------
def comp_LCHS_weights(k):
    dk = np.diff(k)[0]
    Nk = len(k)

    wk = np.zeros(Nk)
    for ik in range(Nk):
        wk[ik] = 1. / (1 + k[ik]*k[ik])
    wk = wk * dk/np.pi
    wk[0]  = 0.5 * wk[0]
    wk[-1] = 0.5 * wk[-1]  
    return wk


# ------------------------------------------------------------------------------------------------
def LCHS_computation(k, dt, Hi, psi_init, Nt_loc, flag_trotterization, flag_print = False):
    # if flag_direct = False, use 2nd order Trotterization.

    # k-grid:
    dk = np.diff(k)[0]
    k_max = k[-1]
    Nk = len(k)

    # matrices:
    Bh, Ba = get_herm_aherm_parts(Hi)
    wk = comp_LCHS_weights(k)
    N = Hi.shape[0]
    
    exp_max = None
    exp_Ba = None
    if flag_trotterization:
        Prop_Ba = -1.j * dt/2. * Ba
        exp_Ba = expm(Prop_Ba)
        Prop_kmax = 1.j * dt * k_max* Bh
        exp_max = expm(Prop_kmax)
        del Prop_Ba, Prop_kmax
    
    exp_LCHS = np.zeros((N,N), dtype=complex)
    for ik in range(Nk):
        temp = np.identity(N, dtype=complex)
        
        exp_dt = None
        if not flag_trotterization:
            Prop_k = -1.j * dt * (Ba + k[ik]*Bh) # here, use Trotterization
            exp_dt = expm(Prop_k)
        else:
            Prop_k = -1.j * dt * (ik * dk) * Bh
            exp_dt = exp_max.dot(expm(Prop_k))
            exp_dt = exp_dt.dot(exp_Ba)
            exp_dt = exp_Ba.dot(exp_dt)
            
        for it in range(Nt_loc):
            temp = exp_dt.dot(temp)
        exp_LCHS += wk[ik] * temp
    del temp, Prop_k, exp_max, exp_Ba, exp_dt, ik
         
    # compare the exponentiating matrices:
    if flag_print:
        exp_ref = np.identity(N, dtype=complex)
        exp_dt  = expm(-dt*Hi)
        for it in range(Nt_loc):
            exp_ref = exp_dt.dot(exp_ref)
        del exp_dt
    
        analyse_exp_matrices(exp_ref, exp_LCHS)
        del exp_ref
        
    # compute the output quantum state:
    psi_t = exp_LCHS.dot(psi_init)
    
    if flag_print:
        print()
        print("sum psi_t_max[max-time]**2: {:0.3e}".format(np.sum(np.abs(psi_t)**2)))
    return psi_t


# ------------------------------------------------------------------------------------------------
def analyse_exp_matrices(exp_1, exp_2):
    print("\n--- Exponentiation matrices ---")
    print(exp_1)
    print()
    print(exp_2)

    print("\n --- Difference between the matrix elements ---")
    abs_err_max = 0.0
    for ir in range(exp_1.shape[0]):
        for ic in range(exp_1.shape[1]):
            diff_comp = exp_1[ir,ic] - exp_2[ir,ic]
            abs_err = np.abs(diff_comp)
            if abs_err > abs_err_max:
                abs_err_max = abs_err
            print("[{:d},{:d}]: {:20.3e}".format(ir,ic, diff_comp))
    print()
    print("max. abs. error: {:0.3e}".format(abs_err_max))
    print("- log of max. abs. error: {:0.3f}".format(-np.log10(abs_err_max)))




# ------------------------------------------------------------------------------------------------
def compute_normalized_matrix(A, name_A):
    return mix.compute_normalized_matrix(A, name_A)


# ------------------------------------------------------------------------------------------------
def get_diag(A, i_shift):
    return mix.get_diag(A, i_shift)











