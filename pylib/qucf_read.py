import numpy as np
import importlib as imp
import h5py
import matplotlib.pyplot as plt
import sys
from numba import jit
import cmath
import pylib.mix as mix

   


def reload():
    mix.reload_module(mix)
    return



# read a matrix and store it in a CSR (comprassed sparse row) format
def read_matrix_store_sparse(path, name, flag_also_complex_conjugate = False):
    print("Reading the matrix from: " + name)
    print("from the path: " + path)
    
    fname = path + "/" + name
    with h5py.File(fname, "r") as f:
        # ---
        bg        = f["basic"]
        date_sim  = bg["date-of-simulation"][()].decode("utf-8")
        reg_names = bg["register-names"][()].decode("utf-8").split(", ")
        regs_nq   = bg["register-nq"][...]
        #---
        bg = f["matrix"]
        name_matrix = bg["name-oracle"][()].decode("utf-8")
        N = bg["N"][()]
        A_real = np.array(bg["real"])
        A_imag = np.array(bg["imag"])
    print("date of the simulation: ", date_sim)
    print("matrix name: ", name_matrix)
    print("N = {:d}".format(N))

    # qubit registers:
    reg_shift = 0
    regs = {}
    reg_shifts = {}
    for i in range(len(regs_nq)):
        regs[reg_names[i]] = regs_nq[i]
        reg_shifts[reg_names[i]] = reg_shift
        reg_shift += regs_nq[i]

    # form the sparse matrix:
    Nnz = 0
    A_values = np.zeros(N*N, dtype=complex)
    A_columns = np.zeros(N*N, dtype=int)
    A_rows = np.zeros(N+1, dtype=int)
    for ir in range(N):
        A_rows[ir] = Nnz
        for ic in range(N):
            ar = A_real[ir * N + ic]
            ai = A_imag[ir * N + ic]
            if np.abs(ar) > 0 or np.abs(ai) > 0:
                Nnz += 1
                A_values[Nnz-1] = ar + 1j * ai
                A_columns[Nnz-1] = ic
    A_rows[N] = Nnz
    A_values  = A_values[:Nnz]
    A_columns = A_columns[:Nnz]
    ddA = {
        "N": N,
        "regs": regs,
        "reg-shifts": reg_shifts,
        "reg-names":  reg_names,
        "A-values":  A_values,
        "A-rows":    A_rows,
        "A-columns": A_columns
    }
    return ddA


def read_matrix(path, name):
    print("Reading the matrix from: " + name)
    print("from the path: " + path)
    
    fname = path + "/" + name
    with h5py.File(fname, "r") as f:
        # ---
        bg        = f["basic"]
        date_sim  = bg["date-of-simulation"][()].decode("utf-8")
        reg_names = bg["register-names"][()].decode("utf-8").split(", ")
        regs_nq   = bg["register-nq"][...]
        #---
        bg = f["matrix"]
        name_matrix = bg["name-oracle"][()].decode("utf-8")
        N = bg["N"][()]
        A_real = np.array(bg["real"])
        A_imag = np.array(bg["imag"])
    print("date of the simulation: ", date_sim)
    print("matrix name: ", name_matrix)
    print("N = {:d}".format(N))

    # qubit registers:
    reg_shift = 0
    regs = {}
    reg_shifts = {}
    for i in range(len(regs_nq)):
        regs[reg_names[i]] = regs_nq[i]
        reg_shifts[reg_names[i]] = reg_shift
        reg_shift += regs_nq[i]

    # form the matrix:
    A  = np.zeros((N, N), dtype=complex)
    Ar = np.zeros((N, N))
    Ai = np.zeros((N, N))

    Arcc = np.zeros((N, N))
    Aicc = np.zeros((N, N))
    Acc = np.zeros((N, N), dtype=complex)
    for ir in range(N):
        for ic in range(N):
            Ar[ir,ic]   = A_real[ir * N + ic]
            Ai[ir,ic]   = A_imag[ir * N + ic]
            A[ir, ic]   = Ar[ir,ic] + 1j * Ai[ir,ic]

            Arcc[ic, ir] = A_real[ir * N + ic]
            Aicc[ic, ir] = - A_imag[ir * N + ic]
            Acc[ic, ir]  = Ar[ir,ic] - 1j * Ai[ir,ic]
            
    # form the matrix mask (structure):
    A_mask = np.zeros((N, N))
    A_mask_cc = np.zeros((N, N))
    for ir in range(N):
        for ic in range(N):
            if( np.abs(A[ir, ic]) > 0.0 ):
                A_mask[ir, ic] = 1
                A_mask_cc[ic, ir] = 1
    ddA = {
        "N": N,
        "regs": regs,
        "reg-shifts": reg_shifts,
        "reg-names": reg_names,
        "A-real": Ar, "A-imag": Ai, "A": A,
        "A-real-cc": Arcc, "A-imag-cc": Aicc, "A-cc": Acc,
        "A-mask": A_mask,
        "A-mask-cc": A_mask_cc
    }
    return ddA


def plot_matrix(B, fontsize = 20, cmap='bwr'):
    # cmap = 'seismic'
    # cmap = 'bwr'
    # cmap = 'bwr'
    # cmap = 'jet'
    # cmap = 'coolwarm'

    Br_max = np.max(np.max( np.abs(B.real) ))

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    cs = ax.matshow(B.real, cmap=cmap, vmin=-Br_max, vmax = Br_max)
    plt.xlabel('columns', fontsize = fontsize)
    plt.ylabel("rows", fontsize = fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
        
    return fig1





