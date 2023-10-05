import numpy as np
import importlib as imp
import h5py
# import pylib.Global_variables as GLO
import matplotlib.pyplot as plt
import sys
from numba import jit

import pylib.mix as mix
import pylib.qucf_oracle as qucf_o


def reload():
    mix.reload_module(mix)
    mix.reload_module(qucf_o)
    return


def init_circuit_kin(dd_q):
    circ = qucf_o.Circuit__()
    anc = qucf_o.Regs__()
    inp = qucf_o.Regs__()
    
    anc.add_reg("axr", dd_q["regs"]["axr"], True)
    anc.add_reg("avr", dd_q["regs"]["avr"], True)

    inp.add_reg("rx", dd_q["regs"]["rx"], False)
    inp.add_reg("rv", dd_q["regs"]["rv"], False)

    circ.set_regs(inp, anc)
    circ.compute_N_registers()
    return circ


def init_circuit_of_defined_size(nx, nv, nx_rel = 3, nv_rel = 3):
    circ = qucf_o.Circuit__()
    anc = qucf_o.Regs__()
    inp = qucf_o.Regs__()
    
    anc.add_reg("axr", nx_rel, True)
    anc.add_reg("avr", nv_rel, True)

    inp.add_reg("rx", nx, False)
    inp.add_reg("rv", nv, False)

    circ.set_regs(inp, anc)
    circ.compute_N_registers()
    return circ


def init_matrix_and_circuit(dd):
    circ = init_circuit_kin(dd)

    N_input_regs = len(circ.input_regs_.names_)
    N = 1
    size_D = [None] * N_input_regs
    for i_reg in range(N_input_regs): # from the most to the least significant qubit;
        N1 = 1 << circ.input_regs_.nqs_[i_reg]
        N *= N1
        size_D[i_reg] = N1 
    D_F = dd["A"].get_slice(0,0,N)
    return [circ, D_F, size_D]


def normalize_matrix_A(A, D_F, nv):
    # --- original matrix ---
    print("original matrix >>>")
    A.print_max_min()

    A_norm = A.copy()

    A_values = A.get_values()
    D_values = D_F.get_values()

    # --- normalization ---
    coef_d = 1./np.sqrt(2)

    coef_norm_D_1 = coef_d**(nv + 2)
    coef_norm_D_2 = np.min(np.min(np.abs(D_values[np.nonzero(D_values)])))
    coef_norm_D = np.min([coef_norm_D_1, coef_norm_D_2])

    coef_norm_A = np.max(np.sqrt(np.sum(np.abs(A_values)**2)))

    values_norm = A_norm.get_values()
    values_norm *= coef_norm_D 
    if coef_norm_A > 1:
        values_norm /= coef_norm_A

    print()
    print("normalized matrix >>>")
    A_norm.print_max_min()
    return A_norm


def extract_fixed_matrix_from_F(nx, nv, A, D):
    B = A.copy()
    B_values = B.get_values()
    for ii in range(B.get_Nnz()):
        B_values[ii] /= D.v(ii)
    
    N = B.get_N()
    Nx, Nv = 1<<nx, 1<<nv
    Nvar = Nx*Nv
    Nnz_new = 0

    # 7 is the number of sections (diagonals):
    # 0  ic = ir - 3
    # 3  ic = ir
    # 6  ic = ir + 3
    B_prel = np.zeros((7, N), dtype=complex)
    B_prel.fill(np.nan)

    # --- diag points ---
    for ir in range(Nv//2, Nv*(Nx-1) + Nv//2):
        B_prel[3, ir] = B.get_matrix_element(ir, ir)
        # B_fixed[ir,ir] = B[ir,ir]
        Nnz_new += 1

    # --- left off-diag elements ---
    for ir in range(Nvar):
        if np.mod(ir,Nv):
            B_prel[2, ir] = B.get_matrix_element(ir, ir-1)
            # B_fixed[ir,ir-1] = B[ir,ir-1]
            Nnz_new += 1

    # --- right off-diag elements ---
    for ir in range(Nvar-1):
        if np.mod(ir+1,Nv):
            B_prel[4, ir] = B.get_matrix_element(ir, ir+1)
            # B_fixed[ir,ir+1] = B[ir,ir+1]
            Nnz_new += 1

    # --- additional points at left and right velocity edges ---
    for ir in range(0,Nvar+1,Nv):
        # left:
        if ir < Nvar:
            B_prel[5, ir] = B.get_matrix_element(ir, ir+2)
            B_prel[6, ir] = B.get_matrix_element(ir, ir+3)
            # B_fixed[ir,ir+2] = B[ir,ir+2]
            # B_fixed[ir,ir+3] = B[ir,ir+3]
            Nnz_new += 2
        # right:
        if ir > 0:
            kk = ir-1
            B_prel[1, kk] = B.get_matrix_element(kk, kk-2)
            B_prel[0, kk] = B.get_matrix_element(kk, kk-3)
            # B_fixed[kk,kk-2] = B[kk,kk-2]
            # B_fixed[kk,kk-3] = B[kk,kk-3]
            Nnz_new += 2

    # map the preliminary matrix to the sparse matrix:
    rows_new    = np.zeros(N+1,     dtype=int)
    columns_new = np.zeros(Nnz_new, dtype=int)
    values_new  = np.zeros(Nnz_new, dtype=complex)

    counter_v = -1
    for ir in range(N):
        rows_new[ir] = counter_v + 1
        for i_diag in range(7):
            v = B_prel[i_diag, ir]
            if not np.isnan(v):
                counter_v += 1
                columns_new[counter_v] = ir - 3 + i_diag
                values_new[counter_v] = v
    if (counter_v + 1) != Nnz_new:
        print("Error: something wrong")
        return
    rows_new[N] = Nnz_new   
    B_fixed = mix.SparseMatrix(N, Nnz_new, rows_new, columns_new, values_new)
    return B_fixed, B


def extract_fixed_matrix_from_F_DENSE(nx, nv, A, D):
    # B = A/D  # have NaN where D had zeros;
    B = np.divide(A, D, out=np.zeros_like(A), where = D!=0)
    B_fixed = np.zeros_like(B)
    Nx, Nv = 1<<nx, 1<<nv
    Nvar = Nx*Nv

    N_nonzero = 0

    # --- diag points ---
    for ir in range(Nv//2, Nv*(Nx-1) + Nv//2):
        B_fixed[ir,ir] = B[ir,ir]
        N_nonzero += 1

    # --- left off-diag elements ---
    for ir in range(Nvar):
        if np.mod(ir,Nv):
            B_fixed[ir,ir-1] = B[ir,ir-1]
            N_nonzero += 1

    # --- right off-diag elements ---
    for ir in range(Nvar-1):
        if np.mod(ir+1,Nv):
            B_fixed[ir,ir+1] = B[ir,ir+1]
            N_nonzero += 1

    # --- additional points at left and right velocity edges ---
    for ir in range(0,Nvar+1,Nv):
        # left:
        if ir < Nvar:
            B_fixed[ir,ir+2] = B[ir,ir+2]
            B_fixed[ir,ir+3] = B[ir,ir+3]
            N_nonzero += 2
        # right:
        if ir > 0:
            kk = ir-1
            B_fixed[kk,kk-2] = B[kk,kk-2]
            B_fixed[kk,kk-3] = B[kk,kk-3]
            N_nonzero += 2

    print("N-nonzero: ", int(N_nonzero))
    return B_fixed




def read_matrix_sparse(path, name_file, name_matrix = "A"):
    fname = path + "/" + name_file
    dd = {}
    print("Reading data from {:s}...".format(name_file))
    with h5py.File(fname, "r") as f:
        # ---
        bg          = f["basic"]
        date_sim    = bg["date-of-simulation"][()].decode("utf-8")
        #---
        bg = f["matrices"]
        N   = bg[name_matrix + "-N"][()]
        Nnz = bg[name_matrix + "-Nnz"][()]
        columns = bg[name_matrix + "-columns"][()]
        rows    = bg[name_matrix + "-rows"][()]
        values_array = bg[name_matrix + "-values"][()]

        #---
        bg = f["grids"]
        x = np.array(bg["x"])
        v = np.array(bg["v"])
        Nx = len(x)
        Nv = len(v)
        nx = int(np.log2(Nx))
        nv = int(np.log2(Nv))
    # ---
    print("date of the simulation: ", date_sim)
    print("N, nx, nv = {:d}, {:d}, {:d}".format(N, nx, nv))

    values = np.zeros(Nnz, dtype=complex)
    for ii in range(len(values_array)):
        v = values_array[ii]
        values[ii] = complex(v[0], v[1])

    # Save the sparse matrix:
    dd["A"] = mix.SparseMatrix(N, Nnz, rows, columns, values)

    # add several variables for consistency with quantum computing:
    dd.update({
        "x": x, "v": v,
        "Nx": Nx, "Nv": Nv
    })
    dd["regs"] = {"rx": nx, "rv": nv}
    print("Done.\n")
    return dd


def plot_colored_A_structure(
        Nx, Nv, F_fixed, D_F, A_CE, A_Cf, A_S, 
        flag_save = False, path_save = None, 
        fontsize = 20, cmap='bwr'
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

    Ns = F_fixed.get_N()
    if Ns != Nx*Nv:
        print("Error: incorrect sizes Nx, Nv are give.")
        return
    nx = int(np.log2(Nx))
    nv = int(np.log2(Nv))

    rows_Ff, cols_Ff = find_rows_columns(F_fixed.form_dense_matrix())
    rows_Fp, cols_Fp = find_rows_columns(D_F.form_dense_matrix())
    rows_CE, cols_CE = find_rows_columns(A_CE.form_dense_matrix())
    rows_Cf, cols_Cf = find_rows_columns(A_Cf.form_dense_matrix())
    rows_S,  cols_S  = find_rows_columns(A_S.form_dense_matrix())

    fig1 = plt.figure(figsize=(10,10))
    ax = fig1.add_subplot(111)

    marker_size = 4

    # combination of these two plots show in profile-elements in red and 
    # flat-elements in blue:
    ax.scatter(cols_Fp, rows_Fp, color="red", s = marker_size) 
    ax.scatter(cols_Ff, rows_Ff, color="blue", s = marker_size)

    ax.scatter(Ns + cols_CE, rows_CE,      color="green", s = marker_size)
    ax.scatter(cols_Cf,      Ns + rows_Cf, color="green", s = marker_size)

    ax.scatter(Ns + cols_S, Ns + rows_S, color="blue", s = marker_size)

    plt.xlim(0, 2*Ns)
    plt.ylim(0, 2*Ns)
    
    plt.gca().invert_yaxis()
    plt.xlabel('columns', fontsize = fontsize)
    plt.ylabel("rows", fontsize = fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # draw submatrices' boundaries:
    ax.axvline(x = Ns-0.5, color = 'black', linewidth = 0.5, linestyle = "-")
    ax.axhline(y = Ns-0.5, color = 'black', linewidth = 0.5, linestyle = "-")

    # text submatrices' names:
    ax.text(49,  14, r'$\textbf{F}$', fontsize=fontsize)
    ax.text(121, 14, r'$\textbf{C}^E$', fontsize=fontsize)
    ax.text(49, 77, r'$\textbf{C}^f$', fontsize=fontsize)
    ax.text(121, 77, r'$\textbf{S}$', fontsize=fontsize)

    # draw blocks' boundaries:
    for ii in range(1, Nx):
        ax.axvline(x = ii*Nv-0.5, ymin = 0.5, ymax = 1.0, color = 'gray', linewidth = 1, linestyle = "--")
        ax.axhline(
            y = ii*Nv-0.5, 
            xmin = 0.0, xmax = 0.5, 
            color = 'gray', linewidth = 1, linestyle = "--"
        )
    for ii in range(1, Nx):
        ax.axvline(
            x = Nx*Nv + ii*Nv-0.5, 
            ymin = 0.5, ymax = 1.0, 
            color = 'gray', linewidth = 1, linestyle = "--"
        )
        ax.axhline(
            y = ii*Nv-0.5, 
            xmin = 0.5, xmax = 1.0, 
            color = 'gray', linewidth = 1, linestyle = "--"
        )
    for ii in range(1, Nx):
        ax.axvline(
            x = ii*Nv-0.5, 
            ymin = 0.0, ymax = 0.5, 
            color = 'gray', linewidth = 1, linestyle = "--")
        ax.axhline(
            y = Nx*Nv + ii*Nv-0.5, 
            xmin = 0.0, xmax = 0.5, 
            color = 'gray', linewidth = 1, linestyle = "--"
        )

    plt.show()

    if flag_save:
        if path_save is None:
            print("Error: a path for saveing a figure is not given.")
            return
        plt.savefig(path_save + "/" + "A-colored-structure-nx{:d}-nv{:d}.png".format(nx, nv))
    
    return