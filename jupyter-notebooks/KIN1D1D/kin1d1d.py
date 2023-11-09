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

    # !!!
    coef_norm_A = np.max(np.sqrt(np.sum(np.abs(A_values)**2)))

    values_norm = A_norm.get_values()
    values_norm *= coef_norm_D 
    if coef_norm_A > 1:
        values_norm /= coef_norm_A

    print()
    print("normalized matrix >>>")
    A_norm.print_max_min()
    return A_norm

# > A_F is a normalized F submatrix (first normalized to A-max and nonsparsity).
# > D_F is a part of the matrix D related to the F submatrix. 
def extract_fixed_profile_matrix_from_F(nx, nv, A_F, D_F):
    # --- Map the preliminary matrices to the sparse matrices ---
    def map_prel_to_sparse(Nnz, B_prel, step_c = 1):
        N_sections = B_prel.shape[0]
        N_loc      = B_prel.shape[1]
        # shift_section = (N_sections//2) * step_c

        i_diag_0 = N_sections//2
        shift_section = lambda i_diag: (i_diag - i_diag_0) * step_c

        rows    = np.zeros(N_loc+1, dtype=int)
        columns = np.zeros(Nnz, dtype=int)
        values  = np.zeros(Nnz, dtype=complex)

        counter_v = -1
        for ir in range(N_loc):
            rows[ir] = counter_v + 1
            for i_diag in range(N_sections):
                v = B_prel[i_diag, ir]
                if not np.isnan(v):
                    counter_v += 1
                    # columns[counter_v] = ir - shift_section + i_diag
                    columns[counter_v] = ir + shift_section(i_diag)
                    values[counter_v] = v
        if (counter_v + 1) != Nnz:
            print("Error: something wrong")
            return
        rows[N] = Nnz   
        return mix.SparseMatrix(N, Nnz, rows, columns, values)


    # -------------------------------------------------------------------
    # --- second normalization ---
    B = A_F.copy()
    B_values = B.get_values()
    for ii in range(B.get_Nnz()):
        B_values[ii] /= D_F.v(ii)
    
    N = B.get_N()
    Nx, Nv = 1<<nx, 1<<nv
    Nvar = Nx*Nv
    
    # 7 is the number of sections (diagonals):
    # 0  ic = ir - 3
    # 3  ic = ir
    # 6  ic = ir + 3
    Nnz_fixed = 0
    B_prel_fixed = np.zeros((7, N), dtype=complex)
    B_prel_fixed.fill(np.nan)

    # 0  ic = ir - 2
    # 2  ic = ir
    # 4  ic = ir + 2
    Nnz_profile = 0
    B_prel_profile = np.zeros((5, N), dtype=complex)
    B_prel_profile.fill(np.nan)

    # *** Extract flat elements ***

    # --- diag points ---
    for ir in range(Nv//2, Nv*(Nx-1) + Nv//2):
        B_prel_fixed[3, ir] = B.get_matrix_element(ir, ir)
        # B_fixed[ir,ir] = B[ir,ir]
        Nnz_fixed += 1

    # --- left off-diag elements ---
    for ir in range(Nvar):
        if np.mod(ir,Nv):
            B_prel_fixed[2, ir] = B.get_matrix_element(ir, ir-1)
            # B_fixed[ir,ir-1] = B[ir,ir-1]
            Nnz_fixed += 1

    # --- right off-diag elements ---
    for ir in range(Nvar-1):
        if np.mod(ir+1,Nv):
            B_prel_fixed[4, ir] = B.get_matrix_element(ir, ir+1)
            # B_fixed[ir,ir+1] = B[ir,ir+1]
            Nnz_fixed += 1

    # --- additional points at left and right velocity edges ---
    for ir in range(0,Nvar+1,Nv):
        # left:
        if ir < Nvar:
            B_prel_fixed[5, ir] = B.get_matrix_element(ir, ir+2)
            B_prel_fixed[6, ir] = B.get_matrix_element(ir, ir+3)
            # B_fixed[ir,ir+2] = B[ir,ir+2]
            # B_fixed[ir,ir+3] = B[ir,ir+3]
            Nnz_fixed += 2
        # right:
        if ir > 0:
            kk = ir-1
            B_prel_fixed[1, kk] = B.get_matrix_element(kk, kk-2)
            B_prel_fixed[0, kk] = B.get_matrix_element(kk, kk-3)
            # B_fixed[kk,kk-2] = B[kk,kk-2]
            # B_fixed[kk,kk-3] = B[kk,kk-3]
            Nnz_fixed += 2

    # *** Extract profile elements ***
    
    # --- diag points at the left and right boundaries ---
    for ir in range(Nv//2):
        B_prel_profile[2, ir] = B.get_matrix_element(ir, ir)
        Nnz_profile += 1

    for ir in range(Nv*(Nx-1) + Nv//2, Nvar):
        B_prel_profile[2, ir] = B.get_matrix_element(ir, ir)
        Nnz_profile += 1

    # --- off-diag points at the left and right boundaries ---
    for ir in range(Nv//2):
        B_prel_profile[3, ir] = B.get_matrix_element(ir, ir + Nv)
        B_prel_profile[4, ir] = B.get_matrix_element(ir, ir + 2*Nv)
        Nnz_profile += 2

    for ir in range(Nv*(Nx-1) + Nv//2, Nvar):
        B_prel_profile[1, ir] = B.get_matrix_element(ir, ir - Nv)
        B_prel_profile[0, ir] = B.get_matrix_element(ir, ir - 2*Nv)
        Nnz_profile += 2

    # --- off-diag bulk points ---
    for ir in range(Nv, Nv*(Nx-1)):
        B_prel_profile[1, ir] = B.get_matrix_element(ir, ir - Nv)
        B_prel_profile[3, ir] = B.get_matrix_element(ir, ir + Nv)
        Nnz_profile += 2

    # *** Map the preliminary matrices to the sparse matrices ***
    if A_F.get_Nnz() != (Nnz_fixed + Nnz_profile):
        print("Error: incorrect splitting.")
        exit(-1)
    B_sparse_fixed   = map_prel_to_sparse(Nnz_fixed,   B_prel_fixed,   1)
    B_sparse_profile = map_prel_to_sparse(Nnz_profile, B_prel_profile, Nv)





    return B_sparse_fixed, B_sparse_profile


def get_B_C_matrix(nv, C):
    norm_d = (1./np.sqrt(2.))**(nv+2)
    B_C = C.copy()
    B_values = B_C.get_values()
    B_values /= norm_d
    return B_C


def get_B_S_matrix(S):
    B_S = S.copy()
    B_values = B_S.get_values()
    B_values[0] /= 0.50
    return B_S


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
        x_grid = np.array(bg["x"])
        v_grid = np.array(bg["v"])
        Nx = len(x_grid)
        Nv = len(v_grid)
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
        "x": x_grid, "v": v_grid,
        "Nx": Nx, "Nv": Nv
    })
    dd["regs"] = {"rx": nx, "rv": nv}
    print("Done.\n")
    return dd


def plot_colored_A_structure(
        Nx, Nv, F_fixed, F_profile, A_CE, A_Cf, A_S, 
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
    rows_Fp, cols_Fp = find_rows_columns(F_profile.form_dense_matrix())
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


def plot_colored_F_structure(
        Nx, Nv, F_fixed, F_profile, 
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
    rows_Fp, cols_Fp = find_rows_columns(F_profile.form_dense_matrix())


    fig1 = plt.figure(figsize=(10,10))
    ax = fig1.add_subplot(111)

    marker_size = 4

    # combination of these two plots show in profile-elements in red and 
    # flat-elements in blue:
    ax.scatter(cols_Fp, rows_Fp, color="red", s = marker_size) 
    ax.scatter(cols_Ff, rows_Ff, color="blue", s = marker_size)

    plt.xlim(-2, Ns+2)
    plt.ylim(-2, Ns+2)
    
    plt.gca().invert_yaxis()
    plt.xlabel('columns', fontsize = fontsize)
    plt.ylabel("rows", fontsize = fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # draw blocks' boundaries:
    for ii in range(Nx+1):
        ax.axvline(
            x = ii*Nv-0.5, 
            ymin = 0.0, ymax = 1.0, 
            color = 'gray', linewidth = 1, linestyle = "--"
        )
        ax.axhline(
            y = ii*Nv-0.5, 
            xmin = 0.0, xmax = 1.0, 
            color = 'gray', linewidth = 1, linestyle = "--"
        )
    plt.show()

    if flag_save:
        if path_save is None:
            print("Error: a path for saveing a figure is not given.")
            return
        plt.savefig(path_save + "/" + "A-colored-structure-nx{:d}-nv{:d}.png".format(nx, nv))
    
    return



def compare_reconstructed(Ar, Ao, prec = 1e-8):
    Nr = len(Ar.get_values())
    No = len(Ao.get_values())

    print()
    if Nr == No:
        print("The same size.")
        print("N_nz = {:d}".format(Nr))
    else:
        print("Different sizes.") 
        
    print()
    flag_same = True
    for ii in range(Nr):
        if not mix.compare_complex_values(
            Ar.get_values()[ii], 
            Ao.get_values()[ii], 
            1e-8
        ):
            print(
                np.abs(Ar.get_values()[ii] - Ao.get_values()[ii])
            )
            flag_same = False
            break
            
    if flag_same:
        print("The values are the same within the precision {:0.1e}.".format(prec))
    else:
        print("The absolute difference between values is larger than {:0.1e}.".format(prec))
    return 


def print_compare_reconstructed(nv, Ar, Ao, ir_start, ic_start, flag_only_real = False):
    Nv = 1<<nv
    
    r0, c0 = ir_start*Nv, ic_start*Nv
    ff = [10, 3, "e"]
    
    print("\n--- I: real ---")
    Ar.print_matrix_real(r0, c0, Nv,ff, Nv, " ")
    print("\n\n--- II: real ---")
    Ao.print_matrix_real(r0, c0, Nv,ff, Nv, " ")
    
    if not flag_only_real:
        print("\n--- I: imag ---")
        Ar.print_matrix_imag(r0, c0, Nv,ff, Nv, " ")
        print("\n\n--- II: imag ---")
        Ao.print_matrix_imag(r0, c0, Nv,ff, Nv, " ")


def plot_compare_reconstructed(
        nv, Ar, Ao, 
        ir_start, ic_start, 
        label_Ar, label_Ao, 
        flag_only_real = False
    ):
    Nv = 1<<nv
    r0, c0 = ir_start*Nv, ic_start*Nv
    one_block_B1 = np.array(Ar.get_slice(r0, c0, Nv).get_values())
    one_block_B2 = np.array(Ao.get_slice(r0, c0, Nv).get_values())
    if len(one_block_B1) != len(one_block_B2):
        print("Error: different number of elements in the block.")
        return
    
    ids_x = range(len(one_block_B1))
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ids_x, one_block_B1.real, "b", linewidth = 2, linestyle='-',  label="{:s}".format(label_Ar))
    ax.plot(ids_x, one_block_B2.real, "r", linewidth = 2, linestyle='--', label="{:s}".format(label_Ao))
    plt.ylabel("prof(v): real")
    ax.legend()
    plt.grid(True)
    plt.show()
    
    if not flag_only_real:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(ids_x, one_block_B1.imag, "b", linewidth = 2, linestyle='-',  label="{:s}".format(label_Ar))
        ax.plot(ids_x, one_block_B2.imag, "r", linewidth = 2, linestyle='--', label="{:s}".format(label_Ao))
        plt.ylabel("prof(v): imag")
        ax.legend()
        plt.grid(True)
        plt.show()
    return

