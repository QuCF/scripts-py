from warnings import resetwarnings
import numpy as np
import scipy.special as ssp
import scipy.constants as sc
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets
import pylib.mix as mix
import time
import scipy.sparse.linalg as spal

import pylib.mix as mix

me = sc.m_e * 1e3 # in gramms
qe = sc.e * 3e9  # in statcoulomb
c_light = sc.c * 1e2 # cm/s
pi4 = 4*np.pi


def reload():
    mix.reload_module(mix)

def init(nqx, nq_layers, epss, r0, kx):
    # nqx - number of qubits to store the spatial grid;
    # nq_layers = log_e(N_layers), where N_layers is the number of dielectric layers;
    # epss: electric permitivies of each dielectric layer (array of size 1<<nq_layers)
    # r0 - spatial width;
    # kx - wavelength in vacuum;

    coef_superposition = 2

    N_layers = 1 << nq_layers

    Nx = 1 << nqx
    Nx_layer = 1 << (nqx - nq_layers)
    print("Modeling: Nx = {:d}".format(Nx))

    if len(epss) is not N_layers:
        print("ERROR: number of the permittivities must be equal to the number of layers.")
        return None

    dd_init = {
        "r0": r0, "nqx": nqx, "Nx": Nx,
        "kx": kx,
        "coef-superposition": coef_superposition,
        "nq-layers": nq_layers, "N-layers": N_layers,
        "Nx-layer": Nx_layer,
        "epss": list(epss)
    }
    dd = create(dd_init)
    return dd


def create(dd):
    # input parameters:
    r0 = dd["r0"]
    kx = dd["kx"]
    Nx = dd["Nx"]
    coef_superposition = dd["coef-superposition"] 

    # normalization factors:
    w_ref  = kx * c_light  # rad/s
    coef_r_norm = 1. / kx
    coef_t_norm = 1. / w_ref

    # space coordinate:
    x = np.linspace(0, r0, Nx) 
    x_norm = x / coef_r_norm
    x_plot = x / r0
    h = (x_norm[1] - x_norm[0])/2.
    dx = h * coef_r_norm  # in cm
    h_plot = dx / r0
    wavelen = 2*np.pi/kx
    w_ref_norm = w_ref * coef_t_norm
    
    # normalized variables:
    kx_norm = kx * coef_r_norm
    kx_plot = kx * r0

    # QC normalization of the system matrix:
    w_epss = []
    for ii in range(dd["N-layers"]):
        w_epss.append(w_ref_norm * dd["epss"][ii])

    max_value = np.max(
        [np.abs(2./h), np.abs(1j*w_ref_norm + 1./h), w_ref_norm] + w_epss
    )

    norm_of_h = max_value * coef_superposition**2

    # results:
    res = {
        'x': x, 'x-norm': x_norm, 'x-plot': x_plot, 'dx': dx, 
        'dx-norm': h, "dx-plot": h_plot,
        'kx-norm': kx_norm, 'kx-plot': kx_plot,
        'w-ref':   w_ref,   'w-ref-norm': w_ref_norm,
        'coef-t-norm': coef_t_norm, 
        'coef-r-norm': coef_r_norm,  
        'norm-of-h': norm_of_h, 
        'max-value': max_value,
        "1/2h": 1./(2*h),
        "wavelength": wavelen,
        "N-vars": 2
    }
    res.update(dd)
    return res


def form_matrix(dd, flag_norm = True):
    N_layers = dd["N-layers"]
    Nx_layer = dd["Nx-layer"]
    epss = dd["epss"]
    Nx = dd["Nx"] 
    w_ref_norm = dd["w-ref-norm"]
    N_vars = dd["N-vars"]
    h = dd["dx-norm"]

    ih = 1./h
    i2h = 1./(2*h)

    # -------------------------------------------------------------
    # --- Matrix of the system ---
    A = np.zeros([N_vars*Nx, N_vars*Nx], dtype = np.complex)

    # --- Block(0,0) ---
    idr, idc = 0, 0
    for i_layer in range(N_layers):
        eps_layer = epss[i_layer]
        for ix in range(Nx_layer):
            A[i_layer*Nx_layer + ix, i_layer*Nx_layer + ix] = 1j*w_ref_norm*eps_layer
    A[0, 0] = 1j*w_ref_norm + ih
    A[0, 1] = 1j*w_ref_norm - ih

    # --- Block(0,1) ---
    idr, idc = 0, 1
    for i in range(1,Nx):
        A[idr*Nx + i, idc*Nx + i] = i2h
        A[idr*Nx + i, idc*Nx + i - 1] = - i2h

    # --- Block(1,0) ---
    idr, idc = 1, 0
    for i in range(Nx-1):
        A[idr*Nx + i, idc*Nx + i] = - i2h
        A[idr*Nx + i, idc*Nx + i + 1] = i2h

    # --- Block(1,1) ---
    idr, idc = 1, 1
    for i in range(Nx):
        A[idr*Nx + i, idc*Nx + i] = 1j*w_ref_norm
    A[idr*Nx + Nx-1, idc*Nx + Nx-1] = 1j*w_ref_norm + ih
    A[idr*Nx + Nx-1, idc*Nx + Nx-2] = 1j*w_ref_norm - ih

    # -------------------------------------------------------------
    # --- RHS: source ---
    b = np.zeros(N_vars * Nx)
    b[N_vars * Nx - 1] = 1  # magnetic source on the right side;
    # b[Nx + int(Nx/2)] = 1  # magnetic source on the right side;
    # b[Nx-1] = 1  # electric source on the right side; 

    # -------------------------------------------------------------
    # --- Normalization ---
    if flag_norm:
        A = A / dd["norm-of-h"]

    print()
    print("i*w + ih: {:20.3e}".format(1j*w_ref_norm + ih))
    print("i2h: {:0.3e}".format(i2h))
    print()

    return A, b


def print_get_norm_h_element_EB(dd):
    N_layers = dd["N-layers"]
    Nx_layer = dd["Nx-layer"]

    norm = dd["norm-of-h"]
    h = dd["dx-norm"]
    ih = (1./h) / norm
    i2h = (1./(2*h)) / norm
    w = dd["w-ref-norm"]/norm
    w_epss = w * np.array(dd["epss"])

    wph = w * 1j + ih
    wmh = w * 1j - ih

    print("N-layers: {:d}".format(N_layers))
    print("Nx in one layer: {:d}".format(Nx_layer))
    print('max. value for norm: {:0.3e}'.format(dd["max-value"]))
    print('norm value: {:0.3e}'.format(norm))
    print("coef-superposition: {:d}".format(dd["coef-superposition"]))
    print()
    print("w: \t{:0.3e}".format(w))
    for ii in range(N_layers):
        print("{:d}-th layer: w * eps: \t{:0.3e}".format(ii, w_epss[ii]))
    print("1/(2h): \t{:0.3e}".format(i2h))
    print("w*j + 1/h: \t{:16.3e}".format(wph))
    print("w*j - 1/h: \t{:16.3e}".format(wmh))

    return w, w_epss, i2h, wph, wmh


def print_matrix_block(dd1, A1, id_row, id_column):
    block = A1[
        dd1["Nx"]*id_row:dd1["Nx"]*(id_row + 1),
        dd1["Nx"]*id_column:dd1["Nx"]*(id_column+1)
    ]
    print("--- Block({:d}, {:d}) ---\n".format(id_row, id_column))
    mix.print_matrix(block, ff=[10,3,"e"], gap_be="")   


def print_max_row_norm(dd, A_norm):
    Nx = dd["Nx"]
    N_vars = dd["N-vars"]
    norm_rows = np.zeros(N_vars * Nx)
    for ii in range(N_vars * Nx):
        norm_rows[ii] = np.sqrt(np.sum(np.abs(A_norm[ii,:])**2))
    print("max-row-norm: {:0.3e}".format(np.max(norm_rows)))
    return
    

def form_matrix_dilated(dd, id_eq_B, N_copies):
    # id_eq_B in [0, Nx]

    N_layers = dd["N-layers"]
    Nx_layer = dd["Nx-layer"]
    epss = dd["epss"]
    Nx = dd["Nx"] 
    w_ref_norm = dd["w-ref-norm"]

    h = dd["dx-norm"]
    ih = 1./h
    i2h = 1./(2*h)

    N_vars = 2

    # -------------------------------------------------------------
    # --- Matrix of the system ---
    A = np.zeros([N_vars*Nx + N_copies, N_vars*Nx + N_copies], dtype = np.complex)

    # --- Block(0,0) ---
    idr, idc = 0, 0
    for i_layer in range(N_layers):
        eps_layer = epss[i_layer]
        for ix in range(Nx_layer):
            A[i_layer*Nx_layer + ix, i_layer*Nx_layer + ix] = 1j*w_ref_norm*eps_layer
    A[0, 0] = 1j*w_ref_norm + ih
    A[0, 1] = 1j*w_ref_norm - ih

    # --- Block(0,1) ---
    idr, idc = 0, 1
    for i in range(1,Nx):
        A[idr*Nx + i, idc*Nx + i] = i2h
        A[idr*Nx + i, idc*Nx + i - 1] = - i2h

    # --- Block(1,0) ---
    idr, idc = 1, 0
    for i in range(Nx-1):
        A[idr*Nx + i, idc*Nx + i] = - i2h
        A[idr*Nx + i, idc*Nx + i + 1] = i2h

    # --- Block(1,1) ---
    idr, idc = 1, 1
    for i in range(Nx):
        A[idr*Nx + i, idc*Nx + i] = 1j*w_ref_norm
    A[idr*Nx + Nx-1, idc*Nx + Nx-1] = 1j*w_ref_norm + ih
    A[idr*Nx + Nx-1, idc*Nx + Nx-2] = 1j*w_ref_norm - ih

    # --- Extension of the matrix ---
    # add equations B_i - B_{id_eq_B} = 0, B_i - copies of B_{id_eq_B}
    idr, idc = 2, 2
    for i in range(N_copies):
        A[idr*Nx + i, idc*Nx + i] = 1
        A[idr*Nx + i, Nx + id_eq_B] = -1

    # -------------------------------------------------------------
    # --- RHS: source ---
    b = np.zeros(N_vars * Nx + N_copies)
    b[N_vars * Nx - 1] = 1  # magnetic source on the right side;

    return A, b


def solve_system(A, b, sel_method):
    psi = None

    if sel_method == 0:
        print("--- Solve the system by the Gaussian method ---")
        # --- Find the inverse matrix ---
        start = time.perf_counter()
        A_inv = np.linalg.inv(A)
        end   = time.perf_counter()
        print("time for inv. calc.: {:0.3e} s".format(end - start))

        # --- Find the system variables ---
        psi = np.dot(A_inv, b)

    if sel_method == 1:
        print("--- Solve the system by the BiCGSTAB method ---")
        start = time.perf_counter()
        psi, info = spal.bicgstab(A, b)
        end   = time.perf_counter()
        print("OUTPUT status from BiCGSTAB: ", info)
        print("time for inv. calc.: {:0.3e} s".format(end - start))

    if sel_method == 2:
        print("--- Solve the system by the GMRES method ---")
        start = time.perf_counter()
        psi, info = spal.gmres(A, b)
        end   = time.perf_counter()
        print("OUTPUT status from GMRES: ", info)
        print("time for inv. calc.: {:0.3e} s".format(end - start))

    return psi


def get_vars(psi, dd):
    Nx = dd["Nx"]

    Ey = psi[0:Nx]
    Bz = psi[Nx:2*Nx]

    xe = dd["x-plot"]
    xb = dd["dx-plot"] + xe
    
    vv = {
        "Ey": Ey,
        "Bz": Bz,
        "xe": xe, 
        "xb": xb
    }
    return vv