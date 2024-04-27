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


# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
class ASE_:
    # path to the final results from QuCF simulations: 
    path_qc_ = "../QuCF/simulations/EVM/"

    # path to the QuCF simulations performed without the oracle OH:
    path_D_ = "../QuCF/simulations/EVM/matrices-D/"
    path_cl_ = "../QuCF/simulations/EVM/classical-sims/"

    file_name_oracle_ = "circuit_OH"
    path_save_ = "../results/KIN1D1D-results/figs/"

    # output file with final QuCF simualtions of the ASE:
    output_qucf_ = "BE_OUTPUT.hdf5"

    cl_ = None

    dd_44_ = None
    dd_45_ = None
    dd_54_ = None
    dd_55_ = None

    oo_extr_ = None

    nx_work_ = None
    nv_work_ = None
    Nx_work_ = None
    Nv_work_ = None
    Nvar_work_ = None
    v_work_ = None

    # classical data for the chosen nx_work_ and nv_work_:
    cl_work_ = None 

    # parameters of the quantum circuit:
    circ_ = None

    # The submatrix of the matrix D (D is of size 2 * Nvar_work_) related to the submatrix F.
    # The submatrix DF_work_ is of size Nvar_work_.
    DF_work_ = None

    # The matrices _orig are submatrices of A after the first normalization:
    F_orig_, CE_orig_, Cf_orig_, S_orig_ = None, None, None, None

    # The matrices B are submatrices of A after the second normalization:
    BF_fixed_, BF_prof_ = None, None
    B_CE_, B_Cf_, BS_ = None, None, None


    def read_D_matrices(self):
        # Matrix D is a matrix representation of a part of a block encoding oracle 
        #   which does not include the suboracle OH.
        print()
        self.dd_33_ = qucf_r.read_matrix_sparse(self.path_D_, "flat_33_OUTPUT.hdf5") #  nx = 4, nv = 4

        print()
        self.dd_44_ = qucf_r.read_matrix_sparse(self.path_D_, "flat_44_OUTPUT.hdf5") #  nx = 4, nv = 4 

        print()
        self.dd_45_ = qucf_r.read_matrix_sparse(self.path_D_, "flat_45_OUTPUT.hdf5") #  nx = 4, nv = 5 

        print()
        self.dd_54_ = qucf_r.read_matrix_sparse(self.path_D_, "flat_54_OUTPUT.hdf5") #  nx = 5, nv = 4

        # print()
        # self.dd_55_ = qucf_r.read_matrix_sparse(self.path_D_, "flat_55_OUTPUT.hdf5") #  nx = 5, nv = 5
        return


    def read_plasma_matrices(self, flag_diff_zero_too = False):
        self.cl_ = {}
        print()

        # --- classical simulations with eta = -0.002 ---
        self.cl_["33"] = read_matrix_sparse(self.path_cl_, "out_3_3_w1.2_Lx100_Lv4_flat.hdf5")
        for ix in range(4, 10):
            for iv in range(4, 9):
                self.cl_["{:d}{:d}".format(ix, iv)] = read_matrix_sparse(
                    self.path_cl_, "out_{:d}_{:d}_w1.2_Lx100_Lv4_flat.hdf5".format(ix, iv)
                )

        # --- classical simulations with eta = 0.0 ---
        if flag_diff_zero_too:
            for ix in range(4, 10):
                for iv in range(4, 8):
                    self.cl_["{:d}{:d}-diff-zero".format(ix, iv)]= read_matrix_sparse(
                        self.path_cl_ + "/w12-zero-diff/", 
                        "out_{:d}_{:d}_w1.2_Lx100_Lv4_flat.hdf5".format(ix, iv)
                    )
        return
    

    def create_D_template(self):
        # --- Create a template to extrapolate the matrix D for larger sizes ---
        grid_44 = qucf_m.SectionsGrid__(init_matrix_and_circuit(self.dd_44_))
        grid_45 = qucf_m.SectionsGrid__(init_matrix_and_circuit(self.dd_45_))
        grid_54 = qucf_m.SectionsGrid__(init_matrix_and_circuit(self.dd_54_))

        self.oo_extr_ = qucf_m.Extrapolation__([grid_44, grid_45, grid_54])
        self.oo_extr_.create_extrapolation_template()
        return
    

    # Return the D matrix calculated by the quantum circuit.
    def get_original_D_matrix(self):
        D_orig = None
        if self.nx_work_ == 4 and self.nv_work_ == 4:
            D_orig = self.dd_44_["A"].get_slice(0, 0, self.Nvar_work_)
        if self.nx_work_ == 4 and self.nv_work_ == 5:
            D_orig = self.dd_45_["A"].get_slice(0, 0, self.Nvar_work_)
        if self.nx_work_ == 5 and self.nv_work_ == 4:
            D_orig = self.dd_54_["A"].get_slice(0, 0, self.Nvar_work_)
        return D_orig
    

    def choose_a_case(self, nx, nv):
        self.nx_work_, self.nv_work_ = int(nx), int(nv)
        self.cl_work_ = self.cl_["{:d}{:d}".format(nx, nv)]

        self.Nx_work_ = 1 << self.nx_work_
        self.Nv_work_ = 1 << self.nv_work_
        self.Nvar_work_ = self.Nx_work_ * self.Nv_work_
        self.v_work_ = self.cl_work_["v"]

        # compute the matrix DF of the chosen size:
        if self.nx_work_ == 3 and self.nv_work_ == 3:
            self.DF_work_ = self.dd_33_["A"]
        else:
            self.circ_ = init_circuit_of_defined_size(self.nx_work_, self.nv_work_, 3, 3)
            self.DF_work_ = self.oo_extr_.reconstruct_matrix(self.circ_)

        # first normalization: normalize the plasma matrix to nonsparsity and the matrix norm:
        A_norm = normalize_matrix_A(self.cl_work_["A"], self.DF_work_, self.nv_work_)

        # extract the submatrices:
        # here, the submatrices are secondly renormalized to the values of the matrix D
        # (which is necessary to compute parameters of quantum gates in the operator OH)
        # A = | F,  CE |
        #     | Cf, S  |
        self.F_orig_ = A_norm.get_slice(0, 0, self.Nvar_work_)
        self.BF_fixed_, self.BF_prof_ = extract_fixed_profile_matrix_from_F(
            self.nx_work_, self.nv_work_, self.F_orig_, self.DF_work_
        )

        self.CE_orig_ = A_norm.get_slice(0, self.Nvar_work_, self.Nvar_work_)
        self.B_CE_ = get_B_C_matrix(self.nv_work_, self.CE_orig_)

        self.Cf_orig_ = A_norm.get_slice(self.Nvar_work_, 0, self.Nvar_work_)
        self.B_Cf_ = get_B_C_matrix(self.nv_work_, self.Cf_orig_)

        self.S_orig_ = A_norm.get_slice(self.Nvar_work_, self.Nvar_work_, self.Nvar_work_)
        self.BS_ = get_B_S_matrix(self.nx_work_, self.nv_work_, self.S_orig_)
        return
    

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# --- FUNCTIONS FOR SUBMATRICIES' PARAMETERS ---
# Here, oo_ase is the object of the ASE_ class.
# --------------------------------------------------------------------------------------------
def preliminary_parameters_for_submatrices(oo_ase):
    Nx = oo_ase.Nx_work_
    Nv = oo_ase.Nv_work_
    Nvh = oo_ase.Nv_work_//2
    N = Nx*Nv

    # *** normalized v-grid ***
    v_grid_norm = np.array(oo_ase.B_Cf_.get_slice(0,0, oo_ase.Nv_work_).get_values().real)
    v_norm_max = np.max(np.abs(v_grid_norm))
    # print("v-norm_max: \t{:0.3e}".format(v_norm_max))

    v_bulk_S = oo_ase.BS_.get_matrix_element(1,1) 
    v_edge_S = oo_ase.BS_.get_matrix_element(0,0)
    # print("S: v-bulk: {:20.3e}".format(v_bulk_S))
    # print("S: v-edge: {:20.3e}".format(v_edge_S))

    coef_ratio = Nv/(Nv - 1)
    coef_ratio_2 = Nvh / (2.*(Nvh - 1))

    print("nx {:d}".format(oo_ase.nx_work_))
    print("nv {:d}".format(oo_ase.nv_work_))
    print("Nvhm1 {:d}".format(Nvh-1))
    print("Nxm2 {:d}".format(Nx-2))
    print("Nxm4 {:d}".format(Nx-4))
    print()
    print("Nvm1 {:d}".format(Nv-1))
    print("Nvm3 {:d}".format(Nv-3))
    print("Nvm4 {:d}".format(Nv-4))
    
    # --- Submatrix Cf ---
    # v-profile: assume that v << 1:
    # parameters for the SIN gate:
    alpha_0 = - v_norm_max
    alpha_1 = v_norm_max * coef_ratio
    print()
    print("//--- Parameters for the submatrix Cf ---")
    print("alpha_0_cf \t{:0.12e}".format(alpha_0))
    print("alpha_1_cf \t{:0.12e}".format(alpha_1))

    # --- Submatrix CE ---
    alpha_0 = - 1.0
    alpha_1 = np.abs(alpha_0) * coef_ratio
    print("\n//--- Parameters for the oracle for the submatrix CE ---")
    print("alpha_0_CE \t{:0.12e}".format(alpha_0))
    print("alpha_1_CE \t{:0.12e}".format(alpha_1))
    del alpha_0, alpha_1
   
    # --- Submatrix S ---
    angle_sb = - 2 * np.arcsin(np.imag(v_bulk_S))
    angle_se = - 2 * np.arcsin(np.imag(v_edge_S)) - angle_sb

    print("\n//--- Parameters for the submatrix S ---")
    print("angle_sb \t{:0.12f}".format(angle_sb))
    print("angle_se \t{:0.12f}".format(angle_se))

    # --- Submatrix F: profiles ---

    # - Left blocks FB1 -
    shift_x = Nx//2 * Nv
    v1 = oo_ase.BF_prof_.get_matrix_element(
        shift_x, shift_x - Nv
    )
    alpha_0_LFB1 = - np.abs(v1)
    alpha_1_LFB1 = np.abs(v1) * coef_ratio

    # corrections:
    shift_x = 3 * Nv
    v1 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x - Nv)
    shift_x = 3 * Nv + (Nvh-1)
    v2 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x - Nv)
    alpha_0_LFB1_corr1 = - np.abs(v1)
    alpha_1_LFB1_corr1 = np.abs(v2 - v1) * coef_ratio_2

    shift_x = 3 * Nv + Nvh
    v1 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x - Nv)
    shift_x = 3 * Nv + (Nv-1)
    v2 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x - Nv)
    alpha_0_LFB1_corr1_2 = np.abs(v1)
    alpha_1_LFB1_corr1_2 = np.abs(v2 - v1) * coef_ratio_2


    shift_x = (Nx-2) * Nv
    v1 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x - Nv)
    shift_x = (Nx-2) * Nv + (Nvh-1)
    v2 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x - Nv)
    alpha_0_LFB1_corr2 = - np.abs(v1)
    alpha_1_LFB1_corr2 = np.abs(v2 - v1) * coef_ratio_2

    shift_x = (Nx-2) * Nv + Nvh
    v1 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x - Nv)
    shift_x = (Nx-2) * Nv + (Nv-1)
    v2 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x - Nv)
    alpha_0_LFB1_corr2_2 = -np.abs(v1)
    alpha_1_LFB1_corr2_2 = -np.abs(v2 - v1) * coef_ratio_2

    shift_x = (Nx-1) * Nv + Nvh
    v1 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x - Nv)
    shift_x = (Nx-1) * Nv + (Nv-1)
    v2 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x - Nv)
    alpha_0_LFB1_corr3 = np.abs(v1)
    alpha_1_LFB1_corr3 = np.abs(v2 - v1) * coef_ratio_2

    # - Right blocks FB1 -
    shift_x = Nx//2 * Nv
    v1 = oo_ase.BF_prof_.get_matrix_element(
        shift_x, shift_x + Nv
    )
    alpha_0_RFB1 = np.abs(v1)
    alpha_1_RFB1 = -np.abs(v1) * coef_ratio

    # corrections:
    shift_x = 0
    v1 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x + Nv)
    shift_x = 0 + (Nvh-1)
    v2 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x + Nv)
    alpha_0_RFB1_corr1 = np.abs(v1)
    alpha_1_RFB1_corr1 = -np.abs(v2 - v1) * coef_ratio_2


    shift_x = Nv
    v1 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x + Nv)
    shift_x = Nv + (Nvh-1)
    v2 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x + Nv)
    alpha_0_RFB1_corr2 = np.abs(v1)
    alpha_1_RFB1_corr2 = -np.abs(v2 - v1) * coef_ratio_2

    shift_x = Nv + Nvh
    v1 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x + Nv)
    shift_x = Nv + (Nv-1)
    v2 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x + Nv)
    alpha_0_RFB1_corr2_2 = -np.abs(v1) # its a negative real value
    alpha_1_RFB1_corr2_2 = -np.abs(v2 - v1) * coef_ratio_2


    shift_x = (Nx-4) * Nv
    v1 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x + Nv)
    shift_x = (Nx-4) * Nv + (Nvh-1)
    v2 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x + Nv)
    alpha_0_RFB1_corr3 = np.abs(v1) 
    alpha_1_RFB1_corr3 = -np.abs(v2 - v1) * coef_ratio_2

    shift_x = (Nx-4) * Nv + Nvh
    v1 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x + Nv)
    shift_x = (Nx-4) * Nv + (Nv-1)
    v2 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x + Nv)
    alpha_0_RFB1_corr3_2 = - np.abs(v1) 
    alpha_1_RFB1_corr3_2 = - np.abs(v2 - v1) * coef_ratio_2


    shift_x = (Nx-2) * Nv
    v1 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x + Nv)
    shift_x = (Nx-2) * Nv + (Nvh-1)
    v2 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x + Nv)
    alpha_0_RFB1_corr4 = np.abs(v1) 
    alpha_1_RFB1_corr4 = -np.abs(v2 - v1) * coef_ratio_2

    shift_x = (Nx-2) * Nv + Nvh
    v1 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x + Nv)
    shift_x = (Nx-2) * Nv + (Nv-1)
    v2 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x + Nv)
    alpha_0_RFB1_corr4_2 = -np.abs(v1) 
    alpha_1_RFB1_corr4_2 = -np.abs(v2 - v1) * coef_ratio_2

    # - Block FL2 -
    v1 = oo_ase.BF_prof_.get_matrix_element(0, 2*Nv)
    alpha_0_FL2 = - np.abs(v1)
    alpha_1_FL2 = np.abs(v1) * coef_ratio

    # - Block FR2 - 
    v1 = oo_ase.BF_prof_.get_matrix_element(N-1, N-2*Nv-1)
    alpha_0_FR2 = - np.abs(v1)
    alpha_1_FR2 = np.abs(v1) * coef_ratio

    # --- Main Diag ---
    coef_lcu = 2.

    # -- left edge --
    # real sin
    shift_x = Nvh-2
    v1 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x)
    shift_x = Nvh-1
    v2 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x)

    v_end = v2.real
    dv = v_end - v1.real
    v_start = v_end - dv * (Nvh-1)

    a0_FB0_L = coef_lcu * v_start
    a1_FB0_L = coef_lcu * (v_end - v_start) * coef_ratio_2

    # imaginary part:
    a0_FB0_iw_L = - 2. * np.arcsin(coef_lcu * v1.imag) 

    # corrections for the LEFT diagonal:
    def find_corr_F_edges_LEFT(id_v, a0_FB0, a1_FB0):
        vv = oo_ase.BF_prof_.get_matrix_element(id_v, id_v)
        a_corr_i = - 2. * np.arcsin(coef_lcu * vv.imag)
        a_inv = 2. * (a0_FB0 + id_v * (2. * a1_FB0) / Nvh)
        a_corr_r = 2. * np.arccos(coef_lcu * vv.real)
        return a_inv, a_corr_r, a_corr_i
    
    aL_inv_v0, aL_corr_v0_r, aL_corr_v0_i = find_corr_F_edges_LEFT(0, a0_FB0_L, a1_FB0_L)
    aL_inv_v2, aL_corr_v2_r, aL_corr_v2_i = find_corr_F_edges_LEFT(2, a0_FB0_L, a1_FB0_L)
    aL_inv_v3, aL_corr_v3_r, aL_corr_v3_i = find_corr_F_edges_LEFT(3, a0_FB0_L, a1_FB0_L)

    # -- right edge --
    shift_x = (Nx-1) * Nv + Nvh
    v1 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x)
    shift_x = (Nx-1) * Nv + Nvh+1
    v2 = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x)

    v_start = v1.real
    dv = v2.real - v_start
    v_end = v_start + dv * (Nvh - 1)

    a0_FB0_R = coef_lcu * v_start
    a1_FB0_R = coef_lcu * (v_end - v_start) * coef_ratio_2

    # imaginary part:
    a0_FB0_iw_R = - coef_lcu * 2. * np.arcsin(v1.imag) 

    # corrections for the RIGHT diagonal:
    def find_corr_F_edges_RIGHT(id_v, a0_FB0, a1_FB0):
        shift_x = (Nx-1) * Nv + id_v
        vv = oo_ase.BF_prof_.get_matrix_element(shift_x, shift_x)
        a_corr_i = - 2. * np.arcsin(coef_lcu * vv.imag)
        a_inv = 2. * (a0_FB0 + (id_v - Nvh) * (2. * a1_FB0) / Nvh)
        a_corr_r = 2. * np.arccos(coef_lcu * vv.real)
        return a_inv, a_corr_r, a_corr_i
    aR_inv_vm1, aR_corr_vm1_r, aR_corr_vm1_i = find_corr_F_edges_RIGHT(Nv-1, a0_FB0_R, a1_FB0_R)
    aR_inv_vm3, aR_corr_vm3_r, aR_corr_vm3_i = find_corr_F_edges_RIGHT(Nv-3, a0_FB0_R, a1_FB0_R)
    aR_inv_vm4, aR_corr_vm4_r, aR_corr_vm4_i = find_corr_F_edges_RIGHT(Nv-4, a0_FB0_R, a1_FB0_R)


    print("\n//--- Parameters for the submatrix F-prof ---")
    print("\n//- diag 0: left edge -")
    print("a0_FB0_L \t{:0.12f}".format(a0_FB0_L))
    print("a1_FB0_L \t{:0.12f}".format(a1_FB0_L))
    print("a0_FB0_iw_L \t{:0.12f}".format(a0_FB0_iw_L))

    print("\n//- diag 0: left edge: corrections -")
    print("aL_inv_v0  \t\t{:0.12f}".format(aL_inv_v0))
    print("aL_corr_v0_r \t{:0.12f}".format(aL_corr_v0_r))
    print("aL_corr_v0_i \t{:0.12f}".format(aL_corr_v0_i))
    print()
    print("aL_inv_v2  \t\t{:0.12f}".format(aL_inv_v2))
    print("aL_corr_v2_r \t{:0.12f}".format(aL_corr_v2_r))
    print("aL_corr_v2_i \t{:0.12f}".format(aL_corr_v2_i))
    print()
    print("aL_inv_v3  \t\t{:0.12f}".format(aL_inv_v3))
    print("aL_corr_v3_r \t{:0.12f}".format(aL_corr_v3_r))
    print("aL_corr_v3_i \t{:0.12f}".format(aL_corr_v3_i))

    print("\n//- diag 0: right edge -")
    print("a0_FB0_R \t{:0.12f}".format(a0_FB0_R))
    print("a1_FB0_R \t{:0.12f}".format(a1_FB0_R))
    print("a0_FB0_iw_R \t{:0.12f}".format(a0_FB0_iw_R))

    print("\n//- diag 0: right edge: corrections -")
    print("aR_inv_vm1    \t{:0.12f}".format(aR_inv_vm1))
    print("aR_corr_vm1_r \t{:0.12f}".format(aR_corr_vm1_r))
    print("aR_corr_vm1_i \t{:0.12f}".format(aR_corr_vm1_i))
    print()
    print("aR_inv_vm3    \t{:0.12f}".format(aR_inv_vm3))
    print("aR_corr_vm3_r \t{:0.12f}".format(aR_corr_vm3_r))
    print("aR_corr_vm3_i \t{:0.12f}".format(aR_corr_vm3_i))
    print()
    print("aR_inv_vm4    \t{:0.12f}".format(aR_inv_vm4))
    print("aR_corr_vm4_r \t{:0.12f}".format(aR_corr_vm4_r))
    print("aR_corr_vm4_i \t{:0.12f}".format(aR_corr_vm4_i))

    print("\n//- diag -1 -")
    print("alpha_0_LFB1 \t{:0.12f}".format(alpha_0_LFB1))
    print("alpha_1_LFB1 \t{:0.12f}".format(alpha_1_LFB1))
    print()
    print("alpha_0_LFB1_corr1 \t{:0.12e}".format(alpha_0_LFB1_corr1))
    print("alpha_1_LFB1_corr1 \t{:0.12e}".format(alpha_1_LFB1_corr1))
    print("alpha_0_LFB1_corr1_2 \t{:0.12e}".format(alpha_0_LFB1_corr1_2))
    print("alpha_1_LFB1_corr1_2 \t{:0.12e}".format(alpha_1_LFB1_corr1_2))
    print()
    print("alpha_0_LFB1_corr2 \t{:0.12e}".format(alpha_0_LFB1_corr2))
    print("alpha_1_LFB1_corr2 \t{:0.12e}".format(alpha_1_LFB1_corr2))
    print("alpha_0_LFB1_corr2_2 \t{:0.12e}".format(alpha_0_LFB1_corr2_2))
    print("alpha_1_LFB1_corr2_2 \t{:0.12e}".format(alpha_1_LFB1_corr2_2))
    print()
    print("alpha_0_LFB1_corr3 \t{:0.12e}".format(alpha_0_LFB1_corr3))
    print("alpha_1_LFB1_corr3 \t{:0.12e}".format(alpha_1_LFB1_corr3))
    print()
    print("\n//- diag +1 -")
    print("alpha_0_RFB1 \t{:0.12f}".format(alpha_0_RFB1))
    print("alpha_1_RFB1 \t{:0.12f}".format(alpha_1_RFB1))
    print()
    print("alpha_0_RFB1_corr1 \t{:0.12e}".format(alpha_0_RFB1_corr1))
    print("alpha_1_RFB1_corr1 \t{:0.12e}".format(alpha_1_RFB1_corr1))
    print()
    print("alpha_0_RFB1_corr2 \t{:0.12e}".format(alpha_0_RFB1_corr2))
    print("alpha_1_RFB1_corr2 \t{:0.12e}".format(alpha_1_RFB1_corr2))
    print("alpha_0_RFB1_corr2_2 \t{:0.12e}".format(alpha_0_RFB1_corr2_2))
    print("alpha_1_RFB1_corr2_2 \t{:0.12e}".format(alpha_1_RFB1_corr2_2))
    print()
    print("alpha_0_RFB1_corr3 \t{:0.12e}".format(alpha_0_RFB1_corr3))
    print("alpha_1_RFB1_corr3 \t{:0.12e}".format(alpha_1_RFB1_corr3))
    print("alpha_0_RFB1_corr3_2 \t{:0.12e}".format(alpha_0_RFB1_corr3_2))
    print("alpha_1_RFB1_corr3_2 \t{:0.12e}".format(alpha_1_RFB1_corr3_2))
    print()
    print("alpha_0_RFB1_corr4 \t{:0.12e}".format(alpha_0_RFB1_corr4))
    print("alpha_1_RFB1_corr4 \t{:0.12e}".format(alpha_1_RFB1_corr4))
    print("alpha_0_RFB1_corr4_2 \t{:0.12e}".format(alpha_0_RFB1_corr4_2))
    print("alpha_1_RFB1_corr4_2 \t{:0.12e}".format(alpha_1_RFB1_corr4_2))
    print()
    print("\n//- diag +-2 -")
    print("alpha_0_FL2 \t{:0.12f}".format(alpha_0_FL2))
    print("alpha_1_FL2 \t{:0.12f}".format(alpha_1_FL2))
    print("alpha_0_FR2 \t{:0.12f}".format(alpha_0_FR2))
    print("alpha_1_FR2 \t{:0.12f}".format(alpha_1_FR2))
    print("pi2 \t{:0.12f}".format(2.*np.pi))
    return


# recheck the QuCF simulation of the submatrices:
def recheck_QuCF_submatrices(oo_ase):
    dd = qucf_r.read_matrix_sparse(oo_ase.path_qc_, oo_ase.output_qucf_) 

    print("\n--- Cf: QuCF version vs original version ---")
    Cf_recon = dd["A"].get_slice(oo_ase.Nvar_work_, 0, oo_ase.Nvar_work_,)
    compare_reconstructed(Cf_recon, oo_ase.Cf_orig_)

    print("\n--- CE: QuCF version vs original version ---")
    CE_recon = dd["A"].get_slice(0, oo_ase.Nvar_work_, oo_ase.Nvar_work_,)
    compare_reconstructed(CE_recon, oo_ase.CE_orig_)

    print("\n--- S: QuCF version vs original version ---")
    S_recon = dd["A"].get_slice(oo_ase.Nvar_work_, oo_ase.Nvar_work_, oo_ase.Nvar_work_)
    compare_reconstructed(S_recon, oo_ase.S_orig_)
    return


# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
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


# --- Returnes only the part of the matrix D related to the submatrix F ---
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


def normalize_matrix_A(A, D_F, nv):
    # --- original matrix ---
    print("original matrix >>>")
    A.print_max_min()

    A_norm = A.copy()

    # A_values = A.get_values()
    D_values = D_F.get_values()

    # --- normalization ---
    coef_d = 1./np.sqrt(2)

    coef_norm_D_1 = coef_d**(nv + 2)
    coef_norm_D_2 = np.min(np.min(np.abs(D_values[np.nonzero(D_values)])))
    coef_norm_D = np.min([coef_norm_D_1, coef_norm_D_2])

    # coef_norm_A = np.max(np.sqrt(np.sum(np.abs(A_values)**2)))
    # print("norm 1: ", np.max(np.sqrt(np.sum(np.abs(A_values)**2))))

    coef_norm_A = 0
    for ir in range(A.get_N()):
        temp = np.sqrt(np.sum(np.abs(A.get_values_in_a_row(ir))**2))
        if temp > coef_norm_A:
            coef_norm_A = temp
    print("norm of the matrix: ", coef_norm_A)

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


    # ----------------------------------------------------------------------------------------------
    # --- rescaling of the matrix elements taking into account the influence of the oracle OF, OM ---
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
    
    # --- diag points at the left and right spatial boundaries ---
    for ir in range(Nv//2):
        B_prel_profile[2, ir] = B.get_matrix_element(ir, ir)
        Nnz_profile += 1

    for ir in range(Nv*(Nx-1) + Nv//2, Nvar):
        B_prel_profile[2, ir] = B.get_matrix_element(ir, ir)
        Nnz_profile += 1

    # --- off-diag points at the left and right spatial boundaries ---
    for ir in range(Nv//2):
        B_prel_profile[3, ir] = B.get_matrix_element(ir, ir + Nv)
        B_prel_profile[4, ir] = B.get_matrix_element(ir, ir + 2*Nv)
        Nnz_profile += 2

    for ir in range(Nv*(Nx-1) + Nv//2, Nvar):
        B_prel_profile[1, ir] = B.get_matrix_element(ir, ir - Nv)
        B_prel_profile[0, ir] = B.get_matrix_element(ir, ir - 2*Nv)
        Nnz_profile += 2

    # --- off-diag spatial bulk points ---
    for ir in range(Nv, Nv*(Nx-1)):
        B_prel_profile[1, ir] = B.get_matrix_element(ir, ir - Nv)
        B_prel_profile[3, ir] = B.get_matrix_element(ir, ir + Nv)
        Nnz_profile += 2

    # *** Map the preliminary matrices to the sparse matrices ***
    if A_F.get_Nnz() != (Nnz_fixed + Nnz_profile):
        print("Error: incorrect splitting.")
        return
    B_sparse_fixed   = map_prel_to_sparse(Nnz_fixed,   B_prel_fixed,   1)
    B_sparse_profile = map_prel_to_sparse(Nnz_profile, B_prel_profile, Nv)

    return B_sparse_fixed, B_sparse_profile


def get_B_C_matrix(nv, C):
    norm_d = (1./np.sqrt(2.))**(nv+2)
    B_C = C.copy()
    B_values = B_C.get_values()
    B_values /= norm_d
    return B_C


def get_B_S_matrix(nx, nv, S):
    Nx, Nv = 1<<nx, 1<<nv
    B_S = S.copy()
    B_values = B_S.get_values()
    for ix in range(Nx):
        B_values[ix*Nv] /= 0.50
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
    # ax.text(49,  14, r'$\textbf{F}$', fontsize=fontsize)
    # ax.text(121, 14, r'$\textbf{C}^E$', fontsize=fontsize)
    # ax.text(49, 77, r'$\textbf{C}^f$', fontsize=fontsize)
    # ax.text(121, 77, r'$\textbf{S}$', fontsize=fontsize)

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
            print("Error: a path for saving a figure is not given.")
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



# def compare_reconstructed(Ar, Ao, prec = 1e-8):
def compare_reconstructed(Ar, Ao):
    Nr = len(Ar.get_values())
    No = len(Ao.get_values())

    if Nr == No:
        print("The same size of the matrices.")
        print("N of nonzero elements = {:d}".format(Nr))
    else:
        print("Different sizes.") 
        
    # flag_same = True
    max_abs_err = 0.0
    for ii in range(Nr):
        abs_err = np.abs(Ar.get_values()[ii] - Ao.get_values()[ii])
        if abs_err > max_abs_err:
            max_abs_err = abs_err

        # if not mix.compare_complex_values(
        #     Ar.get_values()[ii], 
        #     Ao.get_values()[ii], 
        #     prec
        # ):
        #     flag_same = False
    del abs_err

    print("Max. abs. error: {:0.3e}".format(max_abs_err))    
            
    # if flag_same:
    #     print("The values are the same within the precision {:0.1e}.".format(prec))
    # else:
    #     print("The absolute difference between values is larger than {:0.1e}.".format(prec))
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

