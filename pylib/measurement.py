from asyncio import constants
from pickletools import read_unicodestringnl
import sys
import datetime
import re
from typing import no_type_check
import numpy as np
import time
import h5py

import os

import pylib.mix as mix
import pylib.Global_variables as GLO

def reload():
    mix.reload_module(mix)
    mix.reload_module(GLO)


def get_str_complex(v):
    ll = ""
    vr,  vi  = v["real"],  v["imag"]
    avr, avi = np.abs(vr), np.abs(vi)
    coef_zero = 1e-16

    if   vi > 0 and avi > coef_zero and avr > coef_zero:
        ll = "{:0.3e}+{:0.3e}j".format(vr, vi)
    elif vi < 0 and avi > coef_zero and avr > coef_zero:
        ll = "{:0.3e}{:0.3e}j".format(vr, vi)
    elif avi < coef_zero and avr > coef_zero:
        ll = "{:0.3e}".format(vr)
    elif avr < coef_zero and avi > coef_zero:
        ll = "{:0.3e}j".format(vi)
    else:
        ll = "0.0"
    return ll


def get_str_state(q, format_q):
    nq = np.size(q)

    ll = ""
    count_q = 0
    count_fq = 0
    for i in range(nq):
        if count_q == 0:
            ll += "|"
            
        count_q += 1
        ll += "{:d}".format(q[i])
        if count_q == format_q[count_fq]:
            ll += ">"
            count_fq += 1
            count_q = 0
    return ll


def get_complex(ampls):
    N = len(ampls)
    ampls_complex = np.zeros(N, dtype=np.complex)
    for i_state in range(N):
        one_ampl = ampls[i_state]
        ampls_complex[i_state] = np.complex(one_ampl["real"], one_ampl["imag"])
    return ampls_complex


# -------------------------------------------------------------------------------
# --- Read output data from oracle ---
# -------------------------------------------------------------------------------
class MeasOracle__:
    # project name:
    pname_ = ""

    # path to the project:
    path_ = ""

    # information about the project:
    dd_ = None

    # constants:
    constants_ = None

    # states:  
    n_init_states_ = None
    init_states_ = None
    output_all_states_ = None
    output_zero_anc_states_ = None

    probs_ = None
    qubits_probs_ = None

    # states and amplitudes for the given input state;
    work_ampls_ = None
    work_states_ = None  

    __flag_print_ = None


    def __init__(self, flag_print=True):
        self.dd_ = {}
        self.__flag_print_ = flag_print
        return 
        

    def open(self):
        fname = self.path_ + "/" + self.pname_ + "_OUTPUT.hdf5"
        self.dd_["fname"] = fname

        if self.__flag_print_:
            print(f"Reading the file {fname}...")
        with h5py.File(fname, "r") as f:

            # --- basic data ---
            bg = f["basic"]

            self.dd_["date-of-sim"]  = bg["date-of-simulation"][()].decode("utf-8")
            self.dd_["project-name"] = bg["project-name"][()].decode("utf-8")
            self.dd_["launch-path"]  = bg["launch-path"][()].decode("utf-8")
            self.dd_["path-inputs"]  = bg["path-inputs"][()].decode("utf-8")

            self.dd_["nq"] = bg["nq"][()]
            self.dd_["na"] = bg["na"][()]
            if "N_gates" in bg:
                self.dd_["N-gates"] = bg["N_gates"][()]

            reg_names = bg["register-names"][()].decode("utf-8").split(", ")
            reg_nq = bg["register-nq"][...]

            self.dd_["reg-names"] = reg_names
            self.dd_["reg-nq"] = reg_nq

            self.dd_["regs"] = {}
            self.dd_["reg-shifts"] = {}
            reg_shift = 0
            for i in range(len(reg_nq)):
                self.dd_["regs"][reg_names[i]] = reg_nq[i]
                self.dd_["reg-shifts"][reg_names[i]] = reg_shift
                reg_shift += reg_nq[i]

            # --- constants ---
            self.constants_ = {}
            for field in f["constants"]:
                self.constants_[field] = f["constants"][field][()]

            # --- initial and output states ---
            st = f["states"]
            self.n_init_states_ = st["n-init-states"][()]
            self.init_states_            = [None] * self.n_init_states_
            self.output_all_states_      = [None] * self.n_init_states_
            self.output_zero_anc_states_ = [None] * self.n_init_states_
            for ii in range(self.n_init_states_):
                self.init_states_[ii] = {}
                self.init_states_[ii]["state"] = np.transpose(np.array(st["initial-states-{:d}".format(ii)])) 
                self.init_states_[ii]["ampls"] = np.array(st["initial-amplitudes-{:d}".format(ii)])

                line_state = "output-all-states-{:d}".format(ii)
                if line_state in st:
                    self.output_all_states_[ii] = {}
                    self.output_all_states_[ii]["state"] = np.transpose(np.array(st["output-all-states-{:d}".format(ii)])) 
                    self.output_all_states_[ii]["ampls"] = np.array(st["output-all-amplitudes-{:d}".format(ii)])

                line_state = "output-zero-anc-states-{:d}".format(ii)
                if line_state in st:
                    self.output_zero_anc_states_[ii] = {}
                    self.output_zero_anc_states_[ii]["state"] = np.transpose(np.array(st[line_state])) 
                    self.output_zero_anc_states_[ii]["ampls"] = np.array(st["output-zero-anc-amplitudes-{:d}".format(ii)])

            # --- probabilities ---
            line_group = "probabilities"
            if line_group in f:
                pr = f[line_group]
                self.probs_        = np.array(pr["probs"])
                self.qubits_probs_ = np.array(pr["qubits"])

        if self.__flag_print_:
            print("Name of the simulation is", self.dd_["project-name"])
            print("Simulation has been performed ", self.dd_["date-of-sim"])
        return


    def set_zero_ancillae_work_states(self, id_input_state = 0):
        self.work_states_ = self.output_zero_anc_states_[id_input_state]["state"]
        self.work_ampls_  = self.output_zero_anc_states_[id_input_state]["ampls"]
        return

    
    def set_work_states(self, id_input_state = 0):
        self.work_states_ = self.output_all_states_[id_input_state]["state"]
        self.work_ampls_  = self.output_all_states_[id_input_state]["ampls"]
        return
    

    # The chosen initial state becomes the work state
    def set_init_work_states(self, id_input_state = 0):
        self.work_states_ = self.init_states_[id_input_state]["state"]
        self.work_ampls_  = self.init_states_[id_input_state]["ampls"]
        return  


    def print_full_states(self):
        print("Number of initial states: {:d}".format(self.n_init_states_))

        print("\nRegisters: ")
        print(self.dd_["regs"])
        print()

        for ii in range(self.n_init_states_):
            print("\n-------------------------------------")
            print("--- Initial state: {:d}".format(ii))
            state = self.init_states_[ii]["state"]
            ampls = self.init_states_[ii]["ampls"]
            nr, _ = state.shape
            for ir in range(nr):
                str_ampl = get_str_complex(ampls[ir])
                str_state = get_str_state(state[ir], self.dd_["reg-nq"])
                print("{:>22s}   {:s}".format(str_ampl, str_state))

            print("\n -- full output state --")
            state = self.output_all_states_[ii]["state"]
            ampls = self.output_all_states_[ii]["ampls"]
            nr, _ = state.shape
            for ir in range(nr):
                str_ampl = get_str_complex(ampls[ir])
                str_state = get_str_state(state[ir], self.dd_["reg-nq"])
                print("{:>22s}   {:s}".format(str_ampl, str_state))
        return


    def print_zero_anc_states(self):
        print("Number of initial states: {:d}".format(self.n_init_states_))

        print("\nRegisters: ")
        print(self.dd_["regs"])
        print()

        for ii in range(self.n_init_states_):
            print("\n-------------------------------------")
            print("--- Initial state: {:d}".format(ii))
            state = self.init_states_[ii]["state"]
            ampls = self.init_states_[ii]["ampls"]
            nr, _ = state.shape
            for ir in range(nr):
                str_ampl = get_str_complex(ampls[ir])
                str_state = get_str_state(state[ir], self.dd_["reg-nq"])
                print("{:>22s}   {:s}".format(str_ampl, str_state))

            print("\n -- zero-ancilla output state --")
            if(self.output_zero_anc_states_[ii]):
                state = self.output_zero_anc_states_[ii]["state"]
                ampls = self.output_zero_anc_states_[ii]["ampls"]
                nr, _ = state.shape
                for ir in range(nr):
                    str_ampl = get_str_complex(ampls[ir])
                    str_state = get_str_state(state[ir], self.dd_["reg-nq"])
                    print("{:>22s}   {:s}".format(str_ampl, str_state))
        return


    def read_qsvt(self, flag_print=False):
        with h5py.File(self.dd_["fname"], "r") as f:
            gr_qsvt = f["qsvt"]

            temp_str = "n-qsvt-circuits"
            if temp_str not in gr_qsvt:
                n_qsvt = 1
            else:
                n_qsvt = int(gr_qsvt[temp_str][()])

            self.dd_["qsvt-names"] = [None]*n_qsvt
            for ii in range(n_qsvt):
                temp_str = "name-{:d}".format(ii)
                if temp_str not in gr_qsvt:
                    name_qsvt = "qsvt"
                else:
                    name_qsvt = gr_qsvt[temp_str][()].decode("utf-8")
                self.dd_["qsvt-names"][ii] = name_qsvt

            for ii in range(n_qsvt):
                name_qsvt = self.dd_["qsvt-names"][ii]
                gr_one = f[name_qsvt]

                self.dd_[name_qsvt] = {}
                self.dd_[name_qsvt]["type"] = gr_one["type"][()].decode("utf-8")
                self.dd_[name_qsvt]["eps"] = gr_one["eps"][()]
                self.dd_[name_qsvt]["par"] = gr_one["function-parameter"][()]
                self.dd_[name_qsvt]["rescaling_factor"] = gr_one["rescaling_factor"][()]
                if self.dd_[name_qsvt]["type"] == "QSP-ham":
                    self.read_qsp_hamiltonian_parameters(name_qsvt, gr_one, flag_print) 
        return


    def read_qsp_hamiltonian_parameters(self, name_qsp, gr, flag_print):
        self.dd_[name_qsp]["dt"] = gr["dt"][()]
        self.dd_[name_qsp]["nt"] = gr["nt"][()]

        if self.__flag_print_ and flag_print:
            print("\n--- QSP: {:s} ---".format(name_qsp))
            print("dt: {:0.3f}".format(self.dd_[name_qsp]["dt"]))
            print("nt: {:0.3f}".format(self.dd_[name_qsp]["nt"]))
        return
    

    def get_x_grid(self, reg_x):
        reg_nq = self.dd_["regs"][reg_x]
        N = 2**reg_nq
        x_grid = np.array(range(N))/(1.*(N-1))
        return x_grid


    ## --- The MAIN function to extract data from output states ---
    # Return the variable whose amplitudes as a function of x are entangled with "vars_enc".
    # The space dependence on x is encoded in the register "reg_x".
    # "vars_enc" is {"reg_name_1": int_to_choose, ...};
    # "reg_x" is the register name  where different combinations of qubits
    #           correspond to different points on x.
    # All OTHER registers are set to ZERO.
    # -----------------------
    # Prior to launch this function, one needs to call either set_zero_ancillae_work_states()
    #   or set_work_states()
    # -----------------------
    # FOR INSTANCE, assume that there are two registers: rd and rx,
    #   where rd = 0 corresponds to the variable V1 and rd = 1 -> V2, and 
    #   where rx has nx qubits and encodes the space dependence of the variables V1 and V2 on x.
    #   Then, to extract V2(x), launch 
    #       get_var_x({"rd": 0}, "rx")
    # -----------------------
    # If "vars_enc" = {}, take all states.
    def get_var_x(self, vars_enc, reg_x):
        nx = self.dd_["regs"][reg_x]
        Nx = 2**nx
        ampls = np.zeros(Nx, dtype=np.complex)

        # prepare a dictionary that defines a set of states to be considered:
        var_to_cons = {}
        if bool(vars_enc):
            for reg_name in self.dd_["reg-names"]:
                if reg_name in vars_enc.keys():
                    var_to_cons[reg_name] = vars_enc[reg_name]
                    continue
                if reg_name != reg_x:
                    var_to_cons[reg_name] = 0
                    continue

        # consider only states for the chosen variable encoded by vars_enc:
        ampls_to_search, states_to_search = self.get_several_chosen_work_states(var_to_cons)

        # every state in the considered set of states must correspond to one space point:
        nstates = len(states_to_search)
        for i_state in range(nstates):
            int_x = self.convert_reg_state_to_int(reg_x, states_to_search[i_state])
            one_ampl     = ampls_to_search[i_state]
            ampls[int_x] = np.complex(one_ampl["real"], one_ampl["imag"])
        return ampls


    ## Return states and their amplitudes defined by the dictionary "choice":
    # choice = {"reg_name_1": int_to_choose, "reg_name_2": int_to_choose, ...}.
    # The dictionary indicates which state the register must have.
    # If the register is not defined in the dictionary, then 
    # the function returns all available states from this register.
    # If "choice" = {}, taje all output states for the chosen input state.
    def get_several_chosen_work_states(self, choice):
        one_step_states = self.work_states_
        one_step_ampls  = self.work_ampls_

        if bool(choice):
            ch_state = self.create_mask(choice, -1)
            nstates, _ = one_step_states.shape
            res_ampls = []
            res_states = []
            for i_state in range(nstates):
                flag_choose = True
                one_state = one_step_states[i_state]
                for i_qubit in range(self.dd_["nq"]):
                    if ch_state[i_qubit] == -1:
                        continue
                    if ch_state[i_qubit] != one_state[i_qubit]:
                        flag_choose = False
                        break
                if flag_choose:
                    res_ampls.append(one_step_ampls[i_state])
                    res_states.append(one_state)
            return res_ampls, res_states
        else:
            return one_step_ampls, one_step_states


    ## Create an 1D-array of size 2**nq (nq - number of qubits in the circuit),
    # where corresponding pieces of the array are filled by binary representations of
    # the register states from the "choice" map.
    # The rest of the array elements are filled with the "def_value". 
    def create_mask(self, choice, def_value=0):
        nq  = self.dd_["nq"]
        ch_state = [def_value] * nq
        for reg_name, reg_int in choice.items():
            bit_array = mix.find_bit_array_of_int(reg_int, self.dd_["regs"][reg_name])
            for i_bit in range(len(bit_array)):
                i_bit_pos = self.dd_["reg-shifts"][reg_name] + i_bit
                ch_state[i_bit_pos] = bit_array[i_bit]
        return ch_state


    ## The "input_state" is 1-D array of size 2**nq (nq - number of qubits in the circuit);
    # The register name "reg_name" chooses a piece of the array "input_state".
    # This piece contains a bit-array.
    # The function returns the integer representation of the bit-array.
    # The zero-th bit in the register is assumed to be the most significant.
    def convert_reg_state_to_int(self, reg_name, input_state):
        n_reg = self.dd_["regs"][reg_name]
        shift_reg = self.dd_["reg-shifts"][reg_name]
        int_repr = 0
        for iq in range(n_reg):
            int_repr += 2**(n_reg-iq-1) * input_state[shift_reg + iq]
        return int_repr
    

    # id_init_state is an ID of an initial state that you would like to consider.
    def read_qsp_ham_results(self, name_qsp, id_init_state):
        Nt = self.dd_[name_qsp]["nt"]
        ndata = self.dd_["nq"] - self.dd_["na"] # n of nonancilla qubits;
        N = 1 << ndata # size of the vector where information is stored
        print("State-vector size: {:d}".format(N))

        # N_recheck = np.prod(np.array(Ns_split))
        # if N != N_recheck:
        #     print("\nERROR: wrong vector dimensionality (Ns-split): ")
        #     print(Ns_split)
        #     print("prod(Ns-split): {:d}".format(N_recheck))
        #     print("whilst state-vector size: {:d}".format(N))
        #     return
        
        res_vector = np.zeros([Nt+1, N], dtype=complex)
        with h5py.File(self.dd_["fname"], "r") as f:
            gr = f[name_qsp]
            for it in range(Nt+1):
                res_vector[it,:] = get_complex(np.array(
                    gr["ampls-{:d}-{:d}".format(it, id_init_state)]
                ))

        return res_vector
    

    def qsp_get_time_grid(self, name_qsp, norm_of_H):
        Nt = self.dd_[name_qsp]["nt"] + 1
        dt = self.dd_[name_qsp]["dt"]
        t = np.zeros(Nt)

        for it in range(Nt):
            t[it] = dt * it
        t = t/norm_of_H
        return t
    

    # The fields in the resulting dictionary dd are:
    # dd["ampls"][i-iter][i-ampl] is the state amplidute
    #   where i-iter = [0, dd["N-mult"])
    #   and the range i-ampl can change for different i-iter.
    # dd["states"][i-iter][i-ampl] is the bitstring.
    def read_gadget(self, name_gadget, id_init_state):
        dd = {}
        with h5py.File(self.dd_["fname"], "r") as f:
            gr = f[name_gadget]

            dd["N-mult"] = gr["N_mult"][()]
            dd["counter-qubits"] = np.transpose(np.array(gr["counter-qubits"]))

            # read amplitudes:
            ampls = [None] * dd["N-mult"]
            for i_iter in range(dd["N-mult"]):
                ampls[i_iter] = get_complex(np.array(
                    gr["ampls-{:d}-{:d}".format(i_iter, id_init_state)]
                ))
            dd["ampls"] = ampls

            # read states:
            states = [None] * dd["N-mult"]
            for i_iter in range(dd["N-mult"]):
                states[i_iter] = np.transpose(np.array(
                    gr["states-{:d}-{:d}".format(i_iter, id_init_state)]
                )) 
            dd["states"] = states
        return dd
    

    # OUTPUT:
    # > reg_inp_states[regname][id-input-state][id-state-in-INPUT-superposition] = integer,
    #   where integer represents the bitstring stored in the register regname.
    # > reg_out_states[regname][id-input-state][id-state-in-OUPUT-superposition] = integer,
    def form_reg_input_output_states(self):
        # ---
        def state_to_reg_states(reg_states, i_state, state_superposition):
            N_loc_states = len(state_superposition)
            for i_reg in range(N_regs):
                reg_states[reg_names[i_reg]][i_state] = np.zeros(N_loc_states, dtype=int)

            for counter_state in range(N_loc_states):
                one_state = state_superposition[counter_state]
                for i_reg in range(N_regs):
                    bitstring = one_state[reg_shifts[i_reg]:reg_shifts[i_reg + 1]]
                    reg_states[reg_names[i_reg]][i_state][counter_state] = \
                            mix.find_int_from_bit_array(bitstring)
            return reg_states

        # ----
        reg_names = self.dd_["reg-names"] # the names start from the most significant register;
        N_regs = len(reg_names)
        N_states = len(self.init_states_)
        
        # initialization
        reg_inp_states = {}
        reg_out_states = {}
        reg_shifts = [None] * (N_regs+1)
        for i_reg in range(N_regs):
            reg_inp_states[reg_names[i_reg]] = [None] * N_states
            reg_out_states[reg_names[i_reg]] = [None] * N_states
            reg_shifts[i_reg] = self.dd_["reg-shifts"][reg_names[i_reg]]
        reg_shifts[-1] = self.dd_["nq"]

        # in each input superposition of states...
        for i_state in range(N_states):
            inp_state_superposition = self.init_states_[i_state]["state"]
            out_state_superposition = self.output_all_states_[i_state]["state"]

            # ...might be several states
            reg_inp_states = state_to_reg_states(reg_inp_states, i_state, inp_state_superposition)
            reg_out_states = state_to_reg_states(reg_out_states, i_state, out_state_superposition)
        return reg_inp_states, reg_out_states
