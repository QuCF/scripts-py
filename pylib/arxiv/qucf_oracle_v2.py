import numpy as np
import importlib as imp
import h5py
import matplotlib.pyplot as plt
import sys
from numba import jit
import cmath
import pylib.mix as mix

   
def reload():
    mix.reload()
    return


# -------------------------------------------------------------------------------------------------
# --- Register class ---
# -------------------------------------------------------------------------------------------------
class Reg__:
    name_ = ""
    nq_ = 0
    N_ = 0

    # False: bitstring of the register stores absolute matrix indices;
    # True: bistring of the register stores relative positions with respect to matrix diagonal;
    flag_rel_ = False
    max_sh_ = None  # maximum shift from the diagonal;

    rel_values_ = None
    shifts_ = None

    # rel_decode_int_[integer-encoded-into-register] = 
    #       a position in rel_values_ 
    #           = 
    #       an index of a local section;
    rel_decode_int_ = None

    # the maximum number of different values that can be encoded to the register taking
    # taken into account the type of the register (absolute or relative)
    Nval_ = None 

    # if for any integer N, (N - Nval_) <= N_edge_, than N is assumed to be comparable to Nval_:
    # (set by hand)
    N_edge_ = None  

    def __init__(self, name, nq, flag_rel = False):
        self.name_ = name
        self.nq_ = nq
        self.N_ = 1 << self.nq_

        self.flag_rel_ = flag_rel
        if self.flag_rel_:
            self.max_sh_ = (1 << (self.nq_-1)) - 1
            self.Nval_ = 2 * self.max_sh_ + 1
            self.N_edge_ = 1

            self.rel_values_ = np.zeros(2*self.max_sh_ + 1, dtype=int)
            self.shifts_ = np.zeros(2*self.max_sh_ + 1, dtype=int)
            counter = -1
            Nm = 1 << (self.nq_-1)
            for ii in range(-self.max_sh_, self.max_sh_+1):
                counter += 1
                self.rel_values_[counter] = ii if ii >= 0 else (Nm - ii)
                self.shifts_[counter] = ii

            self.rel_decode_int_ = np.zeros(self.N_, dtype=int)
            for int_enc in range(self.N_):
                if int_enc <= self.max_sh_: # a zero and positive shifts
                    self.rel_decode_int_[int_enc] = self.max_sh_ + int_enc
                else: # another zero and negative shifts
                    self.rel_decode_int_[int_enc] = 2 * self.max_sh_ - int_enc + 1
        else:
            self.Nval_ = self.N_
            self.N_edge_ = 2*4
        return 


    def encode_column_index_into_register(self, ir, ic):
        integer = np.nan

        # for a relative register:
        # diag    <-> 0..000
        # diag +1 <-> 0..001
        # diag -1 <-> 1..001
        # diag +2 <-> 0..010
        # diag -2 <-> 1..010
        if self.flag_rel_:
            counter = -1
            for ii in range(-self.max_sh_, self.max_sh_+1):
                counter += 1
                if(ic == (ir + ii)):
                    integer = self.rel_values_[counter]
                    break
        # for an absolute register:
        else: 
            integer = ic
        return integer




# -------------------------------------------------------------------------------------------------
# --- Registers class ---
# -------------------------------------------------------------------------------------------------
class Regs__: # group of registers of the same type (input or ancilla)
    names_ = None
    regs_ = None
    nqs_ = None  # number of qubits in each register starting from the most significant register
    nq_ = 0   # the total number of qubits in the group of registers
    N_regs_ = None

    def __init__(self):
        self.names_ = []
        self.regs_ = {}
        self.nqs_ = []
        return
    
    def add_reg(self, name, nq, flag_rel = False):
        print("Adding a register '{:s}'...".format(name))
        if name not in self.names_:
            self.names_.append(name)
            self.regs_[name] = Reg__(name, nq, flag_rel)
            self.nqs_.append(nq)
            self.nq_ += nq
            self.N_edge_ = 1
            return
        else:
            print("Error: the register is already in the group")
            return
        
    def get_reg(self, ii):
        name_reg = self.names_[ii]
        return self.regs_[name_reg]
    
    def compute_N_registers(self):
        self.N_regs_ = len(self.names_)
        return self.N_regs_
    



# -------------------------------------------------------------------------------------------------
# --- Circuit class ---
# -------------------------------------------------------------------------------------------------
class Circuit__:
    # Assume that ancilla registers are more significant than input registers;
    # First register in a set of registers is the most significant;
    # First qubit in a register is the most significant;

    ancilla_regs_ = None 
    input_regs_ = None
    nq_ = 0

    # each section corresponds to 
    # a column (for an absolute ancilla register) or
    # a diagonal (for a relative ancilla register):
    N_sections_ = None

    def __init__(self):
        return
    
    def set_regs(self, input_reg, ancilla_reg):
        self.ancilla_regs_ = ancilla_reg
        self.input_regs_   = input_reg
        self.nq_ = self.ancilla_regs_.nq_ + self.input_regs_.nq_

    def compute_N_registers(self):
        self.input_regs_.compute_N_registers()
        self.ancilla_regs_.compute_N_registers()
        return

    def compute_N_sections(self):
        self.N_sections_ = 1

        N_anc = self.ancilla_regs_.N_regs_
        for i_anc in range(N_anc):
            anc = self.ancilla_regs_.get_reg(i_anc)
            self.N_sections_ *= anc.Nval_

    def print_structure(self):
        def print_reg(name, N_regs, nqs):
            line_reg = "N-{:s}-registers = {:d} with [".format(name, N_regs)
            for i_reg in range(N_regs):
                if i_reg == N_regs - 1:
                    line_reg += "{:d}] ".format(nqs[i_reg])
                else:
                    line_reg += "{:d}, ".format(nqs[i_reg])
            line_reg += "qubits"
            print(line_reg)

        print_reg("input",   self.input_regs_.N_regs_,   self.input_regs_.nqs_)
        print_reg("ancilla", self.ancilla_regs_.N_regs_, self.ancilla_regs_.nqs_)

        nq_matrix = np.sum(self.input_regs_.nqs_, dtype=int)
        N_matrix = 1 << nq_matrix
        print("Total number of input qubits: {:d}".format(nq_matrix))
        print("Matrix size that can be encoded: {:d}".format(N_matrix))
        return


    # grid of the-same-type-gates (e.g. of Rx);
    # grid is divided in sections whose number is proportional to sparsity;
    # each section is of size Nr, where Nr is the number of row of the encoded matrix;
    def init_grid_of_gates(self):
        self.compute_N_sections()

        grid_of_gates = [None] * self.N_sections_

        Nr = 1<<self.input_regs_.nq_
        for i_section in range(self.N_sections_):
            grid_of_gates[i_section] = np.zeros(Nr); 
            grid_of_gates[i_section].fill(np.nan)
        return grid_of_gates
    

    # integers_anc_column = [int_in_most_significant_areg ... int_in_less_significant_areg]: 
    #   integers encoded in ancilla registers;
    # given integers are encoded in ancilla registers (one integer in each register),
    # find a section corresponding to these encoded integers;
    # hint: the least-significant ancilla register encodes first several sections:
    def get_section_index(self, integers_anc_column, i_anc=0, i_section=0):
        anc = self.ancilla_regs_.get_reg(i_anc)
        i_loc_section = anc.rel_decode_int_[integers_anc_column[i_anc]]
        i_section = i_section * anc.Nval_ + i_loc_section

        i_anc += 1
        if(i_anc < self.ancilla_regs_.N_regs_):
            i_section = self.get_section_index(integers_anc_column, i_anc, i_section)
        return i_section


    # find a list of integers (one integer for each ancilla register) that
    # encodes the given section index section_id;
    # anc_integers[0]: the resulting integer in the most-significant ancilla register.
    # anc_bitstrings[i] is a bitstring encoding the integer anc_integers[i].
    def get_anc_integers_from_section_index(self, section_id, i_anc = 0, anc_integers=None):
        if i_anc == 0:
            anc_integers=[]

        if i_anc == (self.ancilla_regs_.N_regs_-1): # the least significant register:
            loc_section_id = section_id
        else:
            Nv_next_anc = 1
            for ii_next_anc in range(i_anc+1, self.ancilla_regs_.N_regs_):
                Nv_next_anc *= self.ancilla_regs_.get_reg(ii_next_anc).Nval_
            loc_section_id = int(section_id / Nv_next_anc)
            section_id = section_id - Nv_next_anc * loc_section_id

        anc = self.ancilla_regs_.get_reg(i_anc)
        integer_anc = anc.rel_values_[loc_section_id]

        anc_integers.append(integer_anc)
        i_anc += 1
        if i_anc < self.ancilla_regs_.N_regs_:
            anc_integers = self.get_anc_integers_from_section_index(section_id, i_anc, anc_integers)
        return anc_integers
    

    # decompose i_to_decompose into a bistring;
    # then, split the bistring between input registers;
    # a bistring in each input register corresponds to an integer -> integers_row
    def compute_integers_in_input_registers(self, i_to_decompose):
        N_input_regs = self.input_regs_.N_regs_
        bitstring = mix.find_bit_array_of_int(i_to_decompose, self.input_regs_.nq_)

        # get an integer for each register:
        integers_row = [None] * N_input_regs
        id_start = 0
        for ireg in range(N_input_regs):
            nq_reg = self.input_regs_.nqs_[ireg]
            bitstring_reg = bitstring[id_start:(id_start + nq_reg)]
            integers_row[ireg] = mix.find_int_from_bit_array(bitstring_reg)
            id_start += nq_reg
        return integers_row


    # assume that each ancilla register corresponds to an input register:
    def compute_integers_in_ancilla_registers(self, irs, ics):
        N_anc_regs = self.ancilla_regs_.N_regs_
        integers   = [None] * N_anc_regs
        for i_anc_reg in range(N_anc_regs):
            integers[i_anc_reg] = \
                self.ancilla_regs_.get_reg(i_anc_reg).encode_column_index_into_register(
                    irs[i_anc_reg], 
                    ics[i_anc_reg]
                )
        return integers
    

    # Here, we assume that for each input register there is a corresponding ancilla register:
    def get_column_index_from_anc_integers(
            self, integers_anc_regs, integers_input_regs, column_id = 0, i_anc = 0
        ):
        anc = self.ancilla_regs_.get_reg(i_anc)
        if anc.flag_rel_: # a relative register;
            shift_loc = anc.shifts_[anc.rel_decode_int_[integers_anc_regs[i_anc]]]
            id_loc_column = integers_input_regs[i_anc] + shift_loc
        else: # an absolute register;
            id_loc_column = integers_anc_regs[i_anc]

        column_id = column_id * self.input_regs_.get_reg(i_anc).N_ + id_loc_column

        i_anc += 1
        if i_anc < self.ancilla_regs_.N_regs_:
            column_id = self.get_column_index_from_anc_integers(
                integers_anc_regs, integers_input_regs, column_id, i_anc
            )
        return column_id
    
    def compute_row_from_input_reg_integers(self, integers_input_regs):
        id_row = 0
        N_regs = self.input_regs_.N_regs_
        for i_reg in range(N_regs):
            nq = self.input_regs_.nqs_[i_reg]
            N = 1 << nq
            id_row = id_row * N + integers_input_regs[i_reg]
        return id_row
    

    def write_line_for_control_integers(self, reg_integers, flag_input=True):
        N_regs = self.input_regs_.N_regs_ if flag_input else self.ancilla_regs_.N_regs_

        res_line = "control_e {:d} ".format(N_regs)
        for i_reg in range(N_regs):
            res_line += "{:s} {:d} ".format(
                self.input_regs_.names_[i_reg] if flag_input else self.ancilla_regs_.names_[i_reg],
                reg_integers[i_reg]
            )
        return res_line
    

# -------------------------------------------------------------------------------------------------
# --- Group ---
# A group consists of several zones on a given matrix column or diagonal, 
# where matrix elements have the same value. 
# -------------------------------------------------------------------------------------------------
class Group__:
    # group angle:
    a_ = None 

    # row indices where various zones of the group start within a matrix;
    # the indices are sorted in the ascending order:
    irs_start_ = None 

    # row indices where various zones of the group end (without including) within a matrix;
    # the indices are sorted in the ascending order:
    irs_end_ = None

    # array of rows: rows from all zones presented in irs_start_ and irs_end_
    irs_ = None 

    # if True, use control_reg_bits_ to set control nodes, and 
    #   use irs_complement_ to set control nodes
    #   to inverse the group action for the rows from irs_complement_;
    # if False, use irs_ to set control nodes;
    flag_extended_ = None

    # -> if control_reg_bits_[k,i] = 0, set 0-control node at the i-th qubit of the k-th register;
    # -> if control_reg_bits_[k,i] = 1, set 1-control node at the iq-th qubit of the k-th register;
    # -> if control_reg_bits_[k,i] = 2, there is not any control on the iq-th qubit of the k-th register;
    control_reg_bits_ = None

    # reg_integers_ = None
    irs_complement_ = None


    def __init__(self, obj_group=None, angle=None):
        if obj_group is not None:
            self.set_a(obj_group.a_)
            self.irs_start_ = list(obj_group.irs_start_)
            self.irs_end_   = list(obj_group.irs_end_)
            # here, only all a part of information is copied from the given object.
        if angle is not None:
            self.set_a(angle)
            self.irs_start_ = []
            self.irs_end_ = [] 
        return


    def set_a(self, a_new):
        if type(a_new) == list:
            self.a_ = list(a_new)
        else:
            self.a_ = a_new

        if a_new is None:
            print("Angle is None.")
        return


    def construct_complement(self, circ):
        if not self.flag_extended_:
            return
        nq = np.sum(np.array(circ.input_regs_.nqs_), dtype=int)
        N_full = 1 << nq

        # area covered by control nodes:
        mask_control = np.zeros(N_full, dtype=int)
        mask_control = self.form_control_mask(circ, mask_control)

        # original area:
        mask_orig = np.zeros(N_full, dtype=int)
        N_orig = len(self.irs_)
        for ii in range(N_orig):
            mask_orig[self.irs_[ii]] = 1

        # complement to the original area whithin the control area:
        mask_complement = mask_control - mask_orig
        self.irs_complement_ = np.where(mask_complement == 1)[0]

    # get control bits in the i_reg-th register:
    # control_bits[i]:
    # = 0: zero-control node on the i-th qubit;
    # = 1: unit-control node on the i-th qubit;
    # = 2: no control nodes;
    # nqf - number of control nodes (0 or 1).
    def get_control(self, circ, i_reg):
        control_bits = self.control_reg_bits_[i_reg]
        nq = circ.input_regs_.nqs_[i_reg]
        nqf = np.where(control_bits == 2)[0]
        nqf = nq if len(nqf) == 0 else nqf[0]
        return control_bits, nq, nqf

    
    def form_control_mask(self, circ, mask_control, i_reg = 0, sh = 0):
        control_bits, nq, nqf = self.get_control(circ, i_reg)

        int_start = 0
        for i_bit in range(nqf):
            int_start += control_bits[i_bit] * (1 << (nq - 1 - i_bit))

        N_interval = 1 << (nq - nqf)
        int_end = int_start + N_interval

        N = 1 << nq
        sh_init = sh * N
        if i_reg == (circ.input_regs_.N_regs_ - 1): # least significant qubit
            for i_int in range(int_start, int_end):
                i_res = sh_init + i_int
                mask_control[i_res] = 1
        else:
            for i_int in range(int_start, int_end):
                sh = sh_init + i_int
                mask_control = self.form_control_mask(circ, mask_control, i_reg + 1, sh)
        return mask_control
    
    # find all rows that are covered by the control nodes of the extended group;
    # each row is represented as a set of integers where 
    # each integer is encoded into an input register;
    def compute_reg_integers_from_control_nodes(self, circ):
        N_regs = circ.input_regs_.N_regs_
        n_floating_bits = 0
        for i_reg in range(N_regs):
            _, nq, nqf_loc = self.get_control(circ, i_reg)
            n_floating_bits += (nq - nqf_loc)

        N_ints = 1 << n_floating_bits
        control_integers, _ = self.get_control_integers(circ, np.zeros(N_ints, dtype=int))

        array_integers_input_regs = np.zeros((N_ints, N_regs), dtype=int)
        for i_control_int in range(len(control_integers)):
            array_integers_input_regs[i_control_int,:] = \
                circ.compute_integers_in_input_registers(
                    control_integers[i_control_int]
                )
        return array_integers_input_regs
    
    
    def get_control_integers(self, circ, control_integers, i_reg = 0, sh = 0, counter_int=-1):
        control_bits, nq, nqf = self.get_control(circ, i_reg)

        int_start = 0
        for i_bit in range(nqf):
            int_start += control_bits[i_bit] * (1 << (nq - 1 - i_bit))

        N_interval = 1 << (nq - nqf)
        int_end = int_start + N_interval

        N = 1 << nq
        sh_init = sh * N
        if i_reg == (circ.input_regs_.N_regs_ - 1): # least significant qubit
            for i_int in range(int_start, int_end):
                i_res = sh_init + i_int
                counter_int += 1
                control_integers[counter_int] = i_res
        else:
            for i_int in range(int_start, int_end):
                sh = sh_init + i_int
                control_integers, counter_int = \
                    self.get_control_integers(
                        circ, control_integers, i_reg + 1, sh, counter_int
                    )
        return control_integers, counter_int
    

    def write_line_for_input_control_nodes(self, circ):
        def write_line(flags, controls, one_line, keyword):
            N_control_regs = len(np.where(flags)[0])
            if N_control_regs > 0:
                one_line = "{:s} {:d} ".format(keyword, N_control_regs)
                for i_reg in range(N_regs):
                    if flags[i_reg]:
                        one_line += "{:s}[".format(circ.input_regs_.names_[i_reg])
                        for iq_control in controls[i_reg]:
                            one_line += "{:d} ".format(iq_control)
                        one_line += "] "
            return one_line

        N_regs = circ.input_regs_.N_regs_
        flags_zero = [False] * N_regs
        flags_unit = [False] * N_regs
        zero_controls = [None] * N_regs
        unit_controls = [None] * N_regs
        for i_reg in range(N_regs):
            zero_controls[i_reg] = []
            unit_controls[i_reg] = []
            control_bits, _, nqf = self.get_control(circ, i_reg)
            for iq in range(nqf):
                if control_bits[iq] == 0:
                    flags_zero[i_reg] = True
                    zero_controls[i_reg].append(iq)
                if control_bits[iq] == 1:
                    flags_unit[i_reg] = True
                    unit_controls[i_reg].append(iq)
        res_line = write_line(flags_zero, zero_controls, "", "ocontrol")
        res_line = write_line(flags_unit, unit_controls, res_line, "control")
        return res_line


# -------------------------------------------------------------------------------------------------
# --- System of gates ---
# -------------------------------------------------------------------------------------------------
class SystemGates__:
    circ_ = None
    D_ = None # structure matrix
    gate_types_ = None 
    dict_gates_ = None

    ngrp_ext_ = 3
    ngrp_split_ = 1

    def __init__(self, circ, D, groups_Rx, groups_Ry, groups_Rz):
        self.circ_ = circ
        self.D_ = D

        self.gate_types_ = []
        self.dict_gates_ = {}

        gate_type = "Rx"
        self.gate_types_.append(gate_type)
        self.dict_gates_[gate_type] = groups_Rx

        gate_type = "Ry"
        self.gate_types_.append(gate_type)
        self.dict_gates_[gate_type] = groups_Ry

        gate_type = "Rz"
        self.gate_types_.append(gate_type)
        self.dict_gates_[gate_type] = groups_Rz
        return
    

    def reconstruct_matrix_using_groups(self):    
        N = 1 << self.circ_.input_regs_.nq_
        A_recon = np.zeros((N, N), dtype=complex) # reconstructed matrix
        print("Matrix size: {:d}", N)
        for gate_type in self.gate_types_:
            groups_R = self.dict_gates_[gate_type]

            # number of nonzero sections and their indices (int, ix, iv)
            counter_sections = 0
            for i_section in range(self.circ_.N_sections_):
                one_section = groups_R[i_section]
                if not one_section:
                    continue # empty section
                counter_sections += 1
                anc_integers = self.circ_.get_anc_integers_from_section_index(i_section)
                N_groups = len(one_section)
                for i_group in range(N_groups): # consider all groups whithin each section;
                    one_group = one_section[i_group]
                    for i_zone in range(len(one_group.irs_start_)): # consider all zones within each group;
                        N_rows = one_group.irs_end_[i_zone] - one_group.irs_start_[i_zone]
                        for counter_row in range(N_rows): # consider all rows within each zone;
                            ir = one_group.irs_start_[i_zone] + counter_row

                            # get ic:
                            integers_input_regs = self.circ_.compute_integers_in_input_registers(ir)
                            ic = self.circ_.get_column_index_from_anc_integers(anc_integers, integers_input_regs)

                            # compute R:
                            A_recon[ir, ic] = self.apply_gate(A_recon[ir, ic], gate_type, self.D_[ir, ic], one_group.a_)
        return A_recon
    

    def apply_gate(self, element_value, gate_type, value_D, group_angle, flag_reverse=False):
        # we assume that R acts on the unit state, and 
        # the result is encoded into the amplitude of the zero state:
        if not flag_reverse:
            if gate_type == "Rx":
                element_value = - value_D * 1j * np.sin(group_angle/2.)
            if gate_type == "Ry":
                if np.abs(element_value) > 0:
                    element_value = - element_value * np.sin(group_angle/2.)
                else:
                    element_value = - value_D * np.sin(group_angle/2.)
            if gate_type == "Rz":
                if np.abs(element_value) > 0:
                    element_value = element_value * np.exp(1j*group_angle/2.)
                else:
                    element_value = value_D * np.exp(1j*group_angle/2.)
        else:
            if gate_type == "Rx":
                element_value = value_D * 1j * np.sin(group_angle/2.)
            if gate_type == "Ry":
                if np.abs(element_value) > 0:
                    element_value = element_value * np.sin(group_angle/2.)
                else:
                    element_value = value_D * np.sin(group_angle/2.)
            if gate_type == "Rz":
                if np.abs(element_value) > 0:
                    element_value = element_value * np.exp(1j*group_angle/2.)
                else:
                    element_value = value_D * np.exp(1j*group_angle/2.)
        return element_value
    

    def count_groups(self):
        print()
        print("N = {:d}".format(1 << self.circ_.input_regs_.nq_))
        for i_gate in range(len(self.gate_types_)):
            gate_type = self.gate_types_[i_gate]
            print("--- Groups for the gate {:s} ---".format(gate_type))
            groups_R = self.dict_gates_[gate_type]
            counter_groups = 0
            for i_section in range(self.circ_.N_sections_):
                one_section = groups_R[i_section]
                if one_section is None:
                    continue
                N_groups = len(one_section)
                counter_groups += N_groups
            print("Number of groups: {:d}".format(counter_groups))
        print()
        return
    

    def print_angles(self, gate_type):
        print()
        print("Group angles for the gate {:s}".format(gate_type))
        groups_R = self.dict_gates_[gate_type]
        for i_section in range(self.circ_.N_sections_):
            one_section = groups_R[i_section]
            line_angles = ""
            if one_section is None:
                continue
            N_groups = len(one_section)
            print("Section: {:d}".format(i_section))
            for i_group in range(N_groups):
                one_group = one_section[i_group]
                line_angles += "{:0.12e} ".format(one_group.a_)
            print(line_angles)
        return
    

    # Assume that gates act on a target qubit called ae;
    def construct_circuit_OH(self, path, filename):
        file_lines = []
        counter_gates = 0
        gate_types_loc = list(self.gate_types_)

        # --- Control nodes ---
        for i_gate in range(len(gate_types_loc)):
            gate_type = gate_types_loc[i_gate]
            groups_R = self.dict_gates_[gate_type]
            for i_section in range(self.circ_.N_sections_):
                one_section = groups_R[i_section]
                if one_section is None:
                    continue
                ancilla_integers = self.circ_.get_anc_integers_from_section_index(i_section)
                N_groups = len(one_section)
                for i_group in range(N_groups):
                    one_group = one_section[i_group]
                    if one_group.flag_extended_:
                        group_angle = one_group.a_
                        one_line = "gate {:s} ae 1 {:19.12e} ".format(gate_type, group_angle)
                        one_line += one_group.write_line_for_input_control_nodes(self.circ_)
                        one_line += self.circ_.write_line_for_control_integers(ancilla_integers, False)
                        one_line += "end_gate"
                        file_lines.append(one_line)
                        counter_gates += 1

        # --- Complements ---
        file_lines.append("\n")
        gate_types_loc.reverse()
        counter_gates = self.write_control_integers(file_lines, gate_types_loc, False, counter_gates)

        # --- Integers to set control nodes ---
        file_lines.append("\n")
        gate_types_loc.reverse()
        counter_gates = self.write_control_integers(file_lines, gate_types_loc, True, counter_gates)
        
        # --- build section CIRCUIT_STRUCTURE ---
        fullname = path + "/" + filename + ".oracle"
        ff = open(fullname, "w")
        for one_line in file_lines:
            ff.write(one_line + "\n")
        ff.close()

        # --- Log data ---
        self.circ_.print_structure()
        print("N-gates: {:d}".format(counter_gates))
        return
    

    def write_control_integers(self, file_lines, gate_types_loc, flag_irs, counter_gates):
        # if flag_irs = True  -> consider group.irs_ for non-extended groups; 
        # if flag_irs = False -> consider group.irs_complement_ for extended groups;
        func_flag_work = lambda flag_ext: not flag_ext if flag_irs else flag_ext
        keyword_gate = "gate" if flag_irs else "igate"

        for i_gate in range(len(gate_types_loc)):
            gate_type = gate_types_loc[i_gate]
            groups_R = self.dict_gates_[gate_type]
            for i_section in range(self.circ_.N_sections_):
                one_section = groups_R[i_section]
                if one_section is None:
                    continue
                ancilla_integers = self.circ_.get_anc_integers_from_section_index(i_section)
                N_groups = len(one_section)
                for i_group in range(N_groups):
                    one_group = one_section[i_group]
                    if func_flag_work(one_group.flag_extended_):
                        group_angle = one_group.a_
                        control_integers = one_group.irs_ if flag_irs else one_group.irs_complement_
                        N_integers = len(control_integers)
                        for i_integer in range(N_integers): 
                            ir = control_integers[i_integer]
                            input_integers = self.circ_.compute_integers_in_input_registers(ir)
                            one_line = "{:s} {:s} ae 1 {:19.12e} ".format(
                                keyword_gate, gate_type, group_angle
                            )
                            one_line += self.circ_.write_line_for_control_integers(input_integers,   True)
                            one_line += self.circ_.write_line_for_control_integers(ancilla_integers, False)
                            one_line += "end_gate"
                            file_lines.append(one_line)
                            counter_gates += 1
        return counter_gates


    def sort_gates_groups(self):
        for i_gate in range(len(self.gate_types_)):
            gate_type = self.gate_types_[i_gate]
            groups_R  = self.dict_gates_[gate_type]
            groups_R  = self.sort_groups(groups_R)


    # sorted_groups_R[i-section][i-group] = SortedGroup__: 
    # for each register, groups are sorted from the largest to the smallest;
    def sort_groups(self, groups_R):
        sorted_groups_R = [None]*self.circ_.N_sections_
        for i_section in range(self.circ_.N_sections_):
            one_section = groups_R[i_section]
            if one_section is None:
                continue

            N_groups = len(one_section)
            sorted_groups_R[i_section] = [None] * N_groups
            unsorted_groups_R = [None] * N_groups

            group_sizes = np.zeros(N_groups, dtype = int)
            for i_group in range(N_groups):
                one_group = one_section[i_group]
                N_zones = len(one_group.irs_start_)

                # get all row integers for this group:
                integers_group = np.array([], dtype=int)
                for i_zone in range(N_zones):
                    integers_loc = np.array(range(
                        one_group.irs_start_[i_zone],
                        one_group.irs_end_[i_zone]
                    ))
                    integers_group = np.concatenate(
                        (integers_group, integers_loc)
                    )

                # find integers encoded into registers:
                N_ints_full = len(integers_group)
                group_sizes[i_group] = N_ints_full
                unsorted_groups_R[i_group] = integers_group

            # sort the groups from the largest to the smallest:
            indices_sorted_from_least_to_largest = np.argsort(group_sizes) # from smallest to largest;
            for i_group in range(N_groups): 
                i_group_prev = indices_sorted_from_least_to_largest[i_group]
                sorted_groups_R[i_section][i_group] = one_section[i_group_prev]  # ! reference !
                sorted_groups_R[i_section][i_group].irs_ = unsorted_groups_R[i_group_prev]
            sorted_groups_R[i_section].reverse()
        return sorted_groups_R
    

    def extend_gates_groups(self):
        for i_gate in range(len(self.gate_types_)):
            gate_type = self.gate_types_[i_gate]
            groups_R  = self.dict_gates_[gate_type]
            groups_R  = self.extend_sorted_groups(groups_R)


    # extend first ngrp_ext groups in each section:
    def extend_sorted_groups(self, groups_R):
        N_regs = self.circ_.input_regs_.N_regs_
        for i_section in range(self.circ_.N_sections_):
            one_section = groups_R[i_section]
            if one_section is None:
                continue
            counter_ext_group = 0
            N_groups = len(one_section)
            for i_group in range(N_groups):
                one_group = one_section[i_group]

                # -- form matrices with bitstrings --
                N_rows = len(one_group.irs_)
                flag_ext = False
                reg_control_qubits = [None] * N_regs
                if N_rows > 1:
                    counter_ext_group += 1
                    if counter_ext_group < self.ngrp_ext_:
                        flag_ext = True
                        matrices_bs_regs = [None] * N_regs
                        for i_reg in range(N_regs):
                            matrices_bs_regs[i_reg] = np.zeros(
                                (N_rows, self.circ_.input_regs_.nqs_[i_reg]), dtype=int
                            )

                        for i_row in range(N_rows):
                            integers_row = self.circ_.compute_integers_in_input_registers(one_group.irs_[i_row])
                            for i_reg in range(N_regs):
                                nq = self.circ_.input_regs_.nqs_[i_reg]
                                matrices_bs_regs[i_reg][i_row, :] = mix.find_bit_array_of_int(integers_row[i_reg], nq)

                        # in each register, find first most significant qubits where bits stay constant: 
                        for i_reg in range(N_regs): 
                            matrix_bs = matrices_bs_regs[i_reg]
                            reg_control_qubits[i_reg] = np.zeros(self.circ_.input_regs_.nqs_[i_reg], dtype=int)
                            reg_control_qubits[i_reg].fill(2)
                            counter_qubit = 0
                            while True:
                                bit_init = matrix_bs[0, counter_qubit]
                                bit_inv = mix.inverse_bit(bit_init)
                                if len(np.where(matrix_bs[:, counter_qubit] == bit_inv)[0]): 
                                    # bits are not constant in the qubit counter_qubit
                                    break
                                else:
                                    # bits are constant in the qubit counter_qubit
                                    reg_control_qubits[i_reg][counter_qubit] = bit_init
                                    counter_qubit += 1
                                    if counter_qubit == self.circ_.input_regs_.nqs_[i_reg]:
                                        break
                one_group.flag_extended_    = flag_ext
                one_group.control_reg_bits_ = reg_control_qubits if flag_ext else None

                # find the complement group:
                if flag_ext:
                    one_group.construct_complement(self.circ_)
        return groups_R   



# -------------------------------------------------------------------------------------------------
# --- Functions ---
# -------------------------------------------------------------------------------------------------
def normalize_matrix_A(A, D):
    min_D = np.min(np.min(np.abs(D[np.nonzero(D)])))
    max_A = np.max(np.max(np.abs(A)))
    A_norm = min_D * A / max_A

    min_A = np.min(np.min(np.abs(A[np.nonzero(A)])))
    print("amin.(excl. zero) value in D: \t\t{:0.3e}".format(min_D))
    print("amax. value in A: \t\t\t{:0.3e}".format(max_A))
    print("amin.(excl. zero) value in A: \t\t{:0.3e}".format(min_A))
    print()

    max_A_norm = np.max(np.max(np.abs(A_norm)))
    min_A_norm = np.min(np.min(np.abs(A_norm[np.nonzero(A_norm)])))
    print("amax. value in A-norm: \t\t\t{:0.3e}".format(max_A_norm))
    print("amin. (excl. zero)  value in A-norm: \t{:0.3e}".format(min_A_norm))

    return A_norm



# E.g.: A[i,j] = D[i,j] * sin(theta/2);
# We assume that the target ancilla is in the unit state:
@jit(nopython=True)
def compute_angles(A, D):
    N = A.shape[0]
    
    Tx = np.zeros((N,N)); Tx.fill(np.nan)
    Ty = np.zeros((N,N)); Ty.fill(np.nan)
    Tz = np.zeros((N,N)); Tz.fill(np.nan)
    for ir in range(N):
        for ic in range(N):
            va = A[ir,ic]
            d_coef = D[ir, ic]
            va_r = np.real(va)
            va_i = np.imag(va)
            if(np.abs(va) > 0):
                if(np.abs(va_r) > 0):
                    # a complex value:
                    if(np.abs(va_i) > 0):
                        v_amp, v_phase = cmath.polar(va)
                        Ty[ir, ic] = 2. * np.arcsin(-v_amp / d_coef)
                        Tz[ir, ic] = 2. * v_phase
                    # a pure real value:
                    else:
                        Ty[ir,ic] = 2. * np.arcsin(-va_r / d_coef)
                # a pure imaginary value:
                else: 
                    Tx[ir,ic] = 2. * np.arcsin(-va_i / d_coef)
    return Tx, Ty, Tz


# grid_R[i-section][ir] = angle
def create_grid_of_gates(circ, T): 
    grid_R = circ.init_grid_of_gates()
    N = T.shape[0]  
    for ir in range(N):  
        integers_row = circ.compute_integers_in_input_registers(ir)
        for ic in range(N):
            if np.isnan(T[ir, ic]):
                continue

            integers_columns = circ.compute_integers_in_input_registers(ic)
            integers_anc_column = \
                circ.compute_integers_in_ancilla_registers(integers_row, integers_columns)
            
            if None in integers_anc_column:
                print("Error: specified circuit structure cannot encode the given matrix.")
                return None
            
            i_section = circ.get_section_index(integers_anc_column)
            grid_R[i_section][ir] = T[ir, ic]
    return grid_R



# input: grid_R[i-section][ir] = angle;
# output: groups_R[i-section][i-group] = Group__(angle);
# If flag_neighbor = True, each group consists only of neighbor rows.
def create_groups(circ, grid_R, flag_neighbor=False):
    groups_R = [None] * circ.N_sections_

    N = 1 << circ.input_regs_.nq_
    for i_section in range(circ.N_sections_):
        section_one = grid_R[i_section]
        if all(np.isnan(section_one)): # empty section
            continue
        groups_R[i_section] = []

        # a counter to iterate over each non-None row in the section;
        # (it is possible to improve the routine by considering only rows with non-nan):
        counter_row = np.where(np.isnan(section_one) == False)[0][0]

        dict_angles = {} # to find a group using its unique angle;
        counter_groups = -1 # to count the number of groups in the given section;
        group_angle = section_one[counter_row] # each group is associated with a unique angle;
        while counter_row < N:
            if group_angle in dict_angles:
                oo_group = groups_R[i_section][ dict_angles[group_angle] ]
            else:
                oo_group = Group__(angle=group_angle)

            oo_group.irs_start_.append(counter_row)
            curr_angle = group_angle
            while group_angle == curr_angle: # compare with a predefined precision?
                counter_row += 1
                if counter_row == N:
                    break
                curr_angle = section_one[counter_row]
            oo_group.irs_end_.append(counter_row)

            if group_angle not in dict_angles:
                counter_groups += 1
                if not flag_neighbor:
                    dict_angles[group_angle] = counter_groups
                groups_R[i_section].append(Group__(obj_group=oo_group)) 

            while np.isnan(curr_angle):
                counter_row += 1
                if counter_row == N:
                    break
                curr_angle = section_one[counter_row]
            group_angle = curr_angle # next angle          
    return groups_R


# Compare two matrices.
def compare_matrices(circ, A_recon, A):
    N = 1 << circ.input_regs_.nq_

    Matrix_diff_real = np.zeros((N, N))
    Matrix_diff_imag = np.zeros((N, N))
    for ir in range(N):
        for ic in range(N):
            ar, ai = np.real(A_recon[ir, ic]), np.imag(A_recon[ir, ic])
            br, bi = np.real(A[ir, ic]),       np.imag(A[ir, ic])
            # Matrix_diff_real[ir,ic] = np.abs(ar - br) / ar
            # Matrix_diff_imag[ir,ic] = np.abs(ai - bi) / ai

            Matrix_diff_real[ir,ic] = np.abs(ar - br)
            Matrix_diff_imag[ir,ic] = np.abs(ai - bi)
            if Matrix_diff_real[ir,ic] > 1e-6:
                xx = 0
            if Matrix_diff_imag[ir,ic] > 1e-6:
                xx = 0

    max_diff_real = np.max(np.max(Matrix_diff_real))
    max_diff_imag = np.max(np.max(Matrix_diff_imag))

    A_real = np.real(A)
    A_imag = np.imag(A)
    min_min_real = np.min(np.min(np.abs(A_real[np.nonzero(A_real)])))
    if len(A_imag[np.nonzero(A_imag)]):
        min_min_imag = np.min(np.min(np.abs(A_imag[np.nonzero(A_imag)])))
    else:
        min_min_imag = 0
    print("Min. abs. nonzero in A-real: {:0.3e}".format(min_min_real))
    print("Min. abs. nonzero in A-imag: {:0.3e}".format(min_min_imag))
    print()
    print("Max. abs. diff-real: {:0.3e}".format(max_diff_real))
    print("Max. abs. diff-imag: {:0.3e}".format(max_diff_imag))
    return