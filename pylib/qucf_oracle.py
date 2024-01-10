import numpy as np
import importlib as imp
import h5py
import matplotlib.pyplot as plt
import sys
import cmath
import pylib.mix as mix
import pylib.qucf_structures as qucf_str

   
def reload():
    mix.reload_module(mix)
    mix.reload_module(qucf_str)
    return

# -------------------------------------------------------------------------------------------------
# --- ASSUMPTIONS ---
# -> Input registers are less significant than ancilla registes;
# -> For each input register, one must have a corresponding ancilla register;
# -------------------------------------------------------------------------------------------------


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
        # print("Adding a register '{:s}'...".format(name))
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

    prec_ = None
    ff_prec_ = None

    ancilla_regs_ = None 
    input_regs_ = None
    nq_ = 0

    # each section corresponds to 
    # a column (for an absolute ancilla register) or
    # a diagonal (for a relative ancilla register):
    N_sections_ = None

    def __init__(self):
        self.prec_ = 1e-12
        self.ff_prec_ = "{:0." + "{:d}".format(-int(np.log10(self.prec_))) + "e}"
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
    def init_grid_of_sections(self):
        self.compute_N_sections()

        grid_of_gates = [None] * self.N_sections_

        Nr = 1<<self.input_regs_.nq_
        for i_section in range(self.N_sections_):
            grid_of_gates[i_section] = np.zeros(Nr, dtype=complex); 
            grid_of_gates[i_section].fill(np.nan)
        return grid_of_gates
    

    # integers_anc_column = [int_in_most_significant_areg ... int_in_less_significant_areg]: 
    #   integers encoded in ancilla registers;
    # given integers are encoded in ancilla registers (one integer in each register),
    # find a section corresponding to these encoded integers;
    # hint: the least-significant ancilla register encodes first several sections:
    def get_section_index(self, integers_anc_column, i_anc=0, i_section=0):
        anc = self.ancilla_regs_.get_reg(i_anc)
        if anc.flag_rel_:
            i_loc_section = anc.rel_decode_int_[integers_anc_column[i_anc]]
        else:
            i_loc_section = integers_anc_column[i_anc]
        i_section = i_section * anc.Nval_ + i_loc_section

        i_anc += 1
        if(i_anc < self.ancilla_regs_.N_regs_):
            i_section = self.get_section_index(integers_anc_column, i_anc, i_section)
        return i_section


    # find a list of integers (one integer for each ancilla register) that
    # encodes the given section index section_id;
    # anc_integers[0]: the resulting integer in the most-significant ancilla register.
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
        if anc.flag_rel_:
            integer_anc = anc.rel_values_[loc_section_id]
        else:
            integer_anc = loc_section_id

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
        # integers_row = [None] * N_input_regs
        integers_row = np.zeros(N_input_regs,dtype=int); 
        integers_row.fill(np.nan)
        id_start = 0
        for ireg in range(N_input_regs):
            nq_reg = self.input_regs_.nqs_[ireg]
            bitstring_reg = bitstring[id_start:(id_start + nq_reg)]
            integers_row[ireg] = mix.find_int_from_bit_array(bitstring_reg)
            id_start += nq_reg
        return integers_row


    # assume that each ancilla register corresponds to an input register:
    # irs: integers encoded into input registers to represent a matrix row index;
    # ics: integers (that might be) encoded into input registers 
    #   to represent a matrix column index;
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
    

    # ir - matrix row index;
    # ic - matrix column index;
    def compute_integers_in_ancillae_from_row_column(self, ir, ic):
        input_irs = self.compute_integers_in_input_registers(ir)
        integers_columns = self.compute_integers_in_input_registers(ic)
        ancilla_ics = self.compute_integers_in_ancilla_registers(
            input_irs, integers_columns
        )
        return ancilla_ics

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

        # input registers:
        if flag_input:
            res_line = "control_e {:d} ".format(N_regs)
            for i_reg in range(N_regs):
                res_line += "{:s} {:d} ".format(self.input_regs_.names_[i_reg], reg_integers[i_reg])
        # control registers:
        else:
            res_line = ""
            for i_reg in range(N_regs):
                one_reg = self.ancilla_regs_.get_reg(i_reg)
                reg_int = reg_integers[i_reg] 
                keyword_control = "control_e "
                if one_reg.flag_rel_ and reg_integers[i_reg] == 0:
                    # zero can be represented by [00...000] and [10...000]:
                    reg_int = (1 << (one_reg.nq_-1)) - 1  
                    keyword_control = "ocontrol "
                res_line += keyword_control + "{:s} {:d} ".format(one_reg.name_, reg_int)
        return res_line
    
    def get_key(self, a):
        keyword = self.ff_prec_.format(a.real) + " " + self.ff_prec_.format(a.imag)
        return keyword
    
    def get_line_angles(self, angles):
        ay, az = angles
        if ay is None:
            print("Error in line generation: ay is None")
            sys.exit(-1)
        line_angles = self.ff_prec_.format(ay)
        if az is not None:
            line_angles = self.ff_prec_.format(az) + " " + line_angles
        return line_angles

        

# -------------------------------------------------------------------------------------------------
# --- Group ---
# A group consists of several zones on a given matrix column or diagonal, 
# where matrix elements have the same value. 
# -------------------------------------------------------------------------------------------------
class Group__:
    # a complex value attributed to the group:
    __v_ = None 

    # angles associated with the group value:
    __ay_ = None
    __az_ = None

    # the core of the group: the array of rows of the initial group:
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

    # row indices of the complement to the group:
    irs_complement_ = None

    # flag which is used by some routines to know whether 
    #   the current group should be taken into account or not:
    flag_removed_ = None

    # where the corresponding gates gates should be inverted:
    flag_inverse_ = None

 
    def __init__(self, group_value):
        self.set_value(group_value)
        self.irs_ = np.array([], dtype=int)
        self.irs_complement_ = np.array([], dtype=int)
        self.flag_removed_ = False
        self.flag_inverse_ = False
        return

    def set_value(self, new_value):
        self.__v_ = new_value
        self.__ay_, self.__az_ = mix.calc_angles_from_a_value(self.__v_)
        return
    
    def get_value(self):
        return self.__v_
    
    def set_angles(self, ay, az):
        self.__ay_, self.__az_ = (ay, az)
        self.__v_ = mix.Rc(ay,az)[0, 0]
        return

    def get_angles(self):
        return self.__ay_, self.__az_

    def invert_gate(self):
        self.flag_inverse_ = not self.flag_inverse_
        return 
    
    # Construct the full range of sorted rows covered by the given group: 
    def give_full_irs(self):
        if len(self.irs_complement_) > 0:
            irs_work = np.concatenate((self.irs_, self.irs_complement_))
        else:
            irs_work = self.irs_
        return np.sort(irs_work)

    # find rows where the complement of the groups sits:
    def construct_complement(self, circ):
        if not self.flag_extended_:
            return
        N_full = 1 << circ.input_regs_.nq_

        # area covered by control nodes:
        mask_control = np.zeros(N_full, dtype=int)
        mask_control = self.form_control_mask(circ, mask_control)

        # original area:
        mask_orig = np.zeros(N_full, dtype=int)
        N_orig = len(self.irs_)
        for ii in range(N_orig):
            mask_orig[self.irs_[ii]] = 1

        # row indices of the complement:
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
        temp_indices = np.where(control_bits != 2)
        control_bits = control_bits[temp_indices]

        nq = circ.input_regs_.nqs_[i_reg]

        nqf = len(control_bits)
        # nqf = np.where(control_bits == 2)[0]
        # nqf = nq if len(nqf) == 0 else nqf[0]
        return control_bits, nq, nqf

    
    def form_control_mask(self, circ, mask_control, i_reg = 0, sh = 0):
        control_bits, nq, nqf = self.get_control(circ, i_reg)

        int_start = 0
        for i_bit in range(nqf):
            int_start += control_bits[i_bit] * (1 << (nq - 1 - i_bit))

        N_interval = 1 << (nq - nqf) # YBIT: works only for high sign. control qubits;
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
                one_line += "{:s} {:d} ".format(keyword, N_control_regs)
                for i_reg in range(N_regs):
                    if flags[i_reg]:
                        one_line += "{:s}[".format(circ.input_regs_.names_[i_reg])
                        N_nodes = len(controls[i_reg])
                        for counter_node in range(N_nodes):
                            str_line = "{:d} " if counter_node < (N_nodes-1) else "{:d}"
                            one_line += str_line.format(controls[i_reg][counter_node])
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
                    zero_controls[i_reg].append(-1-iq)
                if control_bits[iq] == 1:
                    flags_unit[i_reg] = True
                    unit_controls[i_reg].append(-1-iq)
        res_line = write_line(flags_zero, zero_controls, "", "ocontrol")
        res_line = write_line(flags_unit, unit_controls, res_line, "control")
        return res_line



# -------------------------------------------------------------------------------------------------
# --- System of gates ---
# -------------------------------------------------------------------------------------------------
class SystemGates__:
    circ_ = None
    groups_ = None
    __Nr_small_ = None

    # the parameter indicates how many qubits in input qubits (from all input registers) 
    #   should be investigated to split a group:
    n_split_ = 1

    # used in reconstruct_matrix_using_GRID;
    # grid_values_[i-section][i-row] = [first-group, second-group, ...];
    # if len(grid_values_[i-section][i-row]) > 1, a correcting gate is required 
    #   at the row with the index i-row in the section with the index i-section;
    grid_values_ = None
    
    def __init__(self, circ, groups):
        self.circ_ = circ
        self.groups_ = groups
        self.__Nr_small_ = 4
        return
    
    def set_rows_limit_for_non_extension(self, Nr_non_ext):
        self.__Nr_small_ = Nr_non_ext
        return

    
    # Consider only the core of each group to reconstruct the matrix;
    # can be applied only before launching 
    # the functions SystemGates__.merge_groups and SystemGates__.correct_groups.
    def reconstruct_matrix_using_GROUPS(self):    
        N = 1 << self.circ_.input_regs_.nq_
        N_sections = len(self.groups_) # nonsparsity
        Nnz = 0
        D_sections_columns = np.zeros((N_sections, N), dtype=int);     D_sections_columns.fill(np.nan)
        D_sections_values  = np.zeros((N_sections, N), dtype=complex); D_sections_values.fill(np.nan)
        for i_section in range(N_sections):
            one_section = self.groups_[i_section]
            if not one_section:
                continue # empty section
            anc_integers = self.circ_.get_anc_integers_from_section_index(i_section)
            N_groups = len(one_section)
            for i_group in range(N_groups): # consider all groups whithin each section;
                one_group = one_section[i_group]
                N_rows = len(one_group.irs_)
                for counter_row in range(N_rows): # consider all rows within each group;
                    ir = one_group.irs_[counter_row]
                    integers_input_regs = self.circ_.compute_integers_in_input_registers(ir)
                    ic = self.circ_.get_column_index_from_anc_integers(anc_integers, integers_input_regs)

                    # save the matrix element's value:
                    Nnz += 1
                    D_sections_columns[i_section, ir] = ic
                    D_sections_values[i_section, ir]  = one_group.get_value()
        A_recon = mix.construct_sparse_from_sections(
            N, Nnz, N_sections, D_sections_columns, D_sections_values, self.circ_.prec_
        )
        return A_recon
    


    # should be called after SystemGates__.correct_groups;
    # Consider extended and correcting gates to reconstruct the matrix;
    def reconstruct_matrix_using_GRID(self): 
        N = 1 << self.circ_.input_regs_.nq_
        N_sections = len(self.groups_) # nonsparsity
        Nnz = 0
        D_sections_columns = np.zeros((N_sections, N), dtype=int);     D_sections_columns.fill(np.nan)
        D_sections_values  = np.zeros((N_sections, N), dtype=complex); D_sections_values.fill(np.nan)
        for i_section in range(N_sections):
            one_grid = self.grid_values_[i_section]
            if not one_grid:
                continue 
            anc_integers = self.circ_.get_anc_integers_from_section_index(i_section)
            for ir in range(N):
                if one_grid[ir] is None:
                    continue

                # find the column index:
                integers_input_regs = self.circ_.compute_integers_in_input_registers(ir)
                ic = self.circ_.get_column_index_from_anc_integers(anc_integers, integers_input_regs)

                # compute the value of the matrix element: 
                res_value, _ = mix.action_of_RyRc_gates(one_grid[ir])

                # save the matrix element's value:
                Nnz += 1
                D_sections_columns[i_section, ir] = ic
                D_sections_values[i_section, ir]  = res_value
        A_recon = mix.construct_sparse_from_sections(
            N, Nnz, N_sections, D_sections_columns, D_sections_values, self.circ_.prec_
        )
        return A_recon
    

    def count_groups(self):
        # print("N = {:d}".format(1 << self.circ_.input_regs_.nq_))
        counter_groups = 0
        for i_section in range(self.circ_.N_sections_):
            one_section = self.groups_[i_section]
            if one_section is None:
                continue
            for one_group in one_section:
                if not one_group.flag_removed_:
                    counter_groups += 1
        print("Number of groups: {:d}".format(counter_groups))
        print()
        return
    

    # Assume that gates act on a target ancilla qubit called "ae";
    def construct_circuit_OH(self, path, filename):
        file_lines = []
        counter_gates = 0
        counter_compl_gates = 0
        for i_section in range(self.circ_.N_sections_):
            one_section = self.groups_[i_section]
            if one_section is None:
                continue

            ancilla_integers = self.circ_.get_anc_integers_from_section_index(i_section)
            str_line = self.circ_.write_line_for_control_integers(ancilla_integers, False)
            one_line  = "//--------------------------------------------------------------------------\n"
            one_line += "//--------------------------------------------------------------------------\n"
            one_line += "with {:s} do".format(str_line)
            file_lines.append(one_line)

            for one_group in one_section:
                if one_group.flag_removed_:
                    continue
                ay, az = one_group.get_angles()
                line_angle = self.circ_.get_line_angles((ay, az))
                file_lines.append("// ---")

                flag_Ry = True if az is None else False
                gate_type = "Ry" if flag_Ry else "Rc"
                if one_group.flag_extended_:
                    # --- An extended group ---
                    one_line = "   gate {:s} ae 1 {:s} ".format(gate_type, line_angle)
                    one_line += one_group.write_line_for_input_control_nodes(self.circ_)
                    one_line += "end_gate"
                    file_lines.append(one_line)
                    counter_gates += 1 if flag_Ry else 2
                    counter_compl_gates += 1
                else:
                    # --- A non-extended group ---
                    line_gate = "igate" if one_group.flag_inverse_ else "gate" 
                    for ir in one_group.irs_: 
                        input_integers = self.circ_.compute_integers_in_input_registers(ir)
                        one_line = "   " + line_gate + " {:s} ae 1 {:s} ".format(gate_type, line_angle)
                        one_line += self.circ_.write_line_for_control_integers(input_integers, True)
                        one_line += "end_gate"
                        file_lines.append(one_line)
                        counter_gates += 1 if flag_Ry else 2
                        counter_compl_gates += 1
            file_lines.append("end_with")

        # --- build section CIRCUIT_STRUCTURE ---
        fullname = path + "/" + filename + ".oracle"
        ff = open(fullname, "w")
        for one_line in file_lines:
            ff.write(one_line + "\n")
        ff.close()

        # --- Log data ---
        # self.circ_.print_structure()
        print("N-gates: {:d}".format(counter_gates))
        print("N-gates (assuming Rc as a single gate): {:d}".format(counter_compl_gates))
        return
    

    # sorted_groups[i-section][i-group] = SortedGroup__: 
    # for each register, groups are sorted from the largest to the smallest;
    def sort_groups(self):
        sorted_groups = [None]*self.circ_.N_sections_
        for i_section in range(self.circ_.N_sections_):
            one_section = self.groups_[i_section]
            if one_section is None:
                continue

            N_groups = len(one_section)
            sorted_groups[i_section] = [None] * N_groups
            group_sizes = np.zeros(N_groups, dtype = int)
            for i_group in range(N_groups):
                group_sizes[i_group] = len(one_section[i_group].irs_)

            # sort the groups from the largest to the smallest:
            indices_sorted_from_least_to_largest = np.argsort(group_sizes) # from smallest to largest;
            for i_group in range(N_groups): 
                i_group_prev = indices_sorted_from_least_to_largest[i_group]
                sorted_groups[i_section][i_group] = Group__(one_section[i_group_prev].get_value()) 
                sorted_groups[i_section][i_group].irs_ = np.array(one_section[i_group_prev].irs_)
            sorted_groups[i_section].reverse()
        self.groups_ = sorted_groups
        return


    # extend groups in each section:
    def extend_sorted_groups(self, B_fixed):
        N = 1 << self.circ_.input_regs_.nq_
        N_regs = self.circ_.input_regs_.N_regs_
        for i_section in range(self.circ_.N_sections_):
            one_section = self.groups_[i_section]
            if one_section is None:
                continue
            counter_ext_group = 0

            # *** Prepare data to restrict complements of the section groups ***
            irs_exclude = np.array([], dtype=int)
            anc_integers = self.circ_.get_anc_integers_from_section_index(i_section)

            # to exclude elements beyond limits of the matrix:
            integers_input_regs = self.circ_.compute_integers_in_input_registers(0)
            ic = self.circ_.get_column_index_from_anc_integers(anc_integers, integers_input_regs)
            shift_rc = ic
            if ic < 0:
                irs_exclude = np.concatenate(( irs_exclude, np.array(range(np.abs(ic))) ))
            del anc_integers, integers_input_regs
            
            ic = (N-1) + shift_rc
            if ic >= N:
                irs_exclude = np.concatenate(( irs_exclude, np.array(range(N-(ic-N)-1, N)) ))
                
            # to exclude elements where the matrix elements equal zero:
            irs_zeros = np.zeros(N, dtype=int)
            N_zeros = 0
            for i_row in range(N):
                ic = i_row + shift_rc
                if ic < 0 or ic >= N:
                    continue
                if np.abs(B_fixed.get_matrix_element(i_row, ic)) < self.circ_.prec_:
                    N_zeros += 1
                    irs_zeros[N_zeros-1] = i_row
            irs_exclude = np.concatenate((irs_exclude, irs_zeros[:N_zeros]))
            del shift_rc, ic

            # *** Look for constant bits in each non-small group to extend it ***
            N_groups = len(one_section)
            for i_group in range(N_groups):
                one_group = one_section[i_group]

                # -- form matrices with bitstrings --
                N_rows = len(one_group.irs_)
                flag_ext = False
                reg_control_qubits = [None] * N_regs
                if N_rows > self.__Nr_small_: # if the group size is larger of the imposed threshold;
                    counter_ext_group += 1
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

                # *** Find the complement of the group ***
                if flag_ext:
                    one_group.construct_complement(self.circ_)

                    # restrict the complement range:
                    if len(irs_exclude) > 0:
                        if len(irs_exclude) == 1:
                            one_group.irs_complement_ = np.delete(
                                one_group.irs_complement_,
                                np.where(one_group.irs_complement_ == irs_exclude[0])
                            )
                        else:
                            irs_complement_new = np.zeros(
                                (len(one_group.irs_complement_)), 
                                dtype=int
                            )
                            N_new_compl = 0
                            for ir_comp in one_group.irs_complement_:
                                if ir_comp not in irs_exclude:
                                    N_new_compl += 1
                                    irs_complement_new[N_new_compl-1] = ir_comp
                            one_group.irs_complement_ = irs_complement_new[:N_new_compl]
        return 


    # -> During a single launch of the function, each group can remain the same or be split.
    #   A group can be split only once during a single launch of the function.
    # -> To split groups once again, launch the function again.
    # -> The function investigates the first self.n_split_ qubits
    #   (starting from the most significant qubit).
    #   If a qubit does not keep its bit constant within the group, then the group is split into two groups
    #   such that this qubit keeps constant its bit in each of these new groups.
    def split_groups(self):
        if self.n_split_ > self.circ_.input_regs_.nq_:
            print("Error: SystemGates__.n_split_ cannot be larger than Circuit__.input_regs_.nq_.")
            return None

        for i_section in range(self.circ_.N_sections_):
            one_section = self.groups_[i_section]
            if one_section is None:
                continue
            N_groups = len(one_section)
            counter_group = 0
            while True:
                one_group = one_section[counter_group]
                N_rows = len(one_group.irs_)
                if N_rows > 1:
                    matrix_bs = np.zeros((N_rows, self.circ_.input_regs_.nq_), dtype=int)
                    for i_row in range(N_rows):
                        bs = mix.find_bit_array_of_int(one_group.irs_[i_row], self.circ_.input_regs_.nq_)
                        matrix_bs[i_row, :] = bs

                    # check first n_split_ qubits to find a qubit where bit is not conserved:
                    for counter_qubit in range(self.n_split_):
                        bits_row = matrix_bs[:, counter_qubit]
                        bit_init = bits_row[0]
                        bit_inv  = mix.inverse_bit(bit_init)
                        array_found = np.where(bits_row == bit_inv)[0]
                        if len(array_found): 
                            id_split = array_found[0]
                            l_group = Group__(one_group.get_value())
                            r_group = Group__(one_group.get_value())

                            l_group.irs_ = one_group.irs_[0:id_split]
                            r_group.irs_ = one_group.irs_[id_split:]

                            one_section[counter_group] = r_group
                            one_section.insert(counter_group, l_group)

                            N_groups += 1
                            counter_group += 1
                            break
                counter_group += 1
                if counter_group >= N_groups:
                    break
        return  
    
    
    # Use the function after the function "extend_sorted_groups" has been launched.
    # Merge groups that have the same values and where one group overlaps the other one completely***
    # Assume that an extended group Gext overlaps with a group Go, and assume that
    # Gext.__v_ == Go.__v_.
    # 1. If Go is non-extended, remove from Go.irs_ all row indices that 
    #   already present in Gext.irs_complement_.
    # 2. If Go is extended (have control bits), remove the whole Go if 
    #   Gext overlaps with all irs_ and all irs_.complement_ of the group Go.
    def merge_groups(self):
        for i_section in range(self.circ_.N_sections_):
            one_section = self.groups_[i_section]
            if one_section is None:
                continue
            N_groups = len(one_section)

            # Here, we assume that the groups have been sorted from 
            # the largest (extended) to the smallest (non-extended) groups:
            for i_group in range(N_groups):
                one_group = one_section[i_group]

                # skip if the group has been removed previously have a zero-size complement: 
                if one_group.flag_removed_ or (len(one_group.irs_complement_) == 0):
                    continue
                irs_work = one_group.give_full_irs()

                # check whether the group overlaps with other non-larger groups:
                for other_group in one_section[i_group+1:]:
                    if other_group.flag_removed_:
                        continue

                    # check whether the groups have the same values:
                    if not mix.compare_complex_values(
                        other_group.get_value(), one_group.get_value(), self.circ_.prec_
                    ):
                        continue
                    if other_group.flag_extended_:
                        # if the other group is extended, remove it from consideration 
                        #   if and only if 
                        # the other group is overlapped completely with the current group:
                        irs_other = other_group.give_full_irs()
                        other_group.flag_removed_ = True
                        for ir_check in irs_other:
                            if ir_check not in irs_work:
                                other_group.flag_removed_ = False
                                break
                    else:
                        # if the other group is not extended, 
                        # remove all rows which overlap with the current group
                        # from the other group's irs_:
                        Nr_new = 0
                        irs_new = np.zeros(len(other_group.irs_), dtype=int)
                        for ir_check in other_group.irs_:
                            if ir_check not in irs_work:
                                Nr_new += 1
                                irs_new[Nr_new-1] = ir_check
                        other_group.irs_ = irs_new[:Nr_new]
        return


    def correct_close_groups(self):
        calc_angles = lambda value_goal, init_vec: mix.find_correcting_angles_for_Rc_FLOAT(
            value_goal, init_vec, self.circ_.prec_
        )

        for i_section in range(self.circ_.N_sections_):
            one_section = self.groups_[i_section]
            if one_section is None:
                continue
            N_groups = len(one_section)

            # Here, we assume that the groups have been sorted from 
            # the largest (extended) to the smallest (non-extended) groups:
            for i_group in range(N_groups):
                one_group = one_section[i_group]

                # skip if the group has been removed previously have a zero-size complement: 
                if one_group.flag_removed_ or (len(one_group.irs_complement_) == 0):
                    continue

                # check whether the group overlaps with other non-larger groups:
                for other_group in one_section[i_group+1:]:

                    # consider only extended groups
                    if not other_group.flag_extended_ or other_group.flag_removed_:
                        continue

                    # check whether the groups intersect significantly:
                    # (we are not interested in complement-complement intersections)
                    counter_intersection = 0
                    for ir_check in other_group.irs_:
                        if ir_check in one_group.irs_complement_:
                            counter_intersection += 1

                    if counter_intersection > self.__Nr_small_:
                        # if the overlapping between the groups is large enough, then 
                        # correct the group so that the combined action of two groups
                        # would produce a correct matrix element's value:
                        obtained_vec = mix.action_of_RyRc_gates([one_group])
                        required_value = other_group.get_value()
                        ay_c, az_c = calc_angles(required_value, obtained_vec)

                        # if the computation of angles is successful, correct the other group:
                        if ay_c is not None:
                            other_group.set_angles(ay_c, az_c)
                            xx = 0
        return


    # In each row where several groups act (due to the extension), 
    # add one or several correcting groups to adjust the value at this row.
    def correct_groups(self, B_fixed):
        # --- FUNCTION: add a new group ---
        def add_a_group(
                one_section, counter_row, counter_value, 
                new_value, id_special,
                flag_inverse, last_group,
                groups_on_row,
                ay_c = None, az_c = None
            ):
            new_group = Group__(new_value)
            if ay_c is not None:
                # define the group value by setting the angles (to exclude phase shift):
                new_group.set_angles(ay_c, az_c) 
            new_group.irs_ = [counter_row]
            if flag_inverse:
                new_group.invert_gate()
            one_section.append(new_group)
            if counter_value == id_special:
                if last_group.flag_extended_:
                    groups_on_row.append(new_group)
                else:
                    # if a non-extended group, remove this row from the last group:
                    last_group.irs_ = np.delete(
                        last_group.irs_, 
                        np.where(last_group.irs_ == counter_row)
                    )
                    groups_on_row[-1] = new_group
            else:
                groups_on_row.append(new_group)
            return
        # -----------------------------------------------
        # calc_angles = lambda value_goal, init_vec: mix.find_correcting_angles_for_Rc(
        #     value_goal, init_vec, self.circ_.prec_
        # )
        calc_angles = lambda value_goal, init_vec: mix.find_correcting_angles_for_Rc_FLOAT(
            value_goal, init_vec, self.circ_.prec_
        )
        # -----------------------------------------------
        Nr = 1 << self.circ_.input_regs_.nq_
        N_secs = self.circ_.N_sections_
        self.grid_values_ = [None] * N_secs
        for i_section in range(N_secs):
            # consider a section;
            # further, "on(at) a row ir" will mean that 
            # we consider one element with the index ir that lies in this section:
            one_section = self.groups_[i_section]
            if one_section is None:
                continue
            anc_integers = self.circ_.get_anc_integers_from_section_index(i_section)

            # one_grid[ir] is an array of groups acting on the row ir:
            one_grid = [None] * Nr 

            # *** Sort all complements according to their positions ***
            for one_group in one_section: 
                if one_group.flag_removed_:
                    continue
                irs_work = one_group.give_full_irs()
                for ir in irs_work:
                    if one_grid[ir] is None:
                        one_grid[ir] = [one_group]
                    else:
                        one_grid[ir].append(one_group)
                    
            # *** Find correcting groups for each overlapped element ***
            for counter_row in range(Nr):
                if one_grid[counter_row] is None:
                    continue # some grids are not completely filled;
                if len(one_grid[counter_row]) == 1:
                    continue # this element is not overlapped;
                groups_on_row = one_grid[counter_row]
                last_group = groups_on_row[-1] 

                # the value that it's necessary to obtain on this row:
                integers_input_regs = self.circ_.compute_integers_in_input_registers(counter_row)
                ic = self.circ_.get_column_index_from_anc_integers(anc_integers, integers_input_regs)
                required_value = B_fixed.get_matrix_element(counter_row, ic)

                # if the last group is extended, consider all values at the row;
                # if not, consider all but the last values;
                Nc_comp = len(groups_on_row) \
                    if last_group.flag_extended_ else (len(groups_on_row) - 1)
                groups_comp = groups_on_row[:Nc_comp]

                # compute the vector that obtained after the action of all groups on the row:
                # (assume that group-gates act on an ancilla qubit initialized in the zero state)
                obtained_vec = mix.action_of_RyRc_gates(groups_comp)

                # if the combined action of the groups results in a correct value, then 
                # do not modify groups here:
                if mix.compare_complex_values(obtained_vec[0], required_value, self.circ_.prec_):
                    continue

                # try to compute angles for a correcting group:
                ay_c, az_c = calc_angles(required_value, obtained_vec)

                # if the computation of the angles fails, keep adding groups 
                # to reverse action of previous groups 
                # either until the computation is successful
                # or until all previous groups are reversed:
                counter_value = 0
                while ay_c is None:
                    counter_value += 1

                    # add a reverse group:
                    add_a_group(
                        one_section, counter_row, counter_value, 
                        groups_comp[Nc_comp - counter_value].get_value(), 
                        1, True, last_group, groups_on_row
                    )

                    # if all previous groups were reversed, then 
                    # there is no need to add a correcting group:
                    if counter_value == Nc_comp:
                        break

                    # otherwise, try again to compute angles for a correcting gate:
                    obtained_vec = mix.action_of_RyRc_gates(groups_on_row)
                    ay_c, az_c = calc_angles(required_value, obtained_vec)
 
                
                if counter_value < Nc_comp:
                    # if not all groups are reversed on the row, add a correcting gate:
                    add_a_group(
                        one_section, counter_row, counter_value, 
                        0.6830692, # fake value
                        0, False, last_group, groups_on_row,
                        ay_c, az_c
                    )
                else:
                    # if all groups are reversed, add the action of the last group:
                    oo_grp = Group__(last_group.get_value())
                    oo_grp.irs_ = [counter_row]
                    if last_group.flag_inverse_:
                        oo_grp.invert_gate()
                    one_section.append(oo_grp)
                    groups_on_row.append(oo_grp)

            # *** Save the grid ***
            self.grid_values_[i_section] = one_grid
        return



# -------------------------------------------------------------------------------------------------
# --- Functions ---
# -------------------------------------------------------------------------------------------------

# B_fixed = A/D;
# grid_sections[i-section][ir] = value-of-matrix-element
# Sections are equivalent to diagonals or columns 
#   (depending on the meaning of ancilla registers, absolute or relative)
def create_grid_of_sections(circ, B_fixed): 
    grid_sections = circ.init_grid_of_sections()
    N, _, B_rows, B_columns, B_values = B_fixed.get_data()
    for ir in range(N):  
        integers_row = circ.compute_integers_in_input_registers(ir)
        for i_nz in range(B_rows[ir], B_rows[ir+1]):
            ic = B_columns[i_nz]
            v = B_values[i_nz]
            integers_columns = circ.compute_integers_in_input_registers(ic)
            integers_anc_column = \
                circ.compute_integers_in_ancilla_registers(integers_row, integers_columns)
            
            if None in integers_anc_column:
                print("Error: specified circuit structure cannot encode the given matrix.")
                return None
            
            i_section = circ.get_section_index(integers_anc_column)
            grid_sections[i_section][ir] = v
    return grid_sections


# OPTION 1: within a section, put all rows where 
# matrix elements have the same values into the same group:
# OUTPUT: groups_R[i-section][i-group] = Group__(value);
def create_groups(circ, grid_sections):
    groups_R = [None] * circ.N_sections_
    N = 1 << circ.input_regs_.nq_
    for i_section in range(circ.N_sections_):
        section_one = grid_sections[i_section]
        if all(np.isnan(section_one)): # empty section
            continue
        groups_R[i_section] = []

        # a counter to iterate over each non-None row in the section;
        # (it is possible to improve the routine by considering only rows with non-nan):
        ir = np.where(np.isnan(section_one) == False)[0][0]

        dict_values = {} # to find a group using its unique angle;
        counter_groups = -1 # to count the number of groups in the given section;

        # Each group is associated with a single value:
        vv = section_one[ir]
        while ir < N:
            keyword = circ.get_key(vv)
            if keyword in dict_values:
                oo_group = groups_R[i_section][dict_values[keyword]]
            else:
                oo_group = Group__(vv)

            ir_start = ir
            curr_vv = vv
            while mix.compare_complex_values(vv, curr_vv, circ.prec_): # compare with a predefined precision?
                ir += 1
                if ir == N:
                    break
                curr_vv = section_one[ir]
            array_rows = np.array(range(ir_start, ir))
            oo_group.irs_ = np.concatenate((oo_group.irs_, array_rows))

            if keyword not in dict_values:
                counter_groups += 1
                groups_R[i_section].append(oo_group) 
                dict_values[keyword] = counter_groups

            # next value:
            vv = curr_vv
            while np.isnan(vv):
                ir += 1
                if ir >= N:
                    break
                vv = section_one[ir]       
    return groups_R


# OPTION 2:
# Each group consists only of neighbor rows.
# E.g. consider two rows with indicies j and k:
#    j -> |j_reg1>|j_reg2>, 
#    k -> |k_reg1>|k_reg2>.
# The rows j and k are considered neighbor if 
#   |j_reg1 - k_reg1| <= 1 AND |j_reg2 - k_reg2| <= 1.
# OUTPUT: groups_R[i-section][i-group] = Group__(value);
def create_groups_neighbor(circ, grid_sections):
    # ------------------------------------------------------------------------
    def find_group(vv, irs_cond_group, dict_values, one_section):
        is_new_group = True
        res_id_prev_group = None
        keyword = circ.get_key(vv)
        if keyword not in dict_values:
            is_new_group = True
        else:
            for id_prev_group in range(len(dict_values[keyword])):
                prev_group = one_section[ dict_values[keyword][id_prev_group] ]

                # Option 2: compare all elements with all elements from the both groups
                # Option 2.1: consider the last N_rows elements in the previous group:
                N_cond_rows = len(irs_cond_group)
                N_prev_rows = len(prev_group.irs_)

                N_min = np.min([N_cond_rows, N_prev_rows])

                # if N_cond_rows > N_prev_rows:
                #     continue

                is_neighbor = True
                for counter_row in range(N_min):
                    ir_cond = irs_cond_group[N_cond_rows - counter_row - 1]
                    ir_prev = prev_group.irs_[N_prev_rows - counter_row - 1]
                    ints_cond = circ.compute_integers_in_input_registers(ir_cond)
                    ints_prev = circ.compute_integers_in_input_registers(ir_prev)
                    for ireg in range(N_regs):
                        if np.abs(ints_cond[ireg] - ints_prev[ireg]) > 1:
                            is_neighbor = False
                            break
                    if not is_neighbor:
                        break

                if is_neighbor:
                    is_new_group = False
                    res_id_prev_group = id_prev_group

                    # a group can be neighbor with several previous groups;
                    # here, we consider only the first encountered neighbor group:
                    break
        return is_new_group, keyword, res_id_prev_group

    # ---
    groups_R = [None] * circ.N_sections_
    N = 1 << circ.input_regs_.nq_
    N_regs = circ.input_regs_.N_regs_
    for i_section in range(circ.N_sections_):
        section_one = grid_sections[i_section]
        if all(np.isnan(section_one)): # empty section
            continue
        groups_R[i_section] = []

        # a counter to iterate over each non-None row in the section;
        ir = np.where(np.isnan(section_one) == False)[0][0]

        dict_values = {} # to find a group using its unique angle;
        counter_groups = -1 # to count the number of groups in the given section;

        # Each group is associated with a single value:
        curr_vv = section_one[ir]
        while ir < N:
            ir_start = ir
            vv = curr_vv # next angle  

            # --- * Find rows where the group sits on ---
            while mix.compare_complex_values(vv, curr_vv, circ.prec_): # compare with a predefined precision?
                ir += 1
                if ir == N:
                    break
                curr_vv = section_one[ir]
            array_rows = np.array(range(ir_start, ir))

            # --- Save a new group or merge it with one of previous groups ---
            flag_new_group, keyword, id_prev_group = find_group(
                vv, array_rows, dict_values, groups_R[i_section]
            )
            if flag_new_group:
                counter_groups += 1
                if keyword not in dict_values:
                    dict_values[keyword] = [counter_groups]
                else:
                    dict_values[keyword].append(counter_groups)
                oo_group = Group__(vv)
                oo_group.irs_ = array_rows
                groups_R[i_section].append(oo_group)
            else:
                counter_prev_group = dict_values[keyword][id_prev_group]
                groups_R[i_section][counter_prev_group].irs_ = np.concatenate(
                    (
                        groups_R[i_section][counter_prev_group].irs_,
                        array_rows
                    )
                )

            # --- Find next nonzero element in the section ---
            while np.isnan(curr_vv):
                ir += 1
                if ir >= N:
                    break
                curr_vv = section_one[ir]
        xx = 0
    return groups_R













