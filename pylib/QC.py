import pylib.mix as mix
import pylib.Gate as Gate
import qiskit
import sys
import numpy as np

def reload():
    mix.reload_module(mix)
    mix.reload_module(Gate)


# ------------------------------------------------
# --- Quantum circuit ---
class Circuit:
    # number of qubits
    nq = 0  

    # qubit objects that are considered as channels with an ordered set of gates
    # [0] is the most significant qubit
    qubits = []

    # most significant qubit
    qu_most = None

    # approximations:
    flag_approx_I, aerr_I, rerr_I =  False, 1e-15, 1e-3
    flag_approx_R, err_th = False, 1e-15

    def __init__(self, nq=None, approx = []):
        if nq is not None:
            self.set_qubits(nq)

        for ia, aa in enumerate(approx):
            self.set_approx(aa)

    # set approximations:
    def set_approx(self, aa):
        sel = aa[0].lower()

        if sel == "approx-i":
            self.flag_approx_I = True
            self.aerr_I = aa[1]
            self.rerr_I = aa[2]

        if sel == "approx-r":
            self.flag_approx_R = True
            self.err_th = aa[1]

    # create qubits:
    def set_qubits(self, nq):
        self.nq = nq
        self.qubits = [None] * self.nq

        self.qu_most = self.qubits[0] = Qubit(0, self.nq, flag_most = True)
        upper_qu = self.qu_most
        for iq in range(1, self.nq):
           self.qubits[iq] = Qubit(iq, self.nq, upper_qu=upper_qu)
           upper_qu = self.qubits[iq]
        self.qu_most.set_boundaries()

    # set initial quantum state:
    # q_vect = [i, j, ...] with i,j... = 0 or 1
    # size(q_vect) = self.nq !!!
    # q_vect[0] is the most significant qubit
    def set_initial_state(self, q_vect):
        if len(q_vect) != self.nq:
            print("Wrong size of initial quantum vector")
            sys.exit(-1)

        for iq in range(self.nq):
            self.qubits[iq].set_init_state(q_vect[iq]) 

    # while adding a gate, fill other qubits with empty gates
    def fill_other_qubits(self, ids_qu_work):
        for iq in range(self.nq):
            if iq not in ids_qu_work:
                self.qubits[iq].add_gate(Gate.EmptyGate(iq, self.nq))
        self.qu_most.establish_vertical_connections_between_gates()

    # output text representation of the QC
    def text_output(self, flag_print = True):
        lines = []
        for qq in self.qubits:
            oline = qq.text_output()

            if flag_print:
                print(oline)
            lines.append(oline)

        if flag_print:
            print('')

        return lines
        
    # form a qiskit circuit:
    def form_qiskit_circuit(self):
        qcr = qiskit.QuantumCircuit(self.nq)

        for iq, qq in enumerate(self.qubits):
            if qq.init_state is not None:
                qcr.initialize(qq.init_state, iq)

        self.qu_most.start_gate.qiskit_add_gate(qcr)
        self.qu_most.reset_qiskit_flags()

        return qcr

    # add CNOT gate
    def add_cnot(self, ic, it):
        obj = Gate.CNOT(it, self.nq)
        self.qubits[it].add_gate(obj)
        self.qubits[ic].add_gate(Gate.CNODE(ic, self.nq, obj, type_cnode='1'))

        self.fill_other_qubits([it, ic]) 
        return self

    # add one-qubit rotation matrix
    def add_R(self, sel_R, theta, it):
        if sel_R.lower() == "ry":
            obj = Gate.Ry(it, self.nq, theta)
        if sel_R.lower() == "rz":
            obj = Gate.Rz(it, self.nq, theta)

        self.qubits[it].add_gate(obj)
        self.fill_other_qubits([it])
        return self

    # add Pauli gate:
    def add_pauli(self, sel_pauli, it):
        if sel_pauli.lower() == "x":
            obj = Gate.X(it, self.nq)
        if sel_pauli.lower() == "y":
            obj = Gate.Y(it, self.nq)
        if sel_pauli.lower() == "z":
            obj = Gate.Z(it, self.nq)
        self.qubits[it].add_gate(obj)
        self.fill_other_qubits([it])
        return self

    def add_X(self, it):
        return self.add_pauli("x", it)
    def add_Y(self, it):
        return self.add_pauli("y", it)
    def add_Z(self, it):
        return self.add_pauli("z", it)

    # add Hadamard gate:
    def add_H(self, it):
        obj = Gate.H(it, self.nq)
        self.qubits[it].add_gate(obj)
        self.fill_other_qubits([it])
        return self

    # add Phase gate:
    def add_S(self, it, sel = "S"):
        if sel.lower() == "s":
            obj = Gate.S(it, self.nq)
        if sel.lower() == "sd":
            obj = Gate.Sd(it, self.nq)
        self.qubits[it].add_gate(obj)
        self.fill_other_qubits([it])
        return self

    def add_Sd(self, it):
        return self.add_S(it, "sd")
    
    # add multi-controlled Ry matrix:
    def add_mcR(self, sel_R, U, ids_c, it):
        if sel_R.lower() == "ry":
            obj_mcr = Gate.MCRY(it, self.nq, U)
        if sel_R.lower() == "rz":
            obj_mcr = Gate.MCRZ(it, self.nq, U)
        self.qubits[it].add_gate(obj_mcr)

        for ic in ids_c:
            obj_c = Gate.CNODE(ic, self.nq, obj_mcr, type_cnode='')
            self.qubits[ic].add_gate(obj_c)

        self.fill_other_qubits([it] + ids_c)
        return self

    # add an arbitrary single-qubit unitary gate:
    def add_U(self, U, it, label):
        obj = Gate.UnitaryST(
            id_qubit = it, 
            n_qubits = self.nq,
            U      = U,
            label  = label
        )
        self.qubits[it].add_gate(obj)
        self.fill_other_qubits([it])
        return self

    # add an arbitrary unitary gate that acts on several qubits
    def add_UMT(self, U, ids_tq, label): 
        for iq in ids_tq:
            obj = Gate.UnitaryMT(
                id_qubit = iq, 
                n_qubits = self.nq,
                U      = U,
                ids_tq = ids_tq,
                label  = label
            )
            self.qubits[iq].add_gate(obj)

        self.fill_other_qubits(ids_tq)
        return self

    # get matrix representation of the circuit:
    def get_matrix(self):
        return self.qu_most.get_circuit_matrix()

    # get general information about the circuit:
    def get_info(self):
        ll = ""

        ll+= "n_qubits: {:d}".format(self.nq) + "\n"

        ng = 0
        n_cnot, n_cnot_nonlocal = 0, 0
        for iq, qq in enumerate(self.qubits): 
            n1 = qq.get_n_gates()
            n_cnot_1, n_cnot_nonlocal_1 = qq.get_cnot_n()
            ng += n1
            n_cnot += n_cnot_1
            n_cnot_nonlocal += n_cnot_nonlocal_1

            ll += "q_{:d}: n_gates = {:0.2f}, n_cnot = {:d}, n_cnot_nonlocal = {:d}"\
                .format(iq, n1, n_cnot_1, n_cnot_nonlocal_1) + "\n"

        ll += "Circuit: n_gates = {:0.1f}, n_cnot = {:d}, n_cnot_nonlocal = {:d}"\
                .format(ng, n_cnot, n_cnot_nonlocal)

        return ll

    # get info about number of all gates in the circuit including the empty ones:
    def get_info_number_all_gates(self):
        ll, ng = "", 0
        for iq, qq in enumerate(self.qubits): 
            n1 = qq.get_n_all_gates()
            ng += n1
            ll += "q_{:d}: n_gates = {:0.1f}".format(iq, n1) + "\n"
        ll += "Circuit: ALL gates: n_gates = {:0.1f}".format(ng)
        return ll

    # count all gates in the circuit:
    def get_n_all_gates(self):
        ng = 0
        for iq, qq in enumerate(self.qubits): 
            ng += qq.get_n_all_gates()
        return ng

    # get special info:
    def get_sp_info(self):
        ll = ''
        nu, n_mcr = 0, 0
        for iq, qq in enumerate(self.qubits):
            n_mcr += qq.get_n_mcr()
            nu += qq.get_n_ust()
        
        ll += "Circuit: nu = {:d}, n_mcr = {:d}".format(nu, n_mcr)

        return ll

    # form a circuit from a sum of Pauli products
    def form_from_pauli(self, H_decomp, dt, sel_error_order = 2):
        # INPUT:
        # -> H_decomp - output data from lib.mix.hermitian_to_pauli.
        # -> dt - time step.
        # -> sel_error_order - order of the Trotterization error (defines QC construction)
        # REMARK: id_pauli: 0 - I, 1 - X, 2 - Y, 3 - Z.

        def copy_qu(id_qubit, ilq, id_pauli):
            if id_qubit != ilq and id_pauli != 0:
                self.add_cnot(id_qubit, ilq)

        def change_basis(ids_prod, ilq, flag_back):
            for id_qubit, id_pauli in enumerate(ids_prod):
                if flag_back:
                    id_qubit = ilq - id_qubit

                # X-Pauli matrix: X = H Z H
                if id_pauli == 1:
                    self.add_H(id_qubit)

                # Y-Pauli matrix: Y = S.H X S
                if id_pauli == 2:
                    if not flag_back:
                        self.add_S(id_qubit, "sd")
                        self.add_H(id_qubit)
                    else:
                        self.add_H(id_qubit)
                        self.add_S(id_qubit)

        def copy_one_dir(ids_prod, ilq, flag_back = False):
            if not flag_back:
                change_basis(ids_prod, ilq, flag_back)
                for id_qubit, id_pauli in enumerate(ids_prod):
                    copy_qu(id_qubit, ilq, id_pauli)
            else:
                for id_qubit, id_pauli in enumerate(ids_prod):
                    id_qubit = ilq - id_qubit
                    copy_qu(id_qubit, ilq, id_pauli)
                change_basis(ids_prod, ilq, flag_back)

        def add_one_prod(dd, ilq_init, coef_theta):
            # indices of Pauli matrices present in the considered tensor product
            ids_prod_init = dd[2]

            nI = ids_prod_init.count(0)
            if nI == nq:
                return

            # find least significant qubit on which acts non-trivial matrix:
            ilq_local = ilq_init
            while ids_prod_init[ilq_local] == 0:
                ilq_local -= 1
            ids_prod_local = ids_prod_init[0:ilq_local+1]

            th = coef_theta * dd[0]

            # copy qubits to the least significant one:
            copy_one_dir(ids_prod_local, ilq_local)

            # Z-rotation of the least significant qubit:
            self.add_R('rz', th, ilq_local)

            # copy qubits back to their original positions:
            ids_prod_rev = list(ids_prod_local)
            ids_prod_rev.reverse()
            copy_one_dir(ids_prod_rev, ilq_local, flag_back = True)

        # number of products:
        Nprod = len(H_decomp)

        # number of Pauli matrices in every product is
        nq = np.int(np.log2(np.shape(H_decomp[0][1])[0]))
        
        # index of the least significant qubit:
        ilq_init = nq - 1

        if sel_error_order == 2:
            coef_theta = 1.
            Nprod_work = Nprod
        if sel_error_order == 3:
            coef_theta = 0.5
            Nprod_work = Nprod - 1
        coef_theta = coef_theta * 2 * dt

        for i_dd in range(Nprod_work):
            dd = H_decomp[i_dd]
            add_one_prod(dd, ilq_init, coef_theta)

        if sel_error_order == 3:
            dd = H_decomp[-1]
            add_one_prod(dd, ilq_init, 2*coef_theta)
            for i_dd in range(Nprod_work-1, -1, -1):
                dd = H_decomp[i_dd]
                add_one_prod(dd, ilq_init, coef_theta)


# ------------------------------------------------
# --- Qubit ---
class Qubit:
    # number of qubits
    nq = None

    # qubit position in QC
    id_qubit = None # 0 means the most significant qubit

    # True if it is the most significant qubit
    flag_most = False

    # upper and lower qubits:
    upper_qu, lower_qu = None, None

    # first gate that acts on the qubit
    start_gate = None

    # last added gate:
    last_gate = None

    # previously added gate:
    # !!! be carefull with this parameter during the decomposition !!!
    prev_gate = None

    # initial state:
    init_state = None  # 0 - [1,0], 1 - [0,1]

    def __init__(self, id, n_qubits, flag_most=False, upper_qu=None):    
        self.nq = n_qubits
        if id >= self.nq or id < 0:
            print('--- Error: wrong qubit index ---')
            print('id_qubit is ', id)
            print('n of qubits is ', self.nq)
            sys.exit(-1)

        self.id_qubit = id
        self.flag_most = flag_most

        self.upper_qu = upper_qu
        if self.upper_qu is not None:
            self.upper_qu.lower_qu = self

        self.start_gate = Gate.EmptyGate(self.id_qubit, self.nq)
        self.last_gate = Gate.EmptyGate(self.id_qubit, self.nq)
        self.start_gate.set_right_nei(self.last_gate)

    def set_boundaries(self):
        self.__set_boundary(False)
        self.establish_vertical_connections_between_gates()

        self.__set_boundary(True)
        self.establish_vertical_connections_between_gates()

    def __set_boundary(self, flag_left):
        self.prev_gate = self.start_gate if flag_left else self.last_gate
        if self.lower_qu is not None:
            self.lower_qu.__set_boundary(flag_left)
   
    def establish_vertical_connections_between_gates(self):
        if self.lower_qu is not None:   
            self.prev_gate.set_lower_nei(self.lower_qu.prev_gate)
            self.lower_qu.establish_vertical_connections_between_gates()

    def set_init_state(self, q_state):
        if q_state == 0:
            self.init_state = [1, 0]
        if q_state == 1:
            self.init_state = [0, 1]

    def reset_prev_gate(self):
        self.prev_gate = self.last_gate.left_gate
        if self.lower_qu is not None:
            self.lower_qu.reset_prev_gate()

    def set_prev_gate(self, gate):
        self.prev_gate = gate.get_iqubit_gate(self.id_qubit)
        if self.lower_qu is not None:
            self.lower_qu.set_prev_gate(gate)

    def add_gate(self, obj_gate):
        self.prev_gate.set_right_nei(obj_gate)
        self.prev_gate = obj_gate

    def text_output(self):
        oline = "Qubit{:3d}:   ".format(self.id_qubit)

        count_gate = 0
        oline = oline + self.start_gate.text_output(self.id_qubit, count_gate)
        return oline

    def reset_qiskit_flags(self):
        self.start_gate.reset_qiskit_flags()
        if self.lower_qu is not None:
            self.lower_qu.reset_qiskit_flags()

    def give_gate_to_decompose(self, sel_gate):
        if sel_gate == "UST":
            gate = self.start_gate.give_UST()
        if sel_gate == "nonlocal-CNOT":
            gate = self.start_gate.give_nonlocal_CNOT()
        if sel_gate == "MCR":
            gate = self.start_gate.give_MCR()
        if sel_gate == "UMT":
            gate = self.start_gate.give_UMT()

        if gate is not None:
            return gate, self
        else:
            if self.lower_qu is not None:
                return self.lower_qu.give_gate_to_decompose(sel_gate)
            else:
                return None, None

    def remove_right_nei(self):
        self.prev_gate.right_gate.remove_hor()
        if self.lower_qu is not None:
            self.lower_qu.remove_right_nei()

    def get_circuit_matrix(self):
        if self.flag_most:
            return self.start_gate.get_circuit_matrix()

    def get_n_gates(self):
        ng = self.start_gate.count_it()
        return ng

    def get_n_all_gates(self):
        ng = self.start_gate.count_it_all()
        return ng

    def get_cnot_n(self):
        z = self.start_gate.count_cnot()
        n_cnot = z[0,0]
        n_nonlocal_cnot = z[0,1]
        return n_cnot, n_nonlocal_cnot

    def get_n_mcr(self):
        ng = self.start_gate.count_mcr()
        return ng

    def get_n_ust(self):
        ng = self.start_gate.count_ust()
        return ng


    