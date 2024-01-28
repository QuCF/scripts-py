import pylib.mix as mix
import numpy as np
import sys

def reload():
    mix.reload_module(mix)

# -----------------------------------------------------------
# --- Gate ---
class Gate:
    # matrix representation of a gate:
    U = 1

    # text name of the gate:
    name = 'Gate'

    # gates on the left and on the right from the gate:
    left_gate, right_gate = None, None

    # upper (more significan) and lower (less significant) gates:
    upper_gate, lower_gate = None, None

    # index of a qubit on each the gate sits on:
    id_qubit = None

    # number of qubits; qubit indices start from 0:
    nq = None

    # added already to qiskit circuit:
    flag_added_to_qiskit_circuit = False

    # gate weight:
    wg = 1

    def __init__(self, id_qubit, n_qubits):
        self.id_qubit = id_qubit
        self.nq = n_qubits

    def set_right_nei(self, obj_gate):
        if self.right_gate is not None:
            pr_right = self.right_gate
            obj_gate.set_far_right_nei(pr_right)
        self.right_gate = obj_gate

        # obj_gate has to be without left neighbor before 
        #   the establishment of this connection
        obj_gate.left_gate = self

    def set_far_right_nei(self, obj_gate):
        if self.right_gate is None:
            self.right_gate = obj_gate
            obj_gate.left_gate = self
        else:
            self.right_gate.set_far_right_nei(obj_gate)

    def set_lower_nei(self, obj_lower_nei):
        self.lower_gate = obj_lower_nei
        if self.lower_gate is not None:
            self.lower_gate.upper_gate = self

    def get_nei(self, at_id_qubit, flag_left):
        if at_id_qubit == self.id_qubit:
            return self.left_gate if flag_left else self.right_gate
        else:
            next_gate = self.upper_gate \
                if at_id_qubit < self.id_qubit \
                else self.lower_gate
            return next_gate.get_nei(at_id_qubit, flag_left)

    # get a gate in the same column of the current gate but on the iq's qubit
    def get_iqubit_gate(self, iq):
        if iq == self.id_qubit:
            return self

        flag_up = True if iq < self.id_qubit else False
        if flag_up:
            return self.upper_gate.get_iqubit_gate(iq)
        else:
            return self.lower_gate.get_iqubit_gate(iq)

    def text_output(self, idq, count_gate):
        oline = "G{:d}: {:s}".format(count_gate, self.name)
        count_gate = count_gate + 1
        oline = oline + ';  ' + self.right_gate.text_output(idq, count_gate)
        return oline

    def qiskit_add_gate(self, qcr):
        self.flag_added_to_qiskit_circuit = True

        if self.lower_gate is not None:
            self.lower_gate.qiskit_add_gate(qcr)    

        if self.id_qubit == 0:
            if self.right_gate is not None:
                self.right_gate.qiskit_add_gate(qcr)

    def give_nonlocal_CNOT(self):
        obj = None
        if self.right_gate is not None:
            obj = self.right_gate.give_nonlocal_CNOT()
        return obj

    def give_MCR(self):
        if self.is_mcr():
            return self
        
        obj = None
        if self.right_gate is not None:
            obj = self.right_gate.give_MCR()
        return obj

    def give_UST(self):
        if self.is_ust():
            return self

        obj = None
        if self.right_gate is not None:
            obj = self.right_gate.give_UST()
        return obj

    def give_UMT(self):
        if self.is_umt():
            return self

        obj = None
        if self.right_gate is not None:
            obj = self.right_gate.give_UMT()
        return obj

    # to remove means to delete all references to a gate in the circuit;
    # one cannot delete Empty Gates on the circuit boundaries
    def remove(self):
        self.remove_hor()
        self.remove_vert()

    def remove_hor(self):
        self.left_gate.right_gate = self.right_gate
        self.right_gate.left_gate = self.left_gate

    def remove_vert(self):
        if self.lower_gate is not None:
            self.lower_gate.upper_gate = self.upper_gate

        if self.upper_gate is not None:
            self.upper_gate.lower_gate = self.lower_gate

    # remove the whole column, where the current gate is placed
    def remove_column(self):
        top_gate = self.go_top()
        top_gate.__remove_column_down()
        
    def __remove_column_down(self):
        self.remove_hor()
        if self.lower_gate is not None:
            self.lower_gate.__remove_column_down()
            
    # go to the upper gate in the column:
    def go_top(self):
        if self.upper_gate is not None:
            return self.upper_gate.go_top()
        return self

    # reset gate flags to default states
    def reset_qiskit_flags(self):
        self.flag_added_to_qiskit_circuit = False
        if self.right_gate is not None:
            self.right_gate.reset_qiskit_flags()

    def get_circuit_matrix(self):
        U1 = self.get_matrix()

        if self.lower_gate is not None:
            U2 = self.lower_gate.get_circuit_matrix()
        else:
            U2 = 1

        U = np.kron(U1, U2)
        if self.id_qubit == 0:
            if self.right_gate is not None:
                U_right = self.right_gate.get_circuit_matrix()
                U = U_right @ U
        return U

    def get_matrix(self):
        return np.matrix(self.U)

    # count the gate (including empty) and go to the right neighbor:
    def count_it_all(self):
        wg = self.wg
        if self.is_empty() or self.is_cnode():
            wg = 1
        
        n = wg
        if self.right_gate is not None:
            n += self.right_gate.count_it_all() 
        return n

    # count the gate and go to the right neighbor:
    def count_it(self):
        n = self.wg
        if self.right_gate is not None:
            n += self.right_gate.count_it() 
        return n

    # count the gate if it is CNOT and/or nonlocal CNOT
    def count_cnot(self):
        flag_cnot, flag_nonlocal_cnot = self.is_cnot()
        n   = 1 if flag_cnot else 0
        nnl = 1 if flag_nonlocal_cnot else 0
        z = np.matrix([n, nnl])

        if self.right_gate is not None:
            z += self.right_gate.count_cnot() 
        return z

    # count number of MCR gates:
    def count_mcr(self):
        n = 1 if self.is_mcr() else 0
        if self.right_gate is not None:
            n += self.right_gate.count_mcr()
        return n

    # count number of UST gates:
    def count_ust(self):
        n = 1 if self.is_ust() else 0
        if self.right_gate is not None:
            n += self.right_gate.count_ust()
        return n

    # whether the gate is CNOT
    def is_cnot(self):
        return False, False

    # whether the gate is MCR
    def is_mcr(self):
        return False

    # whether the gate is UST
    def is_ust(self):
        return False

    # whether the gate is UTM
    def is_umt(self):
        return False

    def is_pauli(self):
        return False

    def is_X(self):
        return False

    def is_Y(self):
        return False

    def is_Z(self):
        return False

    # whether the gate is Hadamard
    def is_H(self):
        return False

    # whether the gate is the Phase gate
    def is_S(self):
        return False
    def is_Sd(self):
        return False

    def is_empty(self):
        return False

    def is_cnode(self):
        return False


# -----------------------------------------------------------
# --- Empty gate ---
# Define boundaries in the circuit
class EmptyGate(Gate):
    def __init__(self, id_qubit, n_qubits):
        super().__init__(id_qubit, n_qubits)
        self.name = 'Empty'
        self.wg = 0

        self.U = np.matrix([
            [1.0, 0.0],
            [0.0, 1.0]
        ])

    def text_output(self, idq, count_gate):
        oline = ""
        if self.right_gate is not None:
            oline = self.right_gate.text_output(idq, count_gate)
        return oline

    def is_empty(self):
        return True

# -----------------------------------------------------------
# --- CNOT gate ----
class CNOT(Gate):
    # node (gate) connected to a control qubit:
    cnode = None

    def set_cnode(self, obj_cnode):
        self.cnode = obj_cnode
        self.name = 'CNOT' + "-C{:d}".format(self.cnode.id_qubit) 

    def qiskit_add_gate(self, qcr):
        if not self.flag_added_to_qiskit_circuit:
            qcr.cx(self.cnode.id_qubit, self.id_qubit)  
            self.cnode.flag_added_to_qiskit_circuit = True
        super().qiskit_add_gate(qcr)

    def get_matrix(self):
        it, ic = self.id_qubit, self.cnode.id_qubit 
        diff = np.abs(it - ic)
        sign = int(diff/(it - ic))

        if diff == 1:
            if sign > 0:  # target is a less significant qubit
                U = np.matrix([
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 0., 1.],
                    [0., 0., 1., 0.]
                ])
            else:
                U = np.matrix([
                    [1., 0., 0., 0.],
                    [0., 0., 0., 1.],
                    [0., 0., 1., 0.],
                    [0., 1., 0., 0.]
                ])
            return U
        else:
            print("--- Error: one has to decompose CNOT gates into local ones ---")
            sys.exit(-1)
            # return None

    def give_nonlocal_CNOT(self):
        _, flag_nonlocal = self.is_cnot()
        return self if flag_nonlocal else self.right_gate.give_nonlocal_CNOT()

    # whether the gate is CNOT
    def is_cnot(self):
        it, ic = self.id_qubit, self.cnode.id_qubit
        diff = np.abs(it - ic)
        flag_nonlocal = True if diff != 1 else False
        return True, flag_nonlocal


# ------------------------------------------------
# --- node of a controlled gate ---
# this node plays a role of an artificial gate on a control qubit
#   to quarantee correct position of gates on every qubit
class CNODE(Gate):
    # gate that is controlled by this Control NODE:
    main_gate = None
    wg = 0

    # type of the control node:
    # '1' means that the main gate is activated than the control qubit is '1'
    type_cnode = ''  # or '1', or '0' 

    def __init__(self, id_qubit, n_qubits, obj_main, type_cnode='1'):
        super().__init__(id_qubit, n_qubits)

        self.main_gate = obj_main
        self.type_cnode = type_cnode

        self.main_gate.set_cnode(self)

        self.name = 'CNODE' + self.type_cnode + '-T{:d}'.format(self.main_gate.id_qubit)

    def qiskit_add_gate(self, qcr):
        if not self.flag_added_to_qiskit_circuit:
            qcr.cx(self.id_qubit, self.main_gate.id_qubit)
            self.main_gate.flag_added_to_qiskit_circuit = True
        super().qiskit_add_gate(qcr)

    def is_cnode(self):
        return True


# ------------------------------------------------
# --- rotation gate around Y-axis ---
class Ry(Gate):
    theta = None

    def __init__(self, id_qubit, n_qubits, theta):
        super().__init__(id_qubit, n_qubits)
        
        self.theta = theta
        # self.name = "Ry({:0.2f})".format(self.theta)
        self.name = "Ry"

        th2 = self.theta/2.
        self.U = np.matrix([
            [np.cos(th2), - np.sin(th2)],
            [np.sin(th2),   np.cos(th2)]
        ])

    def qiskit_add_gate(self, qcr):
        if not self.flag_added_to_qiskit_circuit:
            qcr.ry(self.theta, self.id_qubit)
        super().qiskit_add_gate(qcr)

# ------------------------------------------------
# --- rotation gate around Z-axis ---
class Rz(Gate):
    theta = None

    def __init__(self, id_qubit, n_qubits, theta):
        super().__init__(id_qubit, n_qubits)

        self.theta = theta
        # self.name = "Rz({:0.2f})".format(self.theta)
        self.name = "Rz"

        th2 = 1j*self.theta/2.
        self.U = np.matrix([
            [np.exp(-th2),         0.0],
            [0.0,          np.exp(th2)]
        ])

    def qiskit_add_gate(self, qcr):
        if not self.flag_added_to_qiskit_circuit:
            qcr.rz(self.theta, self.id_qubit)
        super().qiskit_add_gate(qcr)


# ------------------------------------------------
# --- base multi-controlled rotation ---
class MCR(Gate):
    cnodes = None

    def __init__(self, id_qubit, n_qubits, U):
        super().__init__(id_qubit, n_qubits)
        self.U = np.matrix(U)
        self.cnodes = []

    def set_cnode(self, obj_c):
        self.cnodes.append(obj_c)
        self.name += "-C{:d}".format(obj_c.id_qubit)

    def qiskit_add_gate(self, qcr):
        print("Error: Currently, there is not visualisation for MCR. Use text representation.")
        sys.exit(-1)

    def is_mcr(self):
        return True


# ------------------------------------------------
# --- multi-controlled Ry ---
class MCRY(MCR):
    cnodes = None
    type_mcr = "ry"

    def __init__(self, id_qubit, n_qubits, U):
        super().__init__(id_qubit, n_qubits, U)
        self.name = 'MCRY'


# ------------------------------------------------
# --- multi-controlled Rz ---
class MCRZ(MCR):
    cnodes = None
    type_mcr = "rz"

    def __init__(self, id_qubit, n_qubits, U):
        super().__init__(id_qubit, n_qubits, U)
        self.name = 'MCRZ'


# ----------------------------------------------------------
# --- single-qubit unitary gate ---
class UnitaryST(Gate):
    def __init__(self, id_qubit, n_qubits, U, label):
        super().__init__(id_qubit, n_qubits)
        self.U = np.matrix(U)
        self.name = label

    def qiskit_add_gate(self, qcr):
        print("Error: Currently, there is not visualisation for an arbitrary single-qubit unitary gate.")
        sys.exit(-1)

    def is_ust(self):
        return True


# ----------------------------------------------------------
# --- multi-qubit unitary gate ---
class UnitaryMT(Gate):
    # indices of target qubits
    ids_tq = None

    def __init__(self, id_qubit, n_qubits, U, ids_tq, label):
        super().__init__(id_qubit, n_qubits)
        self.U = np.matrix(U)
        self.ids_tq = ids_tq
        self.name = label

        self.wg = 1. / len(ids_tq)

    def qiskit_add_gate(self, qcr):
        print("Error: Currently, there is not visualisation for an arbitrary multi-qubit unitary gate.")
        sys.exit(-1)

    def is_umt(self):
        return True

# ----------------------------------------------------------
# --- Pauli gate ---
class Pauli(Gate):
    def is_pauli(self):
        return True

    def qiskit_add_gate(self, qcr):
        if not self.flag_added_to_qiskit_circuit:
            if self.is_X():
                qcr.x(self.id_qubit)
            if self.is_Y():
                qcr.y(self.id_qubit)
            if self.is_Z():
                qcr.z(self.id_qubit)
        super().qiskit_add_gate(qcr)

class X(Pauli):
    name = "X"
    U = mix.X

    def is_X(self):
        return True

class Y(Pauli):
    name = "Y"
    U = mix.Y

    def is_Y(self):
        return True

class Z(Pauli):
    name = "Z"
    U = mix.Z

    def is_Z(self):
        return True


# ----------------------------------------------------------
# --- Hadamard gate ---
class H(Gate):
    name = "H"
    U = mix.H

    def is_H(self):
        return True

    def qiskit_add_gate(self, qcr):
        if not self.flag_added_to_qiskit_circuit:
            qcr.h(self.id_qubit)
        super().qiskit_add_gate(qcr)


# ----------------------------------------------------------
# --- Phase gate ---
class S(Gate):
    name = "S"
    U = mix.S

    def is_S(self):
        return True

    def qiskit_add_gate(self, qcr):
        if not self.flag_added_to_qiskit_circuit:
            qcr.s(self.id_qubit)
        super().qiskit_add_gate(qcr)

class Sd(Gate):
    name = "Sd"

    def __init__(self, id_qubit, n_qubits):
        super().__init__(id_qubit, n_qubits)
        self.U = np.matrix([
            [1.,   0.],
            [0., -1.j]
        ])

    def is_Sd(self):
        return True

    def qiskit_add_gate(self, qcr):
        if not self.flag_added_to_qiskit_circuit:
            qcr.sdg(self.id_qubit)
        super().qiskit_add_gate(qcr)
        