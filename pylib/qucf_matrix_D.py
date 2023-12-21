import numpy as np
import importlib as imp
import h5py
import matplotlib.pyplot as plt
import sys
from numba import jit
import cmath
import pylib.mix as mix
import pylib.qucf_structures as qucf_str
import pylib.qucf_oracle as qucf_o

   
def reload():
    mix.reload_module(mix)
    mix.reload_module(qucf_str)
    mix.reload_module(qucf_o)
    return



# ********************************************************************************************
# * SET
# ********************************************************************************************
class Set__:
    rb_ = None
    re_ = None
    _flag_block_set_ = None

    def __init__(self, rb, re):
        self.rb_ = rb
        self.re_ = re
        return

    def get_Ne(self):
        return (self.re_ - self.rb_)
    
    def is_block_set(self):
        return self._flag_block_set_



# ********************************************************************************************
# * ELEMENT - SET
# ********************************************************************************************
class ElementSet__(Set__):
    # here, v_ is a value (complex or real) of the matrix elements that are stored in this set;
    # all these matrix elements have the same value v_;
    v_ = None

    def __init__(self, v, rb, re):
        super().__init__(rb, re)
        self.v_ = v
        self._flag_block_set_ = False
        return
    
    def is_the_same_as_another_grid(self, another_set):
        if not mix.compare_complex_values(self.v_, another_set.v_):
            print("Warning: element-sets have different values: {:10.12e} vs {:10.12e}".format(
                self.v_, another_set.v_
            ))
        return True



# ********************************************************************************************
# * BLOCK - SET
# ********************************************************************************************
class BlockSet__(Set__):
    # the same inner sets (block-sets or element-sets) within 
    # each block from this block-set:
    sets_ = None 

    # an integer identifier that is used to compare the BlockSet 
    # to its counterpart from another SectionsGrid__:
    id_  = None


    def __init__(self, id_set, rb, re):
        super().__init__(rb, re)
        self.id_  = id_set
        self._flag_block_set_ = True
        return
    

    def establish_inner_structure(self, inner_sets):
        self.sets_ = inner_sets
        return
    

    def is_the_same_as_another_grid(self, another_set):
        if self.id_ != another_set.id_:
            return False # different relative positions of block-sets;
        
        N_sets = len(self.sets_)
        if N_sets != len(another_set.sets_):
            return False # different number of sets;
        
        for i_set in range(N_sets):
            if not self.sets_[i_set].is_the_same_as_another_grid(another_set.sets_[i_set]):
                return False # the sets have different inner structure;
        return True



# ********************************************************************************************
# * BLOCK
# ********************************************************************************************
class Block__:
    circ_ = None

    # [block_1, block_2, ...];
    # block_i can be None (means empty);
    blocks_ = None

    # [set_1, set_2, ...];
    # set_i is a ElementSet__ or BlockSet__;
    # set_i cannot be None;
    sets_ = None

    # the number of elements in the block:
    Ne_ = None

    # the number of sub-blocks in the block:
    Nb_ = None

    # if True, the block does not contain sub-blocks but elements:
    flag_smallest_ = None

    # the very first block in the nested structure is the whole section:
    def __init__(self, circ, block_elements, sizes_blocks):
        self.circ_ = circ
        if len(sizes_blocks) == 1:
            self.flag_smallest_ = True
            self.Ne_ = sizes_blocks[0]
            self.create_element_sets(block_elements)
        else:
            self.flag_smallest_ = False
            self.Nb_ = sizes_blocks[0]

            # first of all, the structure of inner (the smallest) blocks is created,
            # then one moves to the larger blocks: 
            self.create_blocks(block_elements, sizes_blocks[1:])
            self.create_block_sets()
        return
    

    def create_element_sets(self, block_elements):
        Ne = len(block_elements)

        # initialize the sets:
        N_sets = 0
        self.sets_ = [None] * Ne

        # a counter to iterate over each non-nan row in the section;
        ir = np.where(np.isnan(block_elements) == False)[0][0]

        # Each group is associated with a single value:
        vv = block_elements[ir]
        while ir < Ne:
            ir_start = ir
            curr_vv = vv
            while mix.compare_complex_values(vv, curr_vv, self.circ_.prec_): # takes into account nan values; 
                ir += 1
                if ir == Ne:
                    break
                curr_vv = block_elements[ir]

            # save the set:
            N_sets += 1
            self.sets_[N_sets-1] = ElementSet__(vv, ir_start, ir)

            # next value:
            vv = curr_vv
            while np.isnan(vv):
                ir += 1
                if ir >= Ne:
                    break
                vv = block_elements[ir] 
        # ---
        self.sets_ = self.sets_[:N_sets]
        return
    

    def create_blocks(self, block_elements, sizes_bs):
        Ne = len(block_elements)
        subblock_size = Ne//self.Nb_
        self.blocks_ = [None] * self.Nb_
    
        i_begin = 0
        i_end   = subblock_size
        for i_block in range(self.Nb_):
            subblock_elements = block_elements[i_begin:i_end]
            if not all(np.isnan(subblock_elements)): 
                self.blocks_[i_block] = Block__(self.circ_, subblock_elements, sizes_bs)
            i_begin += subblock_size
            i_end   += subblock_size
        return
    

    def create_block_sets(self):
        N_sets = 0
        self.sets_ = [None] * self.Nb_
        irb = next(ii for ii, oo in enumerate(self.blocks_) if oo is not None)

        # block-sets have unique identifiers starting from 0: 0, 1, 2, ...
        # this identifier can be used only to compare sub-blocks within a single block, but
        # it cannot be used to compare a sub-block of one block to a sub-block of another block;
        # it can also be used to compare a block with its counterpart from another grid:
        id_block = -1  
        while irb < self.Nb_:
            id_block += 1
            irb_start = irb
            while True: 
                irb += 1
                if irb == self.Nb_:
                    break
                if self.blocks_[irb] is None:
                    break
                if self.blocks_[irb].is_not_the_same_as(self.blocks_[irb_start]):
                    break

            # save the block-set:
            oo = BlockSet__(id_block, irb_start, irb)
            oo.establish_inner_structure(self.blocks_[irb_start].sets_)

            N_sets += 1
            self.sets_[N_sets-1] = oo

            if irb == self.Nb_:
                break

            # next value:
            while self.blocks_[irb] is None:
                irb += 1
                if irb >= self.Nb_:
                    break
        # ---
        self.sets_= self.sets_[:N_sets]
        return

    
    def is_not_the_same_as(self, another_block):
        if self.flag_smallest_:
            # compare alls sets of elements in the block:
            N_sets = len(self.sets_)
            if N_sets != len(another_block.sets_):
                return True # different number of sets;
        
            for i_set in range(N_sets):
                a_set = self.sets_[i_set]
                b_set = another_block.sets_[i_set]
                if a_set.get_Ne() != b_set.get_Ne():
                    return True # different number of elements in one of the sets;
                if not mix.compare_complex_values(a_set.v_, b_set.v_):
                    return True # different values;
        else:
            # compare all sets in all sub-blocks within this block:
            irs_a = [ii for ii, oo in enumerate(self.blocks_) if oo is not None]
            irs_b = [ii for ii, oo in enumerate(another_block.blocks_) if oo is not None]
            if len(irs_a) != len(irs_b):
                return True # different number of non-empty sub-blocks;
            
            for ii in range(len(irs_a)):
                ir_a = irs_a[ii]
                ir_b = irs_b[ii]
                if ir_a != ir_b:
                    return True # different positions of non-empty blocks;
                if self.blocks_[ir_a].is_not_the_same_as(another_block.blocks_[ir_b]):
                    return True # different structure of non-empty blocks;
        return False # the blocks are the same;
    

    # another_block is a block from another grid:
    def is_the_same_as_another_grid(self, another_block):
        N_sets = len(self.sets_)
        if N_sets != len(another_block.sets_):
            return False # different number of sets;
        if self.flag_smallest_ != another_block.flag_smallest_:
            return False # different nesting of blocks;
        
        for i_set in range(N_sets):
            if not self.sets_[i_set].is_the_same_as_another_grid(another_block.sets_[i_set]):
                return False # different inner structure of sets of the block; 
        return True # two blocks have the same structure;



# ********************************************************************************************
# * Store a matrix in the form of a grid of sections.
# * Each section consists of nested blocks.
# * Each smallest block consists of sets.
# * Each set consists of neighboring elements of the same value.
# ********************************************************************************************
class SectionsGrid__:
    circ_ = None
    sections_ = None # [Block__, Block__, ...]
    ids_sections_ = None


    # If len(sizes_bs_) == 1, then sizes_bs_[0] is the size of the whole matrix.
    # If len(sizes_bs_) == 2, then sizes_bs_[0] is the number of blocks in the matrix, and 
    #   sizes_bs_[1] is the number of elements in each block.
    # If len(sizes_bs_) == 3, then sizes_bs_[0] is the number of blocks in the matrix, and
    #   sizes_bs_[1] is the number of sub-blocks in each block, and
    #   sizes_bs_[2] is the number of elements in each sub-block. 
    # etc.
    sizes_bs_ = None  

    # data = [circ, A, sizes_bs]
    def __init__(self, data):
        self.circ_     = data[0]
        self.sizes_bs_ = data[2]
        grid_sections  = qucf_o.create_grid_of_sections(self.circ_, data[1])

        self.sections_     = []
        self.ids_sections_ = []

        # note that circ.N_sections_ and size of the results list sections_ are different:
        for i_section in range(self.circ_ .N_sections_):
            section_elements = grid_sections[i_section]
            if all(np.isnan(section_elements)): # empty section
                continue
            self.ids_sections_.append(i_section)
            obj_section = Block__(self.circ_, section_elements, self.sizes_bs_)
            self.sections_.append(obj_section)
        return
    

    def get_N_sections(self):
        return len(self.sections_)


    def is_the_same_as_another_grid(self, another_grid):
        N_sections = len(self.sections_)
        if N_sections != len(another_grid.sections_):
            return False # different number of sections;
        
        for i_section in range(N_sections):
            if self.ids_sections_[i_section] != another_grid.ids_sections_[i_section]:
                return False # different positions of sections;
        
        for i_section in range(N_sections):
            if not self.sections_[i_section].is_the_same_as_another_grid(
                another_grid.sections_[i_section]
            ):
                return False # different inner structure of sections;
        return True # two grids have the same structure;



# ********************************************************************************************
# * Template Set
# ********************************************************************************************
class TemplateSet__:
    rb_ = None
    re_ = None
    __flag_block_set_ = None

    # coeffficients to extrapolate the left boundary of the set:
    # rb_ = alpha_b_ + beta_b_ * N_given:
    alpha_b_ = None
    beta_b_ = None

    # coeffficients to extrapolate the right boundary of the set:
    # re_ = alpha_e_ + beta_e_ * N_given:
    alpha_e_ = None
    beta_e_ = None


    def __init__(self):
        return
    

    def compute_coefs_basic(self, grids_sets, grids_sizes, counter_size):
        # find grids that will be used to compute coefficients:
        set_grid_1 = grids_sets[0]
        size_1 = grids_sizes[0][counter_size]

        i_next = next(ii for ii, oo in enumerate(grids_sizes) if oo[counter_size] != size_1)
        set_grid_2 = grids_sets[i_next]
        size_2 = grids_sizes[i_next][counter_size]
        
        # compute the coefficients:
        diff_size = size_1 - size_2

        self.beta_b_ = ((set_grid_1.rb_ - set_grid_2.rb_) * 1.) / diff_size
        self.beta_e_ = ((set_grid_1.re_ - set_grid_2.re_) * 1.) / diff_size

        self.alpha_b_ = set_grid_1.rb_  - self.beta_b_ * size_1
        self.alpha_e_ = set_grid_1.re_  - self.beta_e_ * size_1
        return
    

    def reconstruct_basic(self, sizes, counter):
        self.rb_ = int(self.alpha_b_ + self.beta_b_ * sizes[counter])
        self.re_ = int(self.alpha_e_ + self.beta_e_ * sizes[counter])
        return 



# ********************************************************************************************
# * Template Element Set
# ********************************************************************************************
class TemplateElementSet__(TemplateSet__):
    v_ = None
    def __init__(self, ref_element_set):
        self.__flag_block_set_ = False
        self.v_ = ref_element_set.v_
        return
    
    def compute_coefs(self, grids_sets, grids_sizes, counter_size=0):
        self.compute_coefs_basic(grids_sets, grids_sizes, counter_size)
        return


    def reconstruct(self, sizes, values_section, irs_section, N_values, r_shift_init=0, counter=0):
        self.reconstruct_basic(sizes, counter)
        for r1 in range(self.rb_, self.re_):
            irs_section[N_values] = r_shift_init + r1
            values_section[N_values] = self.v_
            N_values += 1
        return N_values


# ********************************************************************************************
# * Template Block Set
# ********************************************************************************************
class TemplateBlockSet__(TemplateSet__):
    sets_ = None
    id_   = None


    def __init__(self, ref_block_set):
        self.__flag_block_set_ = True
        self.id_ = ref_block_set.id_
        N_sets = len(ref_block_set.sets_)
        self.sets_ = [None] * N_sets
        for i_set in range(N_sets):
            if ref_block_set.sets_[i_set].is_block_set():
                self.sets_[i_set] = TemplateBlockSet__(ref_block_set.sets_[i_set])
            else:
                self.sets_[i_set] = TemplateElementSet__(ref_block_set.sets_[i_set])
        return
    

    def compute_coefs(self, grids_sets, grids_sizes, counter_size=0):
        self.compute_coefs_basic(grids_sets, grids_sizes, counter_size)

        N_grids = len(grids_sets)
        N_sets  = len(self.sets_)
        counter_size += 1
        for i_set in range(N_sets):
            grids_sets_next = [None] * N_grids
            for i_grid in range(N_grids):
                grids_sets_next[i_grid] = grids_sets[i_grid].sets_[i_set]
            self.sets_[i_set].compute_coefs(grids_sets_next, grids_sizes, counter_size)
        return
    

    def reconstruct(
            self, sizes, values_section, irs_section, N_values, 
            r_shift_init = 0, counter = 0
        ):
        self.reconstruct_basic(sizes, counter)

        counter += 1
        Ne_in_block = np.prod(sizes[counter:])
   
        for ir_block in range(self.rb_, self.re_):
            r_shift = r_shift_init + ir_block * Ne_in_block
            for one_set in self.sets_:
                N_values = one_set.reconstruct(
                    sizes, values_section, irs_section, N_values,
                    r_shift_init = r_shift, 
                    counter = counter
                )

        return N_values


# ********************************************************************************************
# * Template Section
# ********************************************************************************************
class TemplateBlock__(Block__):
    def __init__(self, ref_block):
        self.flag_smallest_ = ref_block.flag_smallest_
        N_sets = len(ref_block.sets_)
        self.sets_ = [None] * N_sets
        if self.flag_smallest_:
            for i_set in range(N_sets):
                self.sets_[i_set] = TemplateElementSet__(ref_block.sets_[i_set])
        else:
            for i_set in range(N_sets):
                self.sets_[i_set] = TemplateBlockSet__(ref_block.sets_[i_set])
        return
    

    def compute_coefs(self, grids_blocks, grids_sizes):
        N_grids = len(grids_blocks)
        N_sets  = len(self.sets_)
        for i_set in range(N_sets):
            grids_sets = [None] * N_grids
            for i_grid in range(N_grids):
                grids_sets[i_grid] = grids_blocks[i_grid].sets_[i_set]
            self.sets_[i_set].compute_coefs(grids_sets, grids_sizes)
        return
    

    def reconstruct(self, N, sizes):
        values_section = np.zeros(N, dtype = complex)
        irs_section    = np.zeros(N, dtype = int)

        N_values = 0
        for one_set in self.sets_:
            N_values = one_set.reconstruct(sizes, values_section, irs_section, N_values)

        values_section = values_section[:N_values]
        irs_section = irs_section[:N_values]
        return values_section, irs_section



# ********************************************************************************************
# * Template Grid
# ********************************************************************************************
class TemplateGrid__(SectionsGrid__):
    # coefficients for the reconstruction of the inner structure of the blocks:
    coefs_ = None

    def __init__(self, ref_grid):
        self.ids_sections_ = list(ref_grid.ids_sections_)
        N_sections = len(ref_grid.sections_)
        self.sections_ = [None] * N_sections
        for i_section in range(N_sections):
            self.sections_[i_section] = TemplateBlock__(ref_grid.sections_[i_section])
        return
    

    def compute_coefs(self, grids):
        N_grids = len(grids)
        grids_sizes = [None] * N_grids
        for i_grid in range(N_grids):
            grids_sizes[i_grid] = grids[i_grid].sizes_bs_

        for i_section in range(len(self.sections_)):
            grids_sections = [None] * N_grids
            for i_grid in range(N_grids):
                grids_sections[i_grid] = grids[i_grid].sections_[i_section]
            self.sections_[i_section].compute_coefs(grids_sections, grids_sizes)
        return
    

    def reconstruct_matrix(self, circ):
        # Reconstruct a sparse matrix (with N rows) with Nnz nonzero values:
        # ---
        # > values: Nnz nonzero values in the row-major format:
        # all nonzero elements in the first row, 
        # then all nz elements in the second row, and so on.
        # > columns: Nnz columns of the nonzero values.
        # > rows: of size [N+1]: 
        # rows[i] indicates where the i-th row starts in the array "values";
        # rows[N] = Nnz.
        N = 1 << circ.input_regs_.nq_
        N_sections = len(self.sections_) # nonsparsity
        Nnz = 0
        D_sections_columns = np.zeros((N_sections, N), dtype=int);     D_sections_columns.fill(np.nan)
        D_sections_values  = np.zeros((N_sections, N), dtype=complex); D_sections_values.fill(np.nan)

        N_regs = len(circ.input_regs_.names_)
        sizes = np.zeros(N_regs, dtype=int)
        for i_reg in range(N_regs): # starting from the most-significant qubit:
            sizes[i_reg] = 1 << circ.input_regs_.nqs_[i_reg]

        for i_section in range(N_sections):
            one_section  = self.sections_[i_section]
            id_section   = self.ids_sections_[i_section]
            anc_integers = circ.get_anc_integers_from_section_index(id_section)  
            section_values, irs = one_section.reconstruct(N, sizes)
            for ii, ir in enumerate(irs):
                # find the column index:
                integers_input_regs = circ.compute_integers_in_input_registers(ir)
                ic = circ.get_column_index_from_anc_integers(anc_integers, integers_input_regs)

                # save the matrix element's value:
                Nnz += 1
                D_sections_columns[i_section, ir] = ic
                D_sections_values[i_section, ir]  = section_values[ii]

        # store in the row-major format:
        D = mix.construct_sparse_from_sections(
            N, Nnz, N_sections, D_sections_columns, D_sections_values, circ.prec_
        )
        return D



# ********************************************************************************************
# * Extrapolation procedure * 
# ********************************************************************************************
class Extrapolation__:
    # [SectionsGrid__, SectionsGrid__, ...]
    grids_ = None

    # extrapolation coefficients and values 
    # to reconstruct a matrix of a large size:
    template_ = None


    def __init__(self, grids):
        self.grids_ = grids
        self.check_grids()
        self.compare_grids()
        return
    

    def check_grids(self):
        if (self.grids_ is None):
            print("Error: no grids have been created.")
            return
        
        N_grids = len(self.grids_)
        if N_grids == 0:
            print("Error: no grids have been created.")
            return
        
        # Check whether all grids have the same nested structure (rough analysis):
        N_sizes = np.zeros(N_grids, dtype=int)
        for i_grid in range(N_grids):
            N_sizes[i_grid] = len(self.grids_[i_grid].sizes_bs_)
        if np.any(N_sizes - N_sizes[0]):
            print("Error: a different number of sizes of grids.")
            sys.exit(-1)
        else:
            print("All grids have the same number of sizes.")
        
        # Check whether the number of sizes is enough:
        if N_grids == (N_sizes[0] + 1):
            print("The correct number of grids is provided.")
        if N_grids < (N_sizes[0] + 1):
            print("Error: not enough grids is provided.")
            return
        if N_grids > (N_sizes[0] + 1):
            print("Error: too much grids is provided.")
            return
        return
    

    def compare_grids(self):
        N_grids = len(self.grids_)
        ref_grid = self.grids_[0]
        for i_grid in range(1, N_grids):
            if not ref_grid.is_the_same_as_another_grid(self.grids_[i_grid]):
                print("Error: grids have different structures.")
                return
        print("Grids have similiar structure.")
        return


    def create_extrapolation_template(self):
        ref_grid = self.grids_[0]
        self.template_ = TemplateGrid__(ref_grid)
        self.template_.compute_coefs(self.grids_)
        return
    

    def reconstruct_matrix(self, circ):
        N_regs= len(circ.input_regs_.names_)
        N_grids = len(self.grids_)
        if N_grids != (N_regs + 1):
            print("An incorrect number of input registers is provided.")
            sys.exit(-1)
        D = self.template_.reconstruct_matrix(circ)
        return D
    
