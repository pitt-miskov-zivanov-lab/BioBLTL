"""
author: Difei Tang
email: DIT18@pitt.edu
MeLoDy Lab, University of Pittsburgh

The script of a BLTL generator 

NOTE: 
Final release: Ensure sufficient coverage of diverse BLTL patterns
(e.g., sequential nested and combination) by increasing the probs for further model finetuning; 
Publication: Set the prob as 0.5 to retain the randomness in the generation process.
"""

from copy import deepcopy
from operator import is_
import stat
from attr import has
from transitions import Machine
from pathlib import Path
from typing import Optional, List, Dict, Literal, Union, Tuple

import random
from enum import Enum
import numpy as np  
import json 
import os 
import re

from sympy import N

# invalid species name examples: 
# '3MBz', 'G(i/o)', 'Fox1.2', 'Ara+', 'Ca2+', 'IL-4R', 'exogen_BMP2_I'

def urlify(s):
    # Keep forward slashes, remove other invalid characters
    if s[-1] == 'R': # IL-4R -> IL4R
        s = re.sub(r'\-', '', s)
    elif s[-1] == '+':
        return None
    else:
        s = re.sub(r"[^\w\s/]", '_', s)  
    s = re.sub(r"\s+",'_',s)

    if s.count('_') >= 2:
        return None 

    return s

def check_variables(vars):
    filter_vars = []
    for var in vars:
        if not var[0].isdigit():
            _var = urlify(var)
            if _var:
                filter_vars.append(_var)

    return filter_vars

class Type(Enum):
    F = "F",
    T = "T",
    B = "B",
    D_LR = "D_LR",
    C_LR = "C_LR",
    G_RF = "G_RF",
    G_R = "G_R",
    G_F = "G_F",
    F_RF = "F_RF",
    F_R = "F_R",
    F_F = "F_F",
    F_G = "F_G",
    U_LRF = "U_LRF",
    U_LR = "U_LR",
    U_LF = "U_LF",
    U_RF = "U_RF",
    U_L = "U_L",
    U_R = "U_R",
    U_F = "U_F"

class AROperator(Enum):
    # arithmetic operator
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"

    NOOP = "NOOP"

# NOOP has two meanings
# 1. in a Block, a logical (non-base) node
# 2. in a Arithmetic, it's a real variable
#     or a real value

class REOperator(Enum):
    # relational operator denoting a boolean proposition
    EQ = "=="
    #NEQ = "!=" parser will throw an error if we use this
    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="

    #NOOP = "NOOP"
    
class Arithmetic:
    def __init__(self):
        self.parent = None
        self.op: AROperator = AROperator.NOOP      
        self.var: str = ""       
        self.val: float = 0.0      

        self.L: Optional[Arithmetic] = None    
        self.R: Optional[Arithmetic] = None  
    
    # @property
    # def var(self):
    #     return self._var

class SequenceContext():
    '''A help class for implementing sequential behaviors of repeated varibles in BLTL generation'''
    states = ['ON', 'OFF']

    def __init__(self) -> None:

        self.state = Machine(model=self, states=SequenceContext.states, initial='OFF') 
        self.allow_nested = True

        self.state.add_transition('turn_on', 'OFF', 'ON')
        self.state.add_transition('turn_off', 'ON', 'OFF')

    def reset(self): 
        self.state = 'OFF'
        self.allow_nested = True

class Block:
    def __init__(self):
        self.parent: Optional[Block] = None
        # it can be a relational operator denoting a boolean proposition or NOOP denoting a logical operator
        # NOOP denoting a logic operator (or T/F)
        # 
        # it can't be an arithmetic operator!
        self.op = None      

        self.t = None       
        self.bound = 0.0  

        self._LB: Optional[Block] = None     
        self._RB: Optional[Block] = None     
        self._F: Optional[Block] = None # inner block 
        self._G: Optional[Block] = None      

        self._LA: Optional[Arithmetic] = None    
        self._RA: Optional[Arithmetic] = None   

    @property
    def LB(self):
        return self._LB
    
    @LB.setter
    def LB(self, block):
        self._LB = block
        if block: 
            block.parent = self
    
    @property
    def RB(self):
        return self._RB
    
    @RB.setter
    def RB(self, block):
        self._RB = block
        if block: 
            block.parent = self
    
    @property
    def F(self):
        return self._F
    
    @F.setter
    def F(self, block):
        self._F = block
        if block: 
            block.parent = self
    
    @property
    def G(self):
        return self._G
    
    @G.setter
    def G(self, block):
        self._G = block
        if block: 
            block.parent = self
    
    @property
    def LA(self):
        return self._LA
    
    @LA.setter
    def LA(self, arithmetic):
        self._LA = arithmetic
        if arithmetic: 
            arithmetic.parent = self
    
    @property
    def RA(self):
        return self._RA
    
    @RA.setter
    def RA(self, arithmetic):
        self._RA = arithmetic
        if arithmetic: 
            arithmetic.parent = self

    def print(self, indent: int = 0) -> None:
        """Print the Block as a simple tree structure"""
        indent_str = "  " * indent
        
        # Print current node
        node_type = self.t.name if hasattr(self, 't') and self.t else "Unknown"
        print(f"{indent_str}Block({node_type})")
        
        # Print children
        if self.LB:
            print(f"{indent_str}├─ LB:")
            self.LB.print(indent + 1)
        if self.RB:
            print(f"{indent_str}├─ RB:")
            self.RB.print(indent + 1)
        if self.F:
            print(f"{indent_str}├─ F:")
            self.F.print(indent + 1)
        if self.G:
            print(f"{indent_str}├─ G:")
            self.G.print(indent + 1)
        if self.LA:
            print(f"{indent_str}├─ LA: Arithmetic")
        if self.RA:
            print(f"{indent_str}└─ RA: Arithmetic")
        
def fix_atomic(block: Block, min_value: float, max_value: float) -> Block:
    # if A > max_value or A < min_value, we need to change the value or flip the operator
    flip_map = {
        REOperator.LT: REOperator.GT,
        REOperator.GT: REOperator.LT,
        REOperator.LTE: REOperator.GTE,
        REOperator.GTE: REOperator.LTE,
    }

    prob = random.random()

    if block.op in [REOperator.GT, REOperator.GTE] and block.RA.val == max_value:
        if prob < 0.5:
            block.RA.val = min_value 
        else:
            block.op = flip_map[block.op]
    elif block.op in [REOperator.LT, REOperator.LTE] and block.RA.val == min_value:
        if prob < 0.5:
            block.RA.val = max_value
        else:
            block.op = flip_map[block.op]

    return block

class BLTLGenerator:
    """BLTL generator
    For the BLTL time bounded versions, more help is at Model Checking and BLTL Syntax
    """
    # FIXME: the weights of the strategies and types are manually set

    def __init__(self, biomodels_kb: list=None, mode: str=None):
        self.biomodels_kb = biomodels_kb

        self.curr_model_index = 0
        self.curr_model = self.biomodels_kb[self.curr_model_index]

        self.variables = check_variables(self.curr_model['species'])
        self.avail_variables = self.variables.copy()
        
        self.min_bound = 1.0
        self.max_bound = 100.0
        self.min_value = 0.0
        self.max_value = 1.0

        # constraints from framework2.py
        self.MAX_NEGATION = 1
        self.MAX_UNTIL = 1
        self.MAX_TEMPORAL = 10

        self.temporal_count = 0
        self.negation_count = 0
        self.until_count = 0

        # probs during generation 
        self.connected_tree_prob = 0.5
        self.c_lr_prob = 0.8
        self.d_lr_prob = 0.2
        self.seq_prob = 0.5 # FIXME: adjust for different purposes 

        # we use mode to control: 1) bio or non-bio 2) qual or quan
        self.mode = mode
        self.option = 'qual'

        # sequential behavior
        self.generated_vars = dict()
        self.seq_context = SequenceContext()
        self.sequential_vars = dict() # {var: {bound: t, atomic: == 0}}

        self.has_seq = False
        self.number_combination_seq = 0
        self.number_nested_seq = 0
        self.on_until = False
        self.on_combination = False
        self.on_combination_seq = False

        self.repeated_var = None

    def generate_variable(self) -> str:
        if not self.avail_variables:
            raise ValueError('No more variables available')

        var = random.choice(self.avail_variables)
        self.avail_variables.remove(var)

        return var
    
    def reset(self):
        self.avail_variables = self.variables.copy()

        self.seq_context.reset()
        self.sequential_vars = dict()
        self.repeated_var = None
        self.generated_vars = dict()
        self.has_seq = False
        self.number_combination_seq = 0
        self.number_nested_seq = 0
        self.on_until = False
        self.on_combination = False
        self.on_combination_seq = False

    def reset_counters(self):
        self.temporal_count = 0
        self.negation_count = 0
        self.until_count = 0

    def update_model_index(self):
        """By default, we just update the model index by 1 and reset if the index is greater than the model number"""

        self.curr_model_index = (self.curr_model_index + 1) % len(self.biomodels_kb)

        self.curr_model = self.biomodels_kb[self.curr_model_index]
        self.variables = check_variables(self.curr_model['species'])
        self.avail_variables = self.variables.copy()

    def is_valid_formula(self, block: Block) -> bool:
        """Check if the generated formula satisfies all constraints
        """

        # Reset counters before checking
        self.reset_counters()
        
        def check_block(b: Block) -> bool:
            if b is None:
                return True
                
            # Count temporal operators
            if b.t in [Type.F_R, Type.G_R, Type.U_LR, Type.F_G, Type.G_F, Type.F_F]:
                self.temporal_count += 1
                if self.temporal_count > self.MAX_TEMPORAL:
                    return False
                    
            # Count until operators
            if b.t == Type.U_LR:
                self.until_count += 1
                if self.until_count > self.MAX_UNTIL:
                    return False
                
            # FIXME: count negation operators??

            # Check children
            return (check_block(b.LB) and 
                    check_block(b.RB) and 
                    check_block(b.F) and 
                    check_block(b.G))
        
        return check_block(block)
    
    def _get_var_bound_in_atomic(self, block: Block, var: str, bound: int) -> int:
        """Get the bound of a variable in an atomic proposition"""
        curr_parent = block.parent

        if curr_parent.t in [Type.C_LR, Type.D_LR] and block.t == Type.B:
            bound = self._get_var_bound_in_atomic(curr_parent, var, bound) 
        
        bound = self._get_var_bound(block=curr_parent, bound=bound)
        return bound 

    def _get_var_bound(self, block: Block, bound: int) -> int:
        if hasattr(block, 'bound'):
            bound += block.bound

        if hasattr(block, 'parent') and block.parent:  
            if block.parent.t == Type.F_G:
                bound += block.parent.bound

        return bound
    
    def _get_atomic_blocks(self, block: Block) -> List[Block]:
        """Get atomic blocks within a temporal block"""
        assert block != None
        assert block.t != Type.B 

        # FIXME: For patterns like F_G, we need to add the bound of the parent block
        bound = 0
        if hasattr(block.parent, 't'):
            if block.parent.t == Type.F_G:
                bound += block.parent.bound
        
        # now we enter into F, G, U 
        curr_vars = set()
        self._find_atomic(block=block, bound=bound, vars=curr_vars)
        return curr_vars
    
    def _find_atomic(self, block: Block, bound: int, vars: set) -> Optional[Block]:
        if hasattr(block, 'bound'):
            bound += block.bound

        if block.t in [Type.C_LR, Type.D_LR, Type.U_LR]:
            self._find_atomic(block.LB, bound, vars)
            self._find_atomic(block.RB, bound, vars)
        
        if block.F:
            return self._find_atomic(block.F, bound, vars)

        if block.t == Type.B:
            # NOTE: track states of each generated atomic 
            var = block.LA.var 
            val = block.RA.val

            vars.add(var)

            if var not in self.generated_vars:
                self.generated_vars[var] = {
                    'bound': bound,
                    'value': val
                } 
            else:
                var_state = self.generated_vars[var] 
                var_state['bound'] += bound 

    def generate_bltl(self) -> Block:
        """generate a BLTL formula
        ensure the temporal operator is the outermost operator"""

        max_attempts = 10
        attempts = 0
        
        while attempts < max_attempts:
            self.reset()
            self.reset_counters()

            try:
                logic_connected_type = random.choices(
                    [Type.C_LR, Type.D_LR],
                    weights=[self.c_lr_prob, self.d_lr_prob]  
                )[0]
                
                block = (self.generate_logic_connected_tree(logic_connected_type) 
                        if random.random() < self.connected_tree_prob 
                        else self.generate_single_tree())

                if self.is_valid_formula(block):

                    if self.number_nested_seq == 0 and self.number_combination_seq == 0:
                        self.reset() 

                    # record state changed elements 
                    # seq_vars = self.sequential_vars.keys() 

                    # if len(seq_vars) > 0:
                    #     bltl_str = self.block_to_string(block)

                    #     is_seq = False
                    #     for var in seq_vars: 
                    #         if bltl_str.count(var) > 1:
                    #             is_seq = True
                    #             break 

                    #     if not is_seq:
                    #         self.reset()

                    return block
            except Exception as e:
                    import traceback
                    traceback.print_exc() 
                    print(f'Failed generating BLTL: {e}')
                
            attempts += 1

        raise Exception("Failed generating valid BLTL formula after maximum attempts")
        
    def generate_logic_connected_tree(self, logic_connected_type: Type, parent: Block = None) -> Block:
        block = Block()
        if parent:
            block.parent = parent

        block.t = logic_connected_type
        
        block.LB = self.generate_single_tree(parent=block)

        # recursively generate the right child (as logic connected tree)
        if random.random() < self.connected_tree_prob:
            logic_connected_type = random.choices(
                [Type.C_LR, Type.D_LR],
                weights=[self.c_lr_prob, self.d_lr_prob]  
            )[0]

            block.RB = self.generate_logic_connected_tree(logic_connected_type, parent=block)
        else:
            block.RB = self.generate_single_tree(parent=block)

        # TODO: when to stop sequential behavior
        if self.has_seq:
            self.number_nested_seq = 1 
           
        return block

    def generate_single_tree(self, type=None, parent=None) -> Block:
        """generate a single BLTL tree
        ensure the outermost operator is temporal"""
        block = Block()

        if parent is not None:
            block.parent = parent

        # FIXME: we do not consider Until operator in sequential behaviors: 
        # F[10](FOXP3 = 1)U[20](FOXP3 = 0)

        f_r_prob = 0.3 # F[t]
        g_r_prob = 0.3 # G[t]
        u_lr_prob = 0.15 # U[t]
        f_g_prob = 0.25 # F[t](G[t](φ))

        if type is not None: 
            # already has a temporal operator
            block.t = type  
        else:
            if self.mode == "bio":
                if self.seq_context.state == 'ON':
                    f_r_prob = 0.5
                    g_r_prob = 0.0
                    u_lr_prob = 0.0
                    f_g_prob = 0.5

                temporal_types = [
                    (Type.F_R, f_r_prob),    
                    (Type.G_R, g_r_prob),    
                    (Type.U_LR, u_lr_prob),   
                    (Type.F_G, f_g_prob),    
                ] 
            else:
                # FIXME
                temporal_types = [
                    (Type.F_R, 1.0/6),  
                    (Type.G_R, 1.0/6),
                    (Type.G_F, 1.0/6), 
                    (Type.U_LR, 1.0/6),
                    (Type.F_F, 1.0/6),   
                    (Type.F_G, 1.0/6),
                ]
            
            block.t = random.choices(
                [t[0] for t in temporal_types],
                weights=[t[1] for t in temporal_types]
            )[0]
        
        # TODO: sequential behavior 
        # 1) (G[5] (FOXP3 == 0)) & (F[6] (G[6] (FOXP3 == 1))) & (F[13] (G[2] (FOXP3 == 0))) & (F[16] (G[14] (FOXP3 == 1)))
        # 2) F[10] (FOXP3 == 1 & F[20] (FOXP3 == 0))

        if self.seq_context.state == 'ON': 
            # set time bound constraints for repeated var
            _var_state = self.sequential_vars[self.repeated_var]
            _bound = _var_state['bound']
            if block.t in [Type.F_G, Type.F_R]: 
                block.bound = random.randint(_bound, _bound + 10)
                _var_state['bound'] = block.bound # update its time bound 
            elif block.t == Type.G_R: #F[t](G[t](φ))
                block.bound = random.randint(self.min_bound, self.max_bound / 2)
                _var_state['bound'] += block.bound
        else: 
            block.bound = random.randint(self.min_bound, self.max_bound / 2)
    
        assert hasattr(block, 't')

        # generate the sub-structure based on the operator type
        if block.t in [Type.F_G]:
            return self.generate_nested_temporal(block, parent=block)
        else:
            if block.t == Type.U_LR:
                self.on_until = True 
                block.LB = self.generate_inner_formula(block.bound, parent=block)
                block.RB = self.generate_inner_formula(block.bound, parent=block)
                self.on_until = False 
            else:
                block.F = self.generate_inner_formula(block.bound, parent=block)

        # check for sequential behaviors
        # curr_vars = self._get_atomic_blocks(block)

        # if self.seq_context.state == 'OFF':
        #     to_seq_vars = [var for var in curr_vars if var not in self.sequential_vars.keys()]
        #     if len(to_seq_vars) > 0: 
        #         _var = random.choice(to_seq_vars)
                
        #         if random.random() < self.seq_prob and block.t != Type.U_LR:
        #             self.repeated_var = _var
        #             self.sequential_vars[_var] = deepcopy(self.generated_vars[_var])
        #             self.seq_context.turn_on()

        if self.seq_context.state == 'ON': 
            if random.random() < 0.5: # stop sequence generating 
                self.seq_context.turn_off()
                self.repeated_var = None
                #print('turn off the sequence state: ', self.seq_context.state)

        return block
    
    def generate_nested_temporal(self, block: Block, parent: Block = None) -> Block:
        if parent:
            block.parent = parent

        if block.t == Type.F_G:
            inner_block = self.generate_single_tree(Type.G_R, parent=block)
            block.F = inner_block
        elif block.t == Type.G_F:
            inner_block = self.generate_single_tree(Type.F_R, parent=block)
            block.F = inner_block
        elif block.t == Type.F_F:
            inner_block = self.generate_single_tree(Type.F_R, parent=block)
            block.F = inner_block
            
        return block
    
    def generate_inner_formula(self, bound, parent: Block = None) -> Block:
        """generate an inner formula (no temporal operator)"""

        if (self.seq_context.state == 'ON' and 
            self.seq_context.allow_nested and self.on_combination_seq):
            strategies = [
                (self.generate_atomic, 1.0),       
                ]
        else:
            strategies = [
                (self.generate_atomic, 0.5),           
                (self.generate_boolean_combination, 0.5) # TODO: F(atomic 1 & F[atomic2])
            ]
        
        strategy = random.choices(
            [s[0] for s in strategies],
            weights=[s[1] for s in strategies]
        )[0]
        
        return strategy(parent=parent)
    
    def generate_value_op(self):
        """generate the value and operator for an atomic proposition 
        two options: 0, 1 (qualitative) & 0.0 - 1.0 (quantitative)"""

        # generate op 
        weights = {
            0.0: 0.4,  
            1.0: 0.4, 
            'random': 0.2 
        }

        if self.option == 'qual':
            op = REOperator.EQ

            weights = {
                0: 0.5,  
                1: 0.5, 
            }
        else:
            # prefer == and > operators
            op_choices = [(REOperator.EQ, 0.6), (REOperator.GT, 0.15), (REOperator.LT, 0.15), (REOperator.GTE, 0.1), (REOperator.LTE, 0.1)]
        
            op = random.choices(
                [op[0] for op in op_choices],
                weights=[op[1] for op in op_choices]
            )[0]
        
        # generate value 
        random_value = None
        if self.seq_context.state == 'ON' and not self.on_combination:
            # FIXME: only support qual option now 
            _var_state = self.sequential_vars[self.repeated_var]
            _val = _var_state['value']
            random_value = 1 - _val
            self.sequential_vars[self.repeated_var]['value'] = random_value
        else:
            choice = random.choices(list(weights.keys()), list(weights.values()))[0]
            if choice == 'random':
                random_value = round(random.uniform(self.min_value, self.max_value) / 0.1) * 0.1
                random_value = round(random_value, 1)
            else:
                random_value = choice

        assert random_value != None
        return random_value, op 
    
    def generate_atomic(self, var: str = None, parent: Block = None) -> Block:
        """generate an atomic proposition
        we do not consider proposition like A>B here"""

        block = Block()
        if parent:
            block.parent = parent

        if self.seq_context.state == 'ON' and not self.on_combination and not self.on_until: 
            var = self.repeated_var
            self.has_seq = True
        else:
            var = self.generate_variable()

        # TODO: 
        if self.option != 'qual' and random.random() < 0.2:
                block.t = Type.C_LR
                min_val = self.round_to_nearest_tenth(random.uniform(self.min_value, self.max_value - 0.1))
                max_val = self.round_to_nearest_tenth(random.uniform(min_val + 0.1, self.max_value))
                
                block.LB = self.create_atomic_manual(var, REOperator.GT, min_val)
                block.RB = self.create_atomic_manual(var, REOperator.LT, max_val)
                return block

        block.t = Type.B
        block.LA = Arithmetic()
        block.LA.var = var
        block.RA = Arithmetic()
        
        random_value, op = self.generate_value_op()
        block.op = op

        # avoid <0 or >1
        if random_value == 0 or random_value == 0.0:
            if block.op in [REOperator.LT, REOperator.LTE, REOperator.GTE]:
                if random.random() < 0.5:
                    block.op = REOperator.EQ  # == 0
                else:
                    block.op = REOperator.GT  # > 0

        if random_value == 1 or random_value == 1.0:
            if block.op in [REOperator.GT, REOperator.GTE, REOperator.LTE]:
                if random.random() < 0.5:
                    block.op = REOperator.EQ  # == 1
                else:
                    block.op = REOperator.LT  # < 1

        block.RA.val = random_value

        # TODO: complex arithmetic
        # random_value = round(random.uniform(self.min_value, self.max_value) / 0.1) * 0.1
        # random_value = round(random_value, 1)
        # block.RA.val = random_value

        # record & update the generated variable and its value, bound 
        bound = self._get_var_bound_in_atomic(block, var, bound=0)
        if var not in self.generated_vars:
            self.generated_vars[var] = {
                'bound': bound,
                'value': random_value
            } 
        else:
            var_state = self.generated_vars[var] 
            var_state['bound'] += bound 

        if self.seq_context.state == 'OFF' and not self.on_until:
            if var not in self.sequential_vars.keys():
                if random.random() < self.seq_prob:
                    self.repeated_var = var
                    self.sequential_vars[var] = deepcopy(self.generated_vars[var])
                    self.seq_context.turn_on()
        
        return block
    
    def round_to_nearest_tenth(self, value: float) -> float:
        return round(value / 0.1) * 0.1
    
    def create_atomic_manual(self, var: str, op, val: float) -> Block:
        """this allows for manual creating atomic prop block"""
        new_block = Block()
        new_block.t = Type.B
        new_block.LA = Arithmetic()
        new_block.LA.var = var
        new_block.op = op
        new_block.RA = Arithmetic()
        new_block.RA.val = val
        return new_block
    
    def generate_boolean_combination(self, parent: Block = None) -> Block:
        # FIXME: how to handle the negation? 

        """generate a boolean combination
        prefer C_LR (AND) here"""
        block = Block()
        if parent:
            block.parent = parent

        self.on_combination = True

        if (self.seq_context.allow_nested and 
            self.seq_context.state == 'ON'):
            block.t = Type.C_LR
        else:
            logic_op_types = [(Type.C_LR, self.c_lr_prob), (Type.D_LR, self.d_lr_prob)]
            block.t = random.choices(
                [lt[0] for lt in logic_op_types],
                weights=[lt[1] for lt in logic_op_types]
            )[0] 

        # Manually shut down the sequential behavior and re-turn on after the generation 
        # to avoid issues such as (A==0&A==1)

        block.LB = self.generate_atomic(parent=block)
        
        # state change for the 'left' generated atomic variable 
        if (self.seq_context.allow_nested and 
            self.seq_context.state == 'ON' and 
            block.t == Type.C_LR):
            self.on_combination = False 

            self.on_combination_seq = True 
            block.RB = self.generate_single_tree(parent=block)

            self.number_combination_seq = 1
            self.seq_context.allow_nested = False 
        else:
            block.RB = self.generate_atomic(parent=block)

        self.on_combination = False
        return block

    # def generate_random_arithmetic(self, depth: int = 0) -> Arithmetic:
    #     """generate a random arithmetic expression"""
    #     if depth >= self.max_depth:
    #         return self.generate_simple_arithmetic()

    #     arith = Arithmetic()
    #     choice = random.random()

    #     if choice < 0.4:  
    #         return self.generate_simple_arithmetic()
    #     arith.op = random.choice([
    #         AROperator.ADD, AROperator.SUB, 
    #         AROperator.MUL, AROperator.DIV
    #     ])
    #     arith.L = self.generate_random_arithmetic(depth + 1)
    #     arith.R = self.generate_random_arithmetic(depth + 1)
    #     return arith
    
    def _add_constraint(self, block: Block, temporal_info: Dict):
        """add a new constraint as we created a legal atomic prop block"""
        var = block.var
        bound = self._get_time_bound(block)

        if var not in self.var_constraints: 
            self.var_constraints[var] = bound
        else:
            curr_bound = self.var_constraints[var]
            if bound <= curr_bound: 
                # contradition
                pass 
            else:
                # update the time bound 
                self.var_constraints[var] = bound 

    def block_to_string(self, block: Block) -> str:
        """convert the Block tree to a string"""
        # TODO: remove the space between chars 
        if block.t == Type.B:
            if self.option == 'qual':
                atomic_str = f"{block.LA.var}{block.op.value}{block.RA.val}"
                return atomic_str
            atomic_str = f"{block.LA.var}{block.op.value}{block.RA.val:.1f}"
            return atomic_str
            
        if block.t == Type.C_LR:
            left = self.block_to_string(block.LB)
            right = self.block_to_string(block.RB)
            and_str = f"{left}&{right}"
            return and_str
        elif block.t == Type.D_LR:
            left = self.block_to_string(block.LB)
            right = self.block_to_string(block.RB)
            or_str = f"{left}|{right}"
            return or_str
            
        if block.t in [Type.F_R, Type.G_R]:
            inner = self.block_to_string(block.F)
            op = "F" if block.t == Type.F_R else "G"
            temporal_str = f"{op}[{block.bound}]({inner})"
            return temporal_str
            
        if block.t == Type.U_LR:
            left = self.block_to_string(block.LB)
            right = self.block_to_string(block.RB)
            until_str = f"({left})U[{block.bound}]({right})"
            return until_str
        
        # nested
        if block.t in [Type.F_G, Type.F_F]:
            inner = self.block_to_string(block.F)  # G[t](φ)
            nested_str = f"F[{block.bound}]({inner})"
            return nested_str
        
        if block.t == Type.G_F:
            inner = self.block_to_string(block.F)  # F[t](φ)
            nested_str = f"G[{block.bound}]({inner})"
            return nested_str
        
        return ""
    
def string_to_block(bltl_str: str) -> Block:
    """Convert a BLTL string back to Block structure"""

    # TODO: have not tested this function yet, will use it for further analysis if needed 
    
    def parse_atomic(atomic_str: str) -> Block:
        """Parse atomic proposition like 'var==1' or 'var>0.5'"""
        block = Block()
        block.t = Type.B
        block.LA = Arithmetic()
        block.RA = Arithmetic()
        
        # Find operator
        for op in REOperator:
            if op.value in atomic_str:
                block.op = op
                var, val = atomic_str.split(op.value)
                block.LA.var = var
                block.RA.val = float(val)
                return block
        raise ValueError(f"Invalid atomic proposition: {atomic_str}")

    def find_matching_paren(s: str, start: int) -> int:
        """Find matching closing parenthesis"""
        count = 0
        for i in range(start, len(s)):
            if s[i] == '(':
                count += 1
            elif s[i] == ')':
                count -= 1
                if count == 0:
                    return i
        raise ValueError("Unmatched parentheses")

    def parse_temporal(bltl: str) -> Block:
        """Parse temporal formulas like F[t], G[t], etc."""
        block = Block()
        
        # Parse operator type
        if bltl.startswith('F['):
            if 'G[' in bltl[bltl.find('(')+1:]:
                block.t = Type.F_G
            else:
                block.t = Type.F_R
        elif bltl.startswith('G['):
            if 'F[' in bltl[bltl.find('(')+1:]:
                block.t = Type.G_F
            else:
                block.t = Type.G_R
        
        # Parse bound
        bound_start = bltl.find('[') + 1
        bound_end = bltl.find(']')
        block.bound = float(bltl[bound_start:bound_end])
        
        # Parse inner formula
        inner_start = bltl.find('(') + 1
        inner_end = find_matching_paren(bltl, inner_start)
        inner_formula = bltl[inner_start:inner_end]
        
        if block.t in [Type.F_R, Type.G_R]:
            block.F = parse_formula(inner_formula)
        else:  # Nested temporal
            block.F = parse_temporal(inner_formula)
            
        return block

    def parse_formula(bltl: str) -> Block:
        """Main parsing function"""
        # Remove outer parentheses if present
        bltl = bltl.strip()
        if bltl.startswith('(') and bltl.endswith(')'):
            bltl = bltl[1:-1]
        
        # Check for boolean combinations
        if '&' in bltl and not (bltl.startswith('F[') or bltl.startswith('G[')):
            block = Block()
            block.t = Type.C_LR
            left, right = bltl.split('&', 1)
            block.LB = parse_formula(left)
            block.RB = parse_formula(right)
            return block
        elif '|' in bltl and not (bltl.startswith('F[') or bltl.startswith('G[')):
            block = Block()
            block.t = Type.D_LR
            left, right = bltl.split('|', 1)
            block.LB = parse_formula(left)
            block.RB = parse_formula(right)
            return block
        elif 'U[' in bltl:
            block = Block()
            block.t = Type.U_LR
            left = bltl[:bltl.find('U[')]
            bound_start = bltl.find('[') + 1
            bound_end = bltl.find(']')
            block.bound = float(bltl[bound_start:bound_end])
            right = bltl[bltl.find('](') + 2:-1]
            block.LB = parse_formula(left)
            block.RB = parse_formula(right)
            return block
        elif bltl.startswith(('F[', 'G[')):
            return parse_temporal(bltl)
        else:
            return parse_atomic(bltl)

    return parse_formula(bltl_str)
    
def create_bio_bltl_generator():
    biomodels_path = os.path.join(os.path.dirname(__file__), 'cellcollective_models.json')
    biomodels = []
    with open(biomodels_path, 'r') as f:
        for line in f: 
            data = json.loads(line)
            biomodels.append(data)

    generator = BLTLGenerator(biomodels_kb=biomodels, mode='bio')
    return generator

def test_check_variables():
    vars = ['3MBz', 'G(i/o)', 'Fox1.2', 'Ara+', 'MSK1/2']
    
    vars = check_variables(vars)
    print(vars)

def main():
    # For testing, comment out before running
    # generator = create_bio_bltl_generator()

    # seq_bltls = []
    # for i in range(10):
    #     print('\n')
    #     print('generating BLTLs from the species in the model: ', generator.curr_model['model_name'])

    #     for j in range(10):
    #         block = generator.generate_bltl()

    #         bltl = generator.block_to_string(block)
    #         print(bltl)

    #         #for debugging

    #         if len(generator.sequential_vars) > 1:
    #             seq_bltls.append(bltl)
            
    #     generator.update_model_index()

    # print('\n')
    # print('sequential ones:')
    # for seq_bltl in seq_bltls:
    #     print(seq_bltl)

    #test_check_variables()
    pass 
    
if __name__ == "__main__":
    main()