import py_trees
import numpy as np

from modules.bt_agent.team.node import Node

import pdb

# selector节点下的条件应该是互斥的？？？

class CanEvade(Node):
    def __init__(self, namespace):
        super().__init__(namespace)
    
    def update(self):
        state = self.gb.state
        # 1 see anemy, 2 hp < evade hp, and is 3 under attack can evade
        if self.bb.target_visible == -1:
            return py_trees.common.Status.FAILURE

        for idx in self.bb.group:
            # print ('{} agent hp: {}'.format(idx, state[idx*self.eb.state_ally_feat_size]))
            if state[idx*self.eb.state_ally_feat_size] < self.gb.evade_hp and self.gb.under_attack[idx]:
                return py_trees.common.Status.SUCCESS
        
        return py_trees.common.Status.FAILURE

# kite ver1 condition node
# class CanKite(Node):
#     def __init__(self, namespace):
#         super().__init__(namespace)
    
#     def update(self):
#         state = self.gb.state
#         for idx in self.bb.group:
#             # hp < kite_hp then can kite
#             # print ('{} agent hp: {}'.format(idx, state[idx*self.eb.state_ally_feat_size]))
#             if state[idx*self.eb.state_ally_feat_size] < self.gb.kite_hp:
#                 self.bb.kite_action_type == 'move'
#                 return py_trees.common.Status.SUCCESS
        
#         return py_trees.common.Status.FAILURE

# kite ver2 condition node
class CanKite(Node):
    def __init__(self, namespace):
        super().__init__(namespace)
    
    def update(self):
        state = self.gb.state
        for idx in self.bb.group:
            # hp < kite_hp then can kite
            # print ('{} agent hp: {}'.format(idx, state[idx*self.eb.state_ally_feat_size]))
            if state[idx*self.eb.state_ally_feat_size] < self.gb.kite_hp and self.bb.kite_action_type == 'attack':
                self.bb.kite_action_type == 'move'
                return py_trees.common.Status.SUCCESS
        
        return py_trees.common.Status.FAILURE


class CanAttack(Node):
    def __init__(self, namespace):
        super().__init__(namespace)

    def update(self):
        state = self.gb.state
        for idx in self.bb.group:
            # hp < kite_hp then judge whether to kite
            # print ('{} agent hp: {}'.format(idx, state[idx*self.eb.state_ally_feat_size]))
            if state[idx*self.eb.state_ally_feat_size] < self.gb.evade_hp:                
                if self.bb.kite_action_type != 'move':
                    return py_trees.common.Status.FAILURE
        
        if self.bb.target != -1:
            # refer to kite action, set the kite action type to attack
            self.bb.kite_action_type == 'attack'
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


class CanMove(Node):
    def __init__(self, namespace):
        super().__init__(namespace)
    
    def update(self):
        # canmove = ! (can attack || can kite)
        if self.bb.target != -1:
            return py_trees.common.Status.FAILURE
        
        state = self.gb.state
        for idx in self.bb.group:
            if state[idx*self.eb.state_ally_feat_size] < self.gb.evade_hp:
                return py_trees.common.Status.FAILURE

        # defatul move to east
        self.bb.move_direction = 'w'

        return py_trees.common.Status.SUCCESS
        