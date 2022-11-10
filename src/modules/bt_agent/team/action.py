import py_trees
import numpy as np
from modules.bt_agent.team.node import Node

import pdb

class Move(Node):
    def __init__(self, namespace):
        super().__init__(namespace)
        self.move_id = {}

    def update(self):
        if len(self.move_id) == 0:            
            self.move_id = {
                'e': self.eb.move_east_id,
                'w': self.eb.move_west_id,
                'n': self.eb.move_north_id,
                's': self.eb.move_south_id,
                'N': self.eb.stop_id
            }

        move_direction = self.bb.move_direction
        move_action_id = self.move_id[move_direction]
        avail_actions = self.gb.avail_actions
        group_actions = []
        for idx in self.bb.group:
            if avail_actions[idx][move_action_id] == 1:
                # move to direction
                self.gb.action[idx] = move_action_id
                # group_actions.append(move_action_id)
                
                # must reset the move direction in black board
                self.bb.move_direction = 'N'
            else:       
                # stop     
                self.gb.action[idx] = self.eb.stop_id     
                # group_actions.append(self.eb.stop_id)

        return py_trees.common.Status.SUCCESS

class Move_Queue(Node):
    def __init__(self, namespace):
        super().__init__(namespace)
        self.target_pos = (0,0)
        self.action_queue = []
        self.action_pt = []
        self.action_finish = []
        for idx in self.bb.group:
            self.action_queue.append([])
            self.action_pt.append(-1)
            self.action_finish.append(False)

    def update(self):
        avail_actions = self.gb.avail_actions

        if self.action_pt.sum < 0:
            self.target_pos = self.bb.move_queue_target_pos

        # judge if pointer reach the aciont queue rear
        def reach_action_rear(self, act_q, pt, idx):
            # reach rear then reset action queue and pointer
            if pt == len(act_q):
                self.action_queue[idx] = []
                self.action_pt[idx] = -1
                self.action_finish[idx] = True

        def calc_move_pos_actions(src_pos, tar_pos):
            # ============Todo===================
            # 计算两个坐标的距离
            # 计算每移动一步前进的坐标数
            # 生成动作
            return 

        state = self.gb.state
        for i, idx in enumerate(self.bb.group):
            if self.action_finish[i]:
                self.gb.action[idx] = self.eb.stop_id
                continue 

            pos_x = state[idx*self.eb.state_ally_feat_size + self.eb.state_ally_x_id]
            pos_y = state[idx*self.eb.state_ally_feat_size + self.eb.state_ally_y_id]

            if self.action_pt[i] == -1:
                # calc pos actions and step the first action
                agent_pos = (pos_x, pos_y)
                self.action_queue[i] = calc_move_pos_actions(agent_pos, self.target_pos)
                self.action_pt[i] += 1

                self.gb.action[idx] = self.action_queue[i][self.action_pt[i]]
                self.action_pt[i] += 1

                # only one action in action queue then reset action queue and pointer
                reach_action_rear(self.action_queue[i], self.action_pt[i], i)
            else:
                # step following actions, if reach at the last aciton then reset action queue and pointer
                self.gb.action[idx] = self.action_queue[i][self.action_pt[i]]
                self.action_pt[i] += 1                    
                
                reach_action_rear(self, self.action_queue[i], self.action_pt[i], i)

        # any agent not finish, node running   
        for i, idx in enumerate(self.bb.group):
            if self.action_finish[i] == False:
                return py_trees.common.Status.RUNNING        
        
        # all agent finish, reset action_finish list, return success
        for i, idx in enumerate(self.bb.group):
            self.action_finish[i] = False   

        return py_trees.common.Status.SUCCESS


class Attack(Node):
    def __init__(self, namespace):
        super().__init__(namespace)
    
    def update(self):
        target = self.bb.target
        group_actions = []
        avail_actions = self.gb.avail_actions
        state = self.gb.state
        for idx in self.bb.group:
            if avail_actions[idx][target+self.eb.none_attack_bits] == 1:
                # attack target (id = target id + self.eb.none_attack_bits (noop stop n s e w))
                self.gb.action[idx] = target+self.eb.none_attack_bits
                # group_actions.append(target+6)
            else:
                # out of attack range, move towards the target
                pos_x = state[idx*self.eb.state_ally_feat_size + self.eb.state_ally_x_id]
                pos_y = state[idx*self.eb.state_ally_feat_size + self.eb.state_ally_y_id]
                e_pos_x = state[(idx+1)*self.eb.state_ally_feat_size+\
                                    target*self.eb.state_enemy_feat_size+self.eb.state_enemy_x_id]
                e_pos_y = state[(idx+1)*self.eb.state_ally_feat_size+\
                                    target*self.eb.state_enemy_feat_size+self.eb.state_enemy_y_id]
                # delta_x value: positive - target at east, negative -  target at west
                # delta_y value: positive - target at north, negative - target at south
                delta_x = e_pos_x - pos_x 
                delta_y = e_pos_y - pos_y
                if abs(delta_x) > abs(delta_y):
                    if delta_x < 0:
                        self.gb.action[idx] = self.eb.move_west_id
                        # group_actions.append(5)
                    else:               
                        self.gb.action[idx] = self.eb.move_east_id         
                        # group_actions.append(4)
                else:
                    if delta_y < 0:
                        self.gb.action[idx] = self.eb.move_south_id
                        # group_actions.append(3)
                    else:
                        self.gb.action[idx] = self.eb.move_north_id                        
                        # group_actions.append(2)

        return py_trees.common.Status.SUCCESS


class CalcEvadeDirection(Node):
    def __init__(self, namespace):
        super().__init__(namespace)
    
    def update(self):
        target = self.bb.target
        state = self.gb.state

        for idx in self.bb.group:             
            pos_x = state[idx*self.eb.state_ally_feat_size + self.eb.state_ally_x_id]
            pos_y = state[idx*self.eb.state_ally_feat_size + self.eb.state_ally_y_id]
            e_pos_x = state[(idx+1)*self.eb.state_ally_feat_size+\
                                target*self.eb.state_enemy_feat_size+self.eb.state_enemy_x_id]
            e_pos_y = state[(idx+1)*self.eb.state_ally_feat_size+\
                                target*self.eb.state_enemy_feat_size+self.eb.state_enemy_y_id]
            # delta_x value: positive - target at east, negative -  target at west
            # delta_y value: positive - target at north, negative - target at south
            delta_x = e_pos_x - pos_x 
            delta_y = e_pos_y - pos_y
            if abs(delta_x) > abs(delta_y):
                if delta_x < 0:
                    self.bb.move_direction = 'e'
                else:    
                    self.bb.move_direction = 'w'
            else:
                if delta_y < 0:
                    self.bb.move_direction = 'n'
                else:   
                    self.bb.move_direction = 's'

        return py_trees.common.Status.SUCCESS

# kite ver1 action node
class Kite(Node):
    def __init__(self, namespace):
        super().__init__(namespace)
        self.attack_flag = False

    def update(self):        
        target = self.bb.target
        group_actions = []
        avail_actions = self.gb.avail_actions
        state = self.gb.state

        for idx in self.bb.group:
            if avail_actions[idx][target+self.eb.none_attack_bits] == 1:
                if self.attack_flag == False:
                    # attack target (id = target id + 6 (noop stop n s e w))
                    # group_actions.append(target+self.none_attack_bits)
                    self.gb.action[idx] = target + self.eb.none_attack_bits
                    self.attack_flag = True
                else:                    
                    pos_x = state[idx*self.eb.state_ally_feat_size + self.eb.state_ally_x_id]
                    pos_y = state[idx*self.eb.state_ally_feat_size + self.eb.state_ally_y_id]
                    e_pos_x = state[(idx+1)*self.eb.state_ally_feat_size+\
                                        target*self.eb.state_enemy_feat_size+self.eb.state_enemy_x_id]
                    e_pos_y = state[(idx+1)*self.eb.state_ally_feat_size+\
                                        target*self.eb.state_enemy_feat_size+self.eb.state_enemy_y_id]
                    # delta_x value: positive - target at east, negative -  target at west
                    # delta_y value: positive - target at north, negative - target at south
                    delta_x = e_pos_x - pos_x 
                    delta_y = e_pos_y - pos_y
                    if abs(delta_x) > abs(delta_y):
                        if delta_x < 0:
                            self.gb.action[idx] = self.eb.move_east_id
                            # group_actions.append(self.eb.move_east_id)
                        else:    
                            self.gb.action[idx] = self.eb.move_west_id                    
                            # group_actions.append(self.eb.move_west_id)
                    else:
                        if delta_y < 0:
                            self.gb.action[idx] = self.eb.move_north_id
                            # group_actions.append(self.eb.move_north_id)
                        else:   
                            self.gb.action[idx] = self.eb.move_south_id                     
                            # group_actions.append(self.eb.move_south_id)
                    self.attack_flag = False
            else:
                # out of attack range, stop (Todo: better move strategy)
                self.gb.action[idx] = self.eb.stop_id
                # group_actions.append(self.eb.stop_id)
        return py_trees.common.Status.SUCCESS