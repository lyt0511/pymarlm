import py_trees
import numpy as np
from modules.bt_agent.team.node import Node
from utils.find_path import Map_grid
from utils.state_calc import calcPosList, calcEvadeDirection, calcChaseDirection

from modules.bt_agent.action.move import move_action
from modules.bt_agent.action.attack import attack_action

from modules.bt_agent.skill.escape import escape_skill
from modules.bt_agent.skill.kite import kite_skill_semiauto

import pdb

class Move(Node):
    def __init__(self, namespace):
        super().__init__(namespace)
        self.move_action = move_action()

    def update(self):
        move_direction = self.bb.move_direction        
        move_action_id = self.move_action.move_act_id(move_direction)
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
                # move policy when metting wall    
                # Todo : at corner, then move to center 
                if sum(avail_actions[idx][2:6]) >= 2:
                    self.gb.action[idx] = self.eb.stop_id 
                    return py_trees.common.Status.SUCCESS
                # metting one wall, move to orthogonal direction  
                elif sum(avail_actions[idx][2:6]) == 1:
                    state = self.gb.state              
                    pos_x = state[idx*self.eb.state_ally_feat_size + self.eb.state_ally_x_id]
                    pos_y = state[idx*self.eb.state_ally_feat_size + self.eb.state_ally_y_id]
                    if avail_actions[idx][self.eb.move_east_id] == 1 or \
                       avail_actions[idx][self.eb.move_west_id] == 1:
                        if pos_y >= 0:
                            self.gb.action[idx] = self.eb.move_south_id  
                        else:
                            self.gb.action[idx] = self.eb.move_north_id
                    else:
                        if pos_x >= 0:
                            self.gb.action[idx] = self.eb.move_west_id  
                        else:
                            self.gb.action[idx] = self.eb.move_east_id
                        
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

        map_width = self.eb.map_width
        map_height = self.eb.map_height
        self.map_grid = Map_grid(map_width, map_height)

    def update(self):
        avail_actions = self.gb.avail_actions

        if sum(self.action_pt) < 0:
            self.target_pos = self.bb.move_queue_target_pos

        # judge if pointer reach the aciont queue rear
        def reach_action_rear(self, act_q, pt, idx):
            # reach rear then reset action queue and pointer
            if pt == len(act_q):
                self.action_queue[idx] = []
                self.action_pt[idx] = -1
                self.action_finish[idx] = True

        def calc_move_pos_actions(src_pos, tar_pos):
            # version 1: A* 算法计算两个坐标之间的路径和动作
            # state中的坐标是经过处理的相对坐标，所以要反处理
            center_pos_x = self.eb.map_width / 2
            center_pos_y = self.eb.map_height / 2

            src_x = int(src_pos[0] * self.eb.area_width + center_pos_x)
            src_y = int(src_pos[1] * self.eb.area_height + center_pos_y)
            tar_x = int(tar_pos[0] * self.eb.area_width)
            tar_y = int(tar_pos[1] * self.eb.area_height)

            print (src_x, src_y, tar_x, tar_y)

            return self.map_grid.path_finder.find_path(self.map_grid.grid, src_x, src_y, tar_x, tar_y)

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
                
                print (self.action_queue[i])
                self.action_pt[i] += 1

                self.gb.action[idx] = self.action_queue[i][self.action_pt[i]]
                self.action_pt[i] += 1

                # only one action in action queue then reset action queue and pointer
                reach_action_rear(self, self.action_queue[i], self.action_pt[i], i)
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
        self.attack_action = attack_action()
        self.move_action = move_action()
    
    def update(self):
        target = self.bb.target
        group_actions = []
        avail_actions = self.gb.avail_actions
        state = self.gb.state
        attack_action_id = self.attack_action.attack_act_id(target)

        # 求组内智能体向目标移动的方向
        pos_x,pos_y,e_pos_x,e_pos_y = calcPosList(self.gb.state, self.bb.target,self.bb.group, self.eb.state_ally_feat_size, self.eb.state_ally_x_id, self.eb.state_ally_y_id, self.eb.n_agents, self.eb.state_enemy_feat_size, self.eb.state_enemy_x_id, self.eb.state_enemy_y_id)        
        chase_direction = calcChaseDirection(pos_x,pos_y,e_pos_x,e_pos_y)

        for idx in self.bb.group:
            # 对于组内每个智能体
            # 目标在攻击范围内就攻击
            if avail_actions[idx][target+self.eb.none_attack_bits] == 1:
                # attack target (id = target id + self.eb.none_attack_bits (noop stop n s e w))
                self.gb.action[idx] = attack_action_id
                # group_actions.append(target+6)
                
            # 目标不在攻击范围内就朝所在方向追击
            else:               
                move_action_id = self.move_action.move_act_id(chase_direction[idx])
                self.gb.action[idx] = move_action_id

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
            e_pos_x = self.bb.target_visible_center_pos[0]
            e_pos_y = self.bb.target_visible_center_pos[1]
            # delta_x value: positive - target at east, negative -  target at west
            # delta_y value: positive - target at north, negative - target at south
            delta_x = e_pos_x - pos_x 
            delta_y = e_pos_y - pos_y
            if abs(delta_x) < abs(delta_y):
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
    

class Escape(Node):
    def __init__(self, namespace):
        super().__init__(namespace)
        self.escape_skill = escape_skill()
    
    def update(self):
        pos_x,pos_y,e_pos_x,e_pos_y = calcPosList(self.gb.state, self.bb.target, self.bb.group, self.eb.state_ally_feat_size, self.eb.state_ally_x_id, self.eb.state_ally_y_id, self.eb.n_agents, self.eb.state_enemy_feat_size, self.eb.state_enemy_x_id, self.eb.state_enemy_y_id)
        escape_direction = calcEvadeDirection(pos_x,pos_y,e_pos_x,e_pos_y)
        self.escape_skill.fill(escape_direction)

        move_action_id = self.escape_skill.execute()
        avail_actions = self.gb.avail_actions
        group_actions = []
        for idx in self.bb.group:
            if avail_actions[idx][move_action_id] == 1:
                # move to direction
                self.gb.action[idx] = move_action_id
                # group_actions.append(move_action_id)
                
                # must reset the move direction in black board
                self.bb.move_direction = 'N'
                self.escape_skill.fill('N')
            else:       
                # stop     
                self.gb.action[idx] = self.eb.stop_id     
                # group_actions.append(self.eb.stop_id)

        return py_trees.common.Status.SUCCESS

# kite ver1 action node
class Kite(Node):
    def __init__(self, namespace):
        super().__init__(namespace)
        self.kite_skill = kite_skill_semiauto()

    def update(self):        
        target = self.bb.target
        group_actions = []
        avail_actions = self.gb.avail_actions
        state = self.gb.state

        for idx in self.bb.group:
            if avail_actions[idx][target+self.eb.none_attack_bits] == 1:
                if self.kite_skill.attack_flag == False:
                    # attack target (id = target id + 6 (noop stop n s e w))
                    # group_actions.append(target+self.none_attack_bits)
                    self.kite_skill.change()
                    self.kite_skill.fill(target=target)
                    self.gb.action[idx] = self.kite_skill.execute()
                else:                    
                    pos_x,pos_y,e_pos_x,e_pos_y = calcPosList(self.gb.state, self.bb.target,self.bb.group, self.eb.state_ally_feat_size, self.eb.state_ally_x_id, self.eb.state_ally_y_id, self.eb.n_agents, self.eb.state_enemy_feat_size, self.eb.state_enemy_x_id, self.eb.state_enemy_y_id)
                    kite_direction = calcEvadeDirection(pos_x,pos_y,e_pos_x,e_pos_y)                    

                    self.kite_skill.change()
                    self.kite_skill.fill(direction=kite_direction)
                    self.gb.action[idx] = self.kite_skill.execute()
            else:
                # out of attack range, stop (Todo: better move strategy)
                self.gb.action[idx] = self.eb.stop_id
                # group_actions.append(self.eb.stop_id)
        return py_trees.common.Status.SUCCESS