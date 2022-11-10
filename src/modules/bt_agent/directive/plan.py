import py_trees
from uuid import uuid4
import numpy as np

from modules.bt_agent.directive.node import register_gb_keys, init_gb_keys, register_eb_keys, init_eb_keys
from modules.bt_agent.team import task

import pdb

class Plan():
    def __init__(self, args):
        self.args = args
        self.namespace = 'plan'
        self.create_bb()
        self.create_group()
        self.create_tree()

        self.update_bb()

        self.init_var()

    def init_var(self):
        self.last_agent_hpsp = [0] * self.args.n_agents
    
    def create_bb(self):
        self.bb = py_trees.blackboard.Client(namespace=self.namespace)
        self.gb = py_trees.blackboard.Client(namespace='global')
        self.eb = py_trees.blackboard.Client(namespace='env')

        register_gb_keys(self)
        init_gb_keys(self)
        register_eb_keys(self)
        init_eb_keys(self)

    # Todo: update group assignment in real-time
    def create_group(self):
        for i in range(self.args.n_agents):
            self.gb.group_dict[i] = i

    def create_tree(self):
        #Todo: create directive tree to control assignment
        # self.root = py_trees.composites.Parallel(
        #     children=[
        #         task.Team()
        #     ]
        # )

        self.team = [task.Team(i) for i in range(self.args.n_agents)]

    def update_ally_info(self):
        self.update_attack_target()
        self.update_visible_target()
        self.update_ally_under_attack()
    
    def step(self, state, obs, avail_actions):
        # self.update_group()
        self.gb.action = {}
        for i in range(self.args.n_agents):
            self.gb.action[i] = -1
        #Todo: env can be included in global blackboard
        self.gb.state = state
        self.gb.obs = obs
        self.gb.avail_actions = avail_actions

        # update target
        self.update_ally_info()

        for team in self.team:
            team.step()
        
        # transform action dict to action list
        actions = [act[1] for act in sorted(self.gb.action.items(),key=lambda x:x[0])]
        # print (actions)
        # available action filter
        for idx, avail_action in enumerate(avail_actions):
            # dead agent can only act 0 (Todo: filter dead agent in group_dict and not assignment)
            if avail_action[0] == 1:
                actions[idx] = 0
            # not available action set action to stop
            elif avail_action[actions[idx]] == 0:
                actions[idx] = 1
            else:
                continue
        
        # print ('===================================================')
        # for i in range(self.eb.n_agents):
        #     print("{}".format(py_trees.display.unicode_tree(self.team[i].root, show_status=True)))
        
        # for debug
        # input()

        return actions

    def update_bb(self):
        self.gb.evade_hp = 0.10
        self.gb.kite_hp = 0.24

        self.gb.under_attack = [False for _ in range(self.args.n_agents)]

        self.eb.n_agents = self.args.n_agents
        self.eb.n_enemies = self.args.n_enemies

        self.eb.shield_bits_ally = self.args.shield_bits_ally
        self.eb.shield_bits_enemy = self.args.shield_bits_enemy
        self.eb.unit_type_bits = self.args.unit_type_bits

        # observation
        self.eb.move_feat_size = 4
        self.eb.obs_ally_feat_size = 5 + self.args.shield_bits_ally + self.args.unit_type_bits
        self.eb.obs_agent_hp_id = self.eb.move_feat_size + self.eb.obs_ally_feat_size * \
                                    (self.eb.n_agents + self.eb.n_enemies - 1)

        # state
        # ally - (hp,mp,x,y,shield,unitype)
        self.eb.state_ally_feat_size = 4 + self.args.shield_bits_ally + self.args.unit_type_bits
        self.eb.state_ally_x_id = 2
        self.eb.state_ally_x_id = 3
        # enemy - (hp,x,y,shield,unitype)
        self.eb.state_enemy_feat_size = 3 + self.args.shield_bits_enemy + self.args.unit_type_bits
        self.eb.state_enemy_x_id = 1
        self.eb.state_enemy_x_id = 2        

        self.eb.none_attack_bits = 6
        # action id
        self.eb.noop_id = 0
        self.eb.stop_id = 1
        self.eb.move_north_id = 2
        self.eb.move_south_id = 3
        self.eb.move_east_id = 4
        self.eb.move_west_id = 5

    def update_attack_target(self):
        # get the nearest enemy (Todo: get all enemies?)
        obs = self.gb.obs

        for i, agent_obs in enumerate(obs):
            # get enemies' canAttack, distance
            idx_bases = [self.eb.move_feat_size + idx_enemy * self.eb.obs_ally_feat_size \
                                                for idx_enemy in range(self.eb.n_agents)]
            canatt_idx = [idx_base for idx_base in idx_bases]
            dis_idx = [idx_base+1 for idx_base in idx_bases]

            enemy_canatt = agent_obs[canatt_idx]

            # if enemy_canatt is empty, ith team's target must be set to -1
            # (otherwise previous attack target exists even if it out of range in the following steps)
            if enemy_canatt.sum() == 0:
                self.team[i].bb.target = -1
                continue

            enemy_dis = agent_obs[dis_idx]
            enemy_dis_tuple = [(j,dis) for j,dis in enumerate(enemy_dis.tolist())]
            dis_sort = sorted(enemy_dis_tuple, key=lambda x:x[1])

            for j, _ in dis_sort:
                if enemy_canatt[j] == 1:
                    self.team[i].bb.target = j
                    break

            # pdb.set_trace()

    def update_visible_target(self):
        # get the nearest enemy (Todo: get all enemies?)
        obs = self.gb.obs

        for i, agent_obs in enumerate(obs):
            # get enemies' canAttack, distance
            idx_bases = [self.eb.move_feat_size + idx_enemy * self.eb.obs_ally_feat_size \
                                                for idx_enemy in range(self.eb.n_enemies)]
            
            # visible and hp > 0 enemy, the distance (id+1) must larger than 0                                    
            dis_idx = [idx_base+1 for idx_base in idx_bases]

            enemy_dis = agent_obs[dis_idx]

            # if enemy_dis is empty, ith team's target must be set to -1
            # (otherwise previous visiable target exists even if it out of range in the following steps)
            if enemy_dis.sum() == 0:
                self.team[i].bb.target_visible = -1
                continue

            enemy_dis = agent_obs[dis_idx]
            enemy_dis_tuple = [(j,dis) for j,dis in enumerate(enemy_dis.tolist())]
            dis_sort = sorted(enemy_dis_tuple, key=lambda x:x[1])

            for j, dis in dis_sort:
                if dis > 0:
                    self.team[i].bb.target_visible = j
                    break


    def update_ally_under_attack(self):
        # get the nearest enemy (Todo: get all enemies?)
        obs = self.gb.obs
        hp_id = self.eb.obs_agent_hp_id
        if self.eb.shield_bits_ally > 0:
            sp_id = self.eb.obs_agent_hp_id + 1

        for i, agent_obs in enumerate(obs):
            # get agents' hp + sp(shield if exists)
            hpsp = 0
            if self.eb.shield_bits_ally > 0:
                hpsp = agent_obs[hp_id] + agent_obs[sp_id]
            else:                
                hpsp = agent_obs[hp_id]
            
            if self.last_agent_hpsp[i] - hpsp > 0:
                self.gb.under_attack[i] = True

            # update the last agent hpsp
            self.last_agent_hpsp[i] = hpsp


            # pdb.set_trace() 

    