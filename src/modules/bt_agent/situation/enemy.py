from modules.bt_agent.situation.base import base_situation

class enemy_situation(base_situation):
    def __init__(self, environment_args):
        super().__init__(environment_args)
        self.obs = None
        self.situation_list = {
            'enemy_visible': [[] for i in range(self.eb.n_agents)],
            'enemy_dis_sort': [[] for i in range(self.eb.n_agents)],
            'enemy_hpsp_sort': [[] for i in range(self.eb.n_agents)],
        }

    def update_situation_list(self):
        self.update_enemy_dis_sort_list()
        self.update_enemy_hpsp_sort_list()
        # self.update_enemy_visible_list()
    
    def update_enemy_visible_list(self):
        # Todo: calc visible but cannot be attacked enemies
        return      

    def update_enemy_dis_sort_list(self):
        for i, agent_obs in enumerate(self.obs):
            # get enemies' canAttack, distance
            idx_bases = [self.eb.move_feat_size + idx_enemy * self.eb.obs_ally_feat_size for idx_enemy in range(self.eb.n_agents)]
            dis_idx = [idx_base+1 for idx_base in idx_bases]
            
            # sort target enemy by distance
            enemy_dis = agent_obs[dis_idx]
            enemy_dis_tuple = [(j,dis) for j,dis in enumerate(enemy_dis.tolist())]            
            self.situation_list['enemy_dis_sort'][i] = sorted(enemy_dis_tuple, key=lambda x:x[1])


    def update_enemy_hpsp_sort_list(self):     
        for i, agent_obs in enumerate(self.obs):
            # get enemies' canAttack, distance
            idx_bases = [self.eb.move_feat_size + idx_enemy * self.eb.obs_ally_feat_size for idx_enemy in range(self.eb.n_agents)]
            hp_idx = [idx_base+4 for idx_base in idx_bases]
            enemy_hpsp = agent_obs[hp_idx]
            if self.eb.shield_bits_ally > 0:
                sp_idx = [idx_base+5 for idx_base in idx_bases]
                enemy_sp = agent_obs[sp_idx]
                enemy_hpsp += enemy_sp

            # sort target enemy by hp(+sp)
            enemy_hpsp_tuple = [(j,hpsp) for j,hpsp in enumerate(enemy_hpsp.tolist())]

            self.situation_list['enemy_hpsp_sort'][i] = sorted(enemy_hpsp_tuple, key=lambda x:x[1])