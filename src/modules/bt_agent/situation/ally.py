from modules.bt_agent.situation.base import base_situation

class ally_situation(base_situation):
    def __init__(self, environment_args):
        super().__init__(environment_args)
        self.obs = None
        self.situation_list = {
            'ally_last_hpsp': [[] for i in range(self.eb.n_agents)],
            'ally_under_attack': [False for i in range(self.eb.n_agents)],            
        }
    
    def update_situation_list(self):
        self.update_ally_under_attack()
    
    def update_ally_under_attack(self):
        # get the nearest enemy (Todo: get all enemies?)
        hp_id = self.eb.obs_agent_hp_id
        if self.eb.shield_bits_ally > 0:
            sp_id = self.eb.obs_agent_hp_id + 1

        for i, agent_obs in enumerate(self.obs):
            # get agents' hp + sp(shield if exists)
            hpsp = 0
            if self.eb.shield_bits_ally > 0:
                hpsp = agent_obs[hp_id] + agent_obs[sp_id]
            else:                
                hpsp = agent_obs[hp_id]
            
            if self.situation_list['ally_last_hpsp'][i] - hpsp > 0:
                self.situation_list['ally_under_attack'][i] = True

            # update the last agent hpsp
            self.situation_list['ally_last_hpsp'][i] = hpsp