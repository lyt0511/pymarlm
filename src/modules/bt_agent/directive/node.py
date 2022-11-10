import py_trees

def register_gb_keys(self):
    self.gb.register_key('action', access=py_trees.common.Access.WRITE)
    self.gb.register_key('state', access=py_trees.common.Access.WRITE)
    self.gb.register_key('obs', access=py_trees.common.Access.WRITE)
    self.gb.register_key('avail_actions', access=py_trees.common.Access.WRITE)
    self.gb.register_key('group_dict', access=py_trees.common.Access.WRITE)
    self.gb.register_key('evade_hp', access=py_trees.common.Access.WRITE)
    self.gb.register_key('kite_hp', access=py_trees.common.Access.WRITE)
    self.gb.register_key('under_attack', access=py_trees.common.Access.WRITE)

def init_gb_keys(self):
    self.gb.obs = []
    self.gb.state = []
    self.gb.avail_actions = []
    self.gb.action = {}
    self.gb.group_dict = {}
    self.gb.evade_hp = 0
    self.gb.kite_hp = 0
    self.gb.under_attack = []

def register_eb_keys(self):
    self.eb.register_key('n_agents', access=py_trees.common.Access.WRITE)
    self.eb.register_key('n_enemies', access=py_trees.common.Access.WRITE)
    self.eb.register_key('shield_bits_ally', access=py_trees.common.Access.WRITE)
    self.eb.register_key('shield_bits_enemy', access=py_trees.common.Access.WRITE)
    self.eb.register_key('unit_type_bits', access=py_trees.common.Access.WRITE)
    self.eb.register_key('move_feat_size', access=py_trees.common.Access.WRITE)
    self.eb.register_key('obs_ally_feat_size', access=py_trees.common.Access.WRITE)
    self.eb.register_key('obs_agent_hp_id', access=py_trees.common.Access.WRITE)    
    self.eb.register_key('state_ally_feat_size', access=py_trees.common.Access.WRITE)
    self.eb.register_key('state_ally_x_id', access=py_trees.common.Access.WRITE)
    self.eb.register_key('state_ally_y_id', access=py_trees.common.Access.WRITE)
    self.eb.register_key('state_enemy_feat_size', access=py_trees.common.Access.WRITE)
    self.eb.register_key('state_enemy_x_id', access=py_trees.common.Access.WRITE)
    self.eb.register_key('state_enemy_y_id', access=py_trees.common.Access.WRITE)
    self.eb.register_key('none_attack_bits', access=py_trees.common.Access.WRITE)
    self.eb.register_key('noop_id', access=py_trees.common.Access.WRITE)
    self.eb.register_key('stop_id', access=py_trees.common.Access.WRITE)
    self.eb.register_key('move_north_id', access=py_trees.common.Access.WRITE)
    self.eb.register_key('move_south_id', access=py_trees.common.Access.WRITE)
    self.eb.register_key('move_east_id', access=py_trees.common.Access.WRITE)
    self.eb.register_key('move_west_id', access=py_trees.common.Access.WRITE)

def init_eb_keys(self):
    self.eb.n_agents = 0
    self.eb.n_enemies = 0

    self.eb.shield_bits_ally = 0
    self.eb.shield_bits_enemy = 0
    self.eb.unit_type_bits = 0

    self.eb.move_feat_size = 0

    self.eb.obs_ally_feat_size = 0 
    self.eb.obs_agent_hp_id = 0 

    self.eb.state_ally_feat_size = 0
    self.eb.state_ally_x_id = 0
    self.eb.state_ally_y_id = 0
    # enemy - (hp,x,y,shield,unitype)
    self.eb.state_enemy_feat_size = 0
    self.eb.state_enemy_x_id = 0
    self.eb.state_enemy_y_id = 0    

    self.eb.none_attack_bits = 0
    # action id
    self.eb.noop_id = 0
    self.eb.stop_id = 0
    self.eb.move_north_id = 0
    self.eb.move_south_id = 0
    self.eb.move_east_id = 0
    self.eb.move_west_id = 0

# def register_keys(self):
#     self.bb.register_key