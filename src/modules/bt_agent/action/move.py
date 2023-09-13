import py_trees
from modules.bt_agent.action.base import base_action

class move_action(base_action):
    def __init__(self):
        super().__init__()
        self.move_id = {
            'e': self.base_action_config['move_east_id'],
            'w': self.base_action_config['move_west_id'],
            'n': self.base_action_config['move_north_id'],
            's': self.base_action_config['move_south_id'],
            'N': self.base_action_config['stop_id']
        }

    def move_act_id(self, direction):
        return self.move_id[direction]
