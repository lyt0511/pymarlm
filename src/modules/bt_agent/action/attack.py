import py_trees
from modules.bt_agent.action.base import base_action

class attack_action(base_action):
    def __init__(self, ):
        super().__init__()
        self.attack_start_id = self.base_action_config['attack_start_id']

    def attack_act_id(self, target_id):
        return self.attack_start_id + target_id
