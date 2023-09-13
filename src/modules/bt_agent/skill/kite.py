from modules.bt_agent.skill.base import base_skill
from modules.bt_agent.action.move import move_action
from modules.bt_agent.action.attack import attack_action

class kite_skill_semiauto(base_skill):
    def __init__(self):
        super().__init__()
        self.move_action = move_action()
        self.attack_action = attack_action()
        self.attack_flag = False # true for attacked, false for move
        self.target = None 
        self.direction = None

    def change(self):
        self.attack_flag = ~self.attack_flag

    def fill(self, target=None, direction=None):
        self.target = target
        self.direction = direction

    def execute(self):
        if self.attack_flag:
            return self.attack_action.attack_act_id(self.target)
        else:
            return self.move_action.move_act_id(self.direction)

