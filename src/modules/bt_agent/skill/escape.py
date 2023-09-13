from modules.bt_agent.skill.base import base_skill
from modules.bt_agent.action.move import move_action

class escape_skill(base_skill):
    def __init__(self):
        super().__init__()
        self.escape_direction = 'N'
        self.move_action = move_action()

    def fill(self, direction):
        self.escape_direction = direction

    def execute(self):
        return self.move_action.move_act_id(self.escape_direction)
