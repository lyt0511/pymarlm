from modules.bt_agent.directive.plan import Plan

class Agent:
    def __init__(self, args):
        self.args = args
        self.plan = Plan(args)

    def reset(self):
        self.plan = Plan(self.args)

    def get_action(self, state, obs, avail_actions):
        action = self.plan.step(state, obs, avail_actions)
        return action
