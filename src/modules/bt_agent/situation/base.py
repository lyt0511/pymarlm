class base_situation:
    def __init__(self, environment_args):
        self.component_name = "situation"        
        self.eb = environment_args
        # 态势的逻辑：
        # 态势是对智能体观测的进一步加工
        # 不同的态势分为不同的在base_situation派生的子类，例如敌人、盟友等
        # 态势对观测加工形成situation字典
        # situation字典包含了智能体执行动作或技能所需要的各种列表，例如可观测敌人、敌人远近、敌人血量等
        # 如果需要扩展新的态势列表，则在situation字典里添加新的字段，实现update函数逻辑
        # 注意态势仅是进一步加工，而不是全加工，即攻击敌人长手单位攻击血量低的而短手单位攻击距离近的，态势仅提供远近和血量列表，怎么选择智能体不由态势管
        self.situation = {}
        self.obs = None
    
    def update_obs(self, obs):
        self.obs = obs
        return