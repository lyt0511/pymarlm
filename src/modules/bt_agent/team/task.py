import py_trees
from uuid import uuid4

from modules.bt_agent.team.node import register_keys, init_keys
from modules.bt_agent.team import action, cond

class Team():
    def __init__(self, flag_num=0, unit_mode='long'):
        self.flag_num = flag_num
        self.namespace = str(uuid4())
        self.unit_mode = unit_mode
        self.create_bb()
        self.update_group()
        self.create_tree(unit_mode)

    def create_bb(self):
        self.bb = py_trees.blackboard.Client(namespace=self.namespace)
        self.gb = py_trees.blackboard.Client(namespace='global')
        self.eb = py_trees.blackboard.Client(namespace='env')

        register_keys(self)
        init_keys(self)

    def create_tree(self, unit_mode):
        if unit_mode == 'long':
            self.root = py_trees.composites.Selector(
                children=[
                    py_trees.composites.Sequence(
                        children=[
                            cond.CanEvade(self.namespace),
                            action.Escape(self.namespace)
                        ]
                    ),
                    # kite ver 1: use kite node
                    py_trees.composites.Sequence(
                        children=[
                            cond.CanKite(self.namespace),
                            action.Kite(self.namespace)
                        ]
                    ),
                    # kite ver 2: use evade node
                    # py_trees.composites.Sequence(
                    #     children=[
                    #         cond.CanKite(self.namespace),
                    #         action.CalcEvadeDirection(self.namespace),
                    #         action.Move(self.namespace)
                    #     ]
                    # ),
                    py_trees.composites.Sequence(
                        children=[
                            cond.CanAttack(self.namespace, unit_mode),
                            action.Attack(self.namespace)
                        ]
                    ),
                    py_trees.composites.Sequence(
                        children=[
                            cond.CanMove(self.namespace),
                            action.Move(self.namespace)
                        ]
                    )
                ]
            )
        else:
            self.root = py_trees.composites.Selector(
                children=[
                    py_trees.composites.Sequence(
                        children=[
                            cond.CanAttack(self.namespace, unit_mode),
                            action.Attack(self.namespace)
                        ]
                    ),
                    py_trees.composites.Sequence(
                        children=[
                            cond.CanMove(self.namespace),
                            action.Move(self.namespace)
                        ]
                    )
                ]
            )

        self.bt = py_trees.trees.BehaviourTree(root=self.root)

    def update_group(self):
        # self.bb.group = list(filter(lambda k: k.identity_id in self.uid, self.en.units))
        member_id = self.gb.group_dict[self.flag_num]
        self.bb.group.append(member_id)

        return

    def step(self):
        # self.update_group()
        self.bt.tick()

        return