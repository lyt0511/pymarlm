import py_trees

class base_action:
    def __init__(self):
        self.base_action_config = {
            #### move id in sc2
            'move_east_id': 4,
            'move_west_id': 5,
            'move_north_id': 2,
            'move_south_id': 3,
            'stop_id': 1,

            #### move length for attack action id
            'attack_start_id': 6
        }