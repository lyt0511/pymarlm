def calcPosList(state, target_id, ally_groups, \
                ally_ft_size, ally_x_id, ally_y_id, \
                ally_num, enemy_ft_size, enemy_x_id, enemy_y_id):

    pos_x = []
    pos_y = []
    for idx in ally_groups:             
        pos_x.append(state[idx*ally_ft_size + ally_x_id])
        pos_y.append(state[idx*ally_ft_size + ally_y_id])

    e_pos_x = state[ally_num*ally_ft_size+                   target_id*enemy_ft_size+enemy_x_id]
    e_pos_y = state[ally_num*ally_ft_size+                           target_id*enemy_ft_size+enemy_y_id]

    return pos_x, pos_y, e_pos_x, e_pos_y



def calcEvadeDirection(pos_x_list, pos_y_list, e_pos_x, e_pos_y):

    for i in range(len(pos_x_list)):             
        pos_x = pos_x_list[i]
        pos_y = pos_y_list[i]
        # delta_x value: positive - target at east, negative -  target at west
        # delta_y value: positive - target at north, negative - target at south
        delta_x = e_pos_x - pos_x 
        delta_y = e_pos_y - pos_y
        if abs(delta_x) < abs(delta_y):
            if delta_x < 0:
                return 'e'
            else:    
                return 'w'
        else:
            if delta_y < 0:
                return 'n'
            else:   
                return 's'
            

def calcChaseDirection(pos_x_list, pos_y_list, e_pos_x, e_pos_y):  
    chase_direction = []
    for i in range(len(pos_x_list)):            
        pos_x = pos_x_list[i]
        pos_y = pos_y_list[i]  
        # delta_x value: positive - target at east, negative -  target at west
        # delta_y value: positive - target at north, negative - target at south
        delta_x = e_pos_x - pos_x 
        delta_y = e_pos_y - pos_y
        if abs(delta_x) > abs(delta_y):
            if delta_x < 0:
                chase_direction.append('w')
            else:               
                chase_direction.append('e')  
        else:
            if delta_y < 0:
                chase_direction.append('s')
            else:
                chase_direction.append('n')  
    
    return chase_direction
