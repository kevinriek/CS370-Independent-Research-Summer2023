import math
import random
import numpy as np
import neat
from GameManager import map_manager

def no_ai(map_manager, unit):
    return unit.pos

def rand_ai(map_manager, unit):
    move_list = map_manager.find_movement(unit)
    index = random.randint(0, len(move_list)-1)
    return move_list[index].pos

def move_pick_ai(map_manager, unit, net):
    move_list = map_manager.find_movement(unit)
    input_list = []

    #Allied pieces input
    for i in range(map_manager.Map.shape[0]):
        for j in range(map_manager.Map.shape[1]):
            #print((i, j))
            unit_ref = map_manager.Map[i, j].unit_ref
            if (unit_ref is None):
                input_list.append(0)    #Is allied unit?
                input_list.append(0)    #Is enemy unit?
                input_list.append(0)    #Unit health
            elif unit_ref.Team == 0:
                input_list.append(1)    
                input_list.append(0)
                input_list.append(unit_ref.hp)
            elif unit_ref.Team == 1:
                input_list.append(0)    
                input_list.append(1)
                input_list.append(unit_ref.hp)
            #print(len(input_list))

    input_tup = tuple(input_list)
    format_in = [ '%.2f' % elem for elem in input_tup ]
    #print("input vector: " + str(format_in))

    output = net.activate(input_tup)
    # format_out = [ '%.2f' % elem for elem in output ]
    # print("output vector: " + str(format_out))

    out_dt = {}
    index = 0
    for i in range(map_manager.Map.shape[0]):
        for j in range(map_manager.Map.shape[1]):
            out_dt[(i, j)] = output[(i*map_manager.Map.shape[0]) + j]

    sorted_moves = dict(sorted(out_dt.items(), key=lambda item: -item[1]))
    #print(sorted_moves)
    for move, rank in sorted_moves.items():
        #print(move)
        if map_manager.Map[move].visited:
            return move

    return unit.pos

def neat_ai(map_manager, unit, net):
    my_units = []
    op_units = []

    if (map_manager.curr_team == 0): 
        my_units = map_manager.Teams[0].units
        op_units = map_manager.Teams[1].units
    elif(map_manager.curr_team == 1):
        my_units = map_manager.Teams[1].units
        op_units = map_manager.Teams[0].units
    
    move_list = map_manager.find_movement(unit)
    input_list = list(np.zeros((168)))

    win_move = (0, 0)
    win_weight = -float("inf")
    for move in move_list:
        # Remove this 
        # #Cannot stay in the same spot: avoid getting trapped in a 'hole'
        # if (move.pos == unit.pos):
        #     pass
        saved_pos = unit.pos    #save unit pos for later
        saved_rot = unit.rotation

        unit.pos = move.pos
        unit.rotation = move.rotation

        if move.is_attack:      #move is of Tile Class
            map_manager.sim_combat(unit, move.unit_ref) 

        index = 0
        #This unit's input is always first
        #input_list[index] = (unit.temp_hp / 100.0)
        #index += 1

        #Allied pieces input
        for i in range(len(my_units)):
            c_unit = None
            if map_manager.curr_team == 1:
                c_unit = my_units[len(my_units) - (i + 1)]
            else:
                c_unit = my_units[i]

            if map_manager.curr_team == 1:
                input_list[index] = 1 - (c_unit.pos[0])/(map_manager.Map.shape[0]-1)
                input_list[index+1] = 1 - (c_unit.pos[1])/(map_manager.Map.shape[1]-1)
            else:
                input_list[index] = (c_unit.pos[0])/(map_manager.Map.shape[0]-1)
                input_list[index+1] = (c_unit.pos[1])/(map_manager.Map.shape[1]-1)
            input_list[index+2] = (c_unit.temp_hp / 100.0)

            tile = map_manager.Map[c_unit.pos]
            input_list[index+3] = (c_unit.Att / 100.0)
            input_list[index+4] = (c_unit.Def * tile.terrain.def_mod / 100.0)
            input_list[index+5] = (c_unit.max_move / 15.0)
            input_list[index+6] = ((c_unit.range + tile.terrain.range_mod) / 15.0)
            index += 7

        #Opponent pieces input
        for i in range(len(op_units)):
            c_unit = None
            if map_manager.curr_team == 1:
                c_unit = op_units[len(my_units) - (i + 1)]
            else:
                c_unit = op_units[i]

            if map_manager.curr_team == 1:
                input_list[index] = 1 - (c_unit.pos[0])/(map_manager.Map.shape[0]-1)
                input_list[index+1] = 1 - (c_unit.pos[1])/(map_manager.Map.shape[1]-1)
            else:
                input_list[index] = (c_unit.pos[0])/(map_manager.Map.shape[0]-1)
                input_list[index+1] = (c_unit.pos[1])/(map_manager.Map.shape[1]-1)
            input_list[index+2] = (c_unit.temp_hp / 100.0)

            tile = map_manager.Map[c_unit.pos]
            input_list[index+3] = (c_unit.Att / 100.0)
            input_list[index+4] = (c_unit.Def * tile.terrain.def_mod / 100.0)
            input_list[index+5] = (c_unit.max_move / 15)
            input_list[index+6] = ((c_unit.range + tile.terrain.range_mod) / 15.0)
            index += 7

        #VERY IMPORTANT, must be executed after SIMULATED combat!!!
        if (move.is_attack):
            map_manager.reset_temp_hp() 

        #print("move: " + str(move.pos))

        input_tup = tuple(input_list)
        #format_in = [ '%.2f' % elem for elem in input_tup ]
        #print("input vector: " + str(format_in))
        
        output = net.activate(input_tup)
        
        #format_out = [ '%.2f' % elem for elem in output ]
        #print("output vector: " + str(format_out))

        #print(len(output[0]))
        #print(len(win_weight))
        if (output[0] >= win_weight):
            #print("true")
            win_move = move.pos
            win_weight = output[0]

        unit.pos = saved_pos    #revert back to saved pos
        unit.rotation = saved_rot

    return win_move

def simple_script_ai(map_manager, unit):
    move_list = map_manager.find_movement(unit)
    my_units = []
    op_units = []

    if (map_manager.curr_team == 0): 
        my_units = map_manager.Teams[0].live_units
        op_units = map_manager.Teams[1].live_units
    elif(map_manager.curr_team == 1):
        my_units = map_manager.Teams[1].live_units
        op_units = map_manager.Teams[0].live_units
    
    
    win_move = (0, 0)
    weights = {}
    
    for move in move_list:
        weight = 0.0
        saved_pos = unit.pos    #save unit pos for later
        unit.pos = move.pos
        
        if move.is_attack:      #move is of Tile Class
            #move_pos = move.move_parent.pos
            map_manager.sim_combat(unit, move.unit_ref) 
            #High weight for all attacks
            weight += 25
        
        for op_unit in op_units:
            #Bonus weight for getting near enemies
            weight += 5 / (math.dist(op_unit.pos, unit.pos) + 1)

        if (move.is_attack):
            map_manager.reset_temp_hp() 
        
        weights[move.pos] = weight
        unit.pos = saved_pos    #revert back to saved pos
    
    #print(weights)
    win_move = random.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]
    #print(win_move)
    return win_move    #Tuple


def script_ai(map_manager, unit):
    move_list = map_manager.find_movement(unit)
    my_units = []
    op_units = []

    if (map_manager.curr_team == 0): 
        my_units = map_manager.Teams[0].live_units
        op_units = map_manager.Teams[1].live_units
    elif(map_manager.curr_team == 1):
        my_units = map_manager.Teams[1].live_units
        op_units = map_manager.Teams[0].live_units
    
    
    win_move = (0, 0)
    win_weight = -float("inf")
    
    for move in move_list:
        weight = 0.0
        saved_pos = unit.pos    #save unit pos for later
        unit.pos = move.pos
        
        if move.is_attack:      #move is of Tile Class
            #move_pos = move.move_parent.pos
            map_manager.sim_combat(unit, move.unit_ref) 
            #High weight for all attacks
            weight += 25
        
        for op_unit in op_units:
            #Bonus weight for getting near enemies
            weight += 5 / (math.dist(op_unit.pos, unit.pos) + 1)
        
        # for my_unit in my_units:
        #     if my_unit == unit:
        #         continue
        #     #Bonus weight for staying near friends
        #     weight += 1 / (math.dist(my_unit.pos, unit.pos) + 1)
            
        
        if (move.is_attack):
            map_manager.reset_temp_hp() 
        
        if (weight > win_weight):
            win_move = move.pos
            win_weight = weight
        
        unit.pos = saved_pos    #revert back to saved pos
    
    return win_move    #Tuple

def minimax_ai(map_manager, unit, net, k):
    pass


def sim_game(manager, setup_index, zero_first, my_net, op_net):
    # Feedback for this game
    fdbk_sum = 0

    # Reporting stats for this game
    stats = {'mean_dist':[], 
            'prop_f_all':[],
            'prop_h_all':[],
            'prop_f_i':[],
            'prop_h_i':[],
            'prop_f_a':[],
            'prop_h_a':[],
            'prop_f_c':[],
            'prop_h_c':[],

            'mean_loc_all': [],
            'mean_loc_i':[],
            'mean_loc_a':[],
            'mean_loc_c': [],
            }

    #also resets map
    #manager.setup_rand(units_per_side)
    manager.apply_map_layout(setup_index)
    #Alternate which side starts
    if not zero_first:
        #Note that this makes the turn count end earlier
        manager.curr_team = 1

    while (manager.game_joever() == -1 and manager.turn_count < 20): #Turn Count limit may have to be modified
        for unit in manager.Teams[manager.curr_team].live_units:
            win_move = (0, 0)
            if manager.curr_team == 0:
                win_move = neat_ai(manager, unit, my_net)
                #win_move = move_pick_ai(manager, unit, my_net)
            elif manager.curr_team == 1:
                if op_net is None:
                    win_move = script_ai(manager, unit)
                else:
                    win_move = neat_ai(manager, unit, op_net)
                    #win_move = move_pick_ai(manager, unit, op_net)
            manager.move_unit(unit, win_move)

        update_stats(manager, stats)
        #Next Turn
        manager.Turn()
    
    if (setup_index == 0 and zero_first):
        print("First game result vs: {}".format(op_net))
        print(manager)

    ret_stats = {}
    for key, val in stats.items():
        if type(val[0]) is tuple:
            ret_stats[key] = mean_loc(stats[key])
        else:
            ret_stats[key] = sum(stats[key]) / len(stats[key])

    # if (manager.game_joever() == 0):
    #     fdbk_sum += 1
    if (manager.game_feedback() == 0):
        fdbk_sum += 1
    #fdbk_sum += manager.game_feedback()
    return (fdbk_sum, ret_stats)


def eval_performance(pool, manager, my_net, op_net):
    games_run = len(manager.map_layouts) * 2
    wins = 0

    stats_all = {'mean_dist':[], 
            'prop_f_all':[],
            'prop_h_all':[],
            'prop_f_i':[],
            'prop_h_i':[],
            'prop_f_a':[],
            'prop_h_a':[],
            'prop_f_c':[],
            'prop_h_c':[],

            'mean_loc_all': [],
            'mean_loc_i':[],
            'mean_loc_a':[],
            'mean_loc_c':[],
            }

    input_ls = []
    for i in range(games_run):
        #also resets map
        #manager.setup_rand(units_per_side)
        layout_index = i % len(manager.map_layouts)
        
        zero_first = False
        #Alternate which side starts
        if i >= len(manager.map_layouts):
            #Note that this makes the turn count end earlier
            zero_first = True

        input_ls.append((manager, layout_index, zero_first, my_net, op_net))

    map_tuples = pool.starmap(sim_game, input_ls)
    wins = 0

    for tup in map_tuples:
        wins += tup[0]
        stats = tup[1]
        for key, val in stats.items():
            stats_all[key].append(val)
    
    
    for key, val in stats_all.items():
        if type(val[0]) is tuple:
            stats_all[key] = mean_loc(val)
        else:
            stats_all[key] = sum(val) / len(val)

    #print("stats: {}".format(stats_all))
    winrate = (wins / games_run)
    return winrate, stats_all

def update_stats(manager, stats):
    mean_dist_list = []
    mean_loc_list = []
    mean_loc_a = []
    mean_loc_i = []
    mean_loc_c = []
    count_f_all = 0
    count_f_a = 0
    count_f_i = 0
    count_f_c = 0
    count_h_all = 0
    count_h_a = 0
    count_h_i = 0
    count_h_c = 0

    count_a = 0
    count_i = 0
    count_c = 0

    for unit1 in manager.Teams[0].live_units:
        print(len(manager.Teams[0].live_units))
        if manager.Map[unit1.pos].terrain == 'Forest':
            count_f_all +=1
            if unit1.type == 'archer':
                count_f_a +=1
            elif unit1.type == 'cavalry':
                count_f_c +=1
            elif unit1.type == 'infantry':
                count_f_i +=1
        if manager.Map[unit1.pos].terrain == 'Hills':
            count_h_all +=1
            if unit1.type == 'archer':
                count_h_a +=1
            elif unit1.type == 'cavalry':
                count_h_c +=1
            elif unit1.type == 'infantry':
                count_h_i +=1

        mean_loc_list.append(unit1.pos)
        if unit1.type == 'archer':
            count_a += 1
            mean_loc_a.append(unit1.pos)
        elif unit1.type == 'cavalry':
            count_c += 1
            mean_loc_c.append(unit1.pos)
        elif unit1.type == 'infantry':
            count_i += 1
            mean_loc_i.append(unit1.pos)

        dist_list = []
        for unit2 in manager.Teams[0].live_units:
            if unit1 != unit2:    
                dist_list.append(math.dist(unit1.pos, unit2.pos))
        if len(dist_list) > 0:
            mean_dist_list.append(sum(dist_list) / len(dist_list))

    mean_dist = 0
    if len(mean_dist_list) > 0:
        mean_dist = sum(mean_dist_list) / len(mean_dist_list)

    if mean_dist > 0:
        stats['mean_dist'].append(mean_dist)
    if len(manager.Teams[0].live_units) > 0:
        stats['prop_f_all'].append(count_f_all / len(manager.Teams[0].live_units))
        stats['prop_h_all'].append(count_h_all / len(manager.Teams[0].live_units))
    if count_i > 0:
        stats['prop_f_i'].append(count_f_i / count_i)
        stats['prop_h_i'].append(count_h_i / count_i)
    if count_a > 0:
        stats['prop_f_a'].append(count_f_a / count_a)
        stats['prop_h_a'].append(count_h_a / count_a)
    if count_c > 0:
        stats['prop_f_c'].append(count_f_c / count_c)
        stats['prop_h_c'].append(count_h_c / count_c)
    
    ret = mean_loc(mean_loc_list)
    if ret is not None:
        stats['mean_loc_all'].append(ret)
    
    ret = mean_loc(mean_loc_i)
    if ret is not None:
        stats['mean_loc_i'].append(ret)
    
    ret = mean_loc(mean_loc_a)
    if ret is not None:
        stats['mean_loc_a'].append(ret)
    
    ret = mean_loc(mean_loc_c)
    if ret is not None:
        stats['mean_loc_c'].append(ret)


def mean_loc(arr):
    if len(arr) == 0:
        return None
    sum_x = 0
    sum_y = 0
    for item in arr:
        sum_x += item[1]
        sum_y += item[0]

    x = sum_x / len(arr)
    y = sum_y / len(arr)
    return (y, x)