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
    input_list = list(np.zeros((22)))

    win_move = (0, 0)
    win_weight = -float("inf")
    for move in move_list:
        #Cannot stay in the same spot: avoid getting trapped in a 'hole'
        if (move.pos == unit.pos):
            pass
        saved_pos = unit.pos    #save unit pos for later
        unit.pos = move.pos

        if move.is_attack:      #move is of Tile Class
            move_pos = move.move_parent.pos
            map_manager.sim_combat(unit, move.unit_ref) 

        index = 0
        #This unit's input is always first
        input_list[index] = (unit.temp_hp / 100.0)
        index += 1

        #Allied pieces input
        for i in range(len(my_units)):
            c_unit = my_units[i]
            if c_unit == unit:  #Input for selected unit entered at beginning
                continue
            input_list[index] = (c_unit.pos[0] - move.pos[0])/(map_manager.Map.shape[0]-1)
            input_list[index+1] = (c_unit.pos[1] - move.pos[1])/(map_manager.Map.shape[1]-1)
            input_list[index+2] = (c_unit.temp_hp / 100.0)
            index += 3

        #Opponent pieces input
        for i in range(len(op_units)):
            c_unit = op_units[i]
            input_list[index] = (c_unit.pos[0] - move.pos[0])/(map_manager.Map.shape[0]-1)
            input_list[index+1] = (c_unit.pos[1] - move.pos[1])/(map_manager.Map.shape[1]-1)
            input_list[index+2] = (c_unit.temp_hp / 100.0)
            index += 3

        #VERY IMPORTANT, must be executed after SIMULATED combat!!!
        if (move.is_attack):
            map_manager.reset_temp_hp() 

        #print("move: " + str(move.pos))

        input_tup = tuple(input_list)
        format_in = [ '%.2f' % elem for elem in input_tup ]
        #print("input vector: " + str(format_in))
        
        output = net.activate(input_tup)
        
        format_out = [ '%.2f' % elem for elem in output ]
        #print("output vector: " + str(format_out))

        #print(len(output[0]))
        #print(len(win_weight))
        if (output[0] >= win_weight):
            #print("true")
            win_move = move.pos
            win_weight = output[0]

        unit.pos = saved_pos    #revert back to saved pos

    return win_move

def script_ai(map_manager, unit):
    move_list = map_manager.find_movement(unit)
    my_units = []
    op_units = []

    if (map_manager.curr_team == 0): 
        my_units = map_manager.Teams[0].units
        op_units = map_manager.Teams[1].units
    elif(map_manager.curr_team == 1):
        my_units = map_manager.Teams[1].units
        op_units = map_manager.Teams[0].units
    
    
    win_move = (0, 0)
    win_weight = -float("inf")
    
    for move in move_list:
        weight = 0.0
        saved_pos = unit.pos    #save unit pos for later
        unit.pos = move.pos
        
        if move.is_attack:      #move is of Tile Class
            move_pos = move.move_parent.pos
            map_manager.sim_combat(unit, move.unit_ref) 
            #High weight for all attacks
            weight += 25
        
        for op_unit in op_units:
            #Bonus weight for getting near enemies
            weight += 3 / (math.dist(op_unit.pos, unit.pos) + 1)
        
        for my_unit in my_units:
            if my_unit == unit:
                continue
            #Bonus weight for staying near friends
            weight += 1 / (math.dist(my_unit.pos, unit.pos) + 1)
            
        
        if (move.is_attack):
            map_manager.reset_temp_hp() 
        
        if (weight > win_weight):
            win_move = move.pos
            win_weight = weight
        
        unit.pos = saved_pos    #revert back to saved pos
    
    return win_move    #Tuple


def script_performance(games, best_genome, config, dimensions):
    my_net = neat.nn.FeedForwardNetwork.create(best_genome, config)

    games_run = games
    wins = 0
    dimensions = (8,8)
    manager = map_manager(dimensions)

    for i in range(games_run):
        pos_pick = random.randint(0, 1)
        pos_list = []
        for i in range(4):
            pos_list.append((i+2, 0))
        for i in range(4):
            pos_list.append((i+2, dimensions[1]-1))

        manager.reset_map()
        units0 = []
        units1 = []
        if pos_pick == 0:
            for i in range(4):
                units0.append(manager.place_unit(pos_list[i], 0))
            for i in range(4, 8):
                units1.append(manager.place_unit(pos_list[i], 1))
        else:
            for i in range(4):
                units1.append(manager.place_unit(pos_list[i], 1))
            for i in range(4, 8):
                units0.append(manager.place_unit(pos_list[i], 0))

        my_units = []
        op_units = []
        #print("GENOME EVALUATION: ")
        while (manager.game_result() == -1 and manager.turn_count < 8): #Turn Count limit may have to be modified

            for unit in manager.Teams[manager.curr_team].units:
                win_move = (0, 0)
                if manager.curr_team == 0:
                    win_move = neat_ai(manager, unit, my_net)
                elif manager.curr_team == 1:
                    win_move = script_ai(manager, unit)
                manager.move_unit(unit, win_move)
                #print(manager)

            
            #Next Turn
            manager.Turn()
        
        #if (math.dist(pos1, pos2) == 0):
        #    wins += 1
        if (manager.game_result() == 0):
            #print(manager)
            #print("{}, {}, {}, {}, {}, {}".format(pos_1, pos_2, pos_3, pos_4, pos_5, pos_6))
            wins +=1

        #dist_list.append(math.dist(pos1, pos2))

    winrate = (wins / games_run)
    return winrate