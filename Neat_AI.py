import numpy as np
import math 
import queue
import os
import neat
import visualize
import random

import GameManager
from GameManager import map_manager, Tile, Unit, Team

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        games_run = 20
        wins = 0
        dimensions = (8,8)
        manager = map_manager(dimensions)

        
        for i in range(games_run):
            pos_pick = random.randint(0, 1)
            pos_1 = (0, 0)
            pos_2 = (0, 0)
            if (pos_pick == 0):
                #Team 0 units between 0 and 2 in the x, full length in the y
                pos_1 = (random.randint(0, dimensions[1]-1), random.randint(0, 2))    
                #Team 1 units between 5 and 7 in the x, one half of y      
                pos_2 = (random.randint(0, (dimensions[1]//2)-1), random.randint(dimensions[0]-3, dimensions[0]-1))
                pos_3 = (random.randint(dimensions[1]//2, dimensions[1]-1), random.randint(dimensions[0]-3, dimensions[0]-1))
            elif (pos_pick == 1):
                #Team 0 units between 5 and 7 in the x, full length in the y
                pos_1 = (random.randint(0, dimensions[1]-1), random.randint(dimensions[0]-3, dimensions[0]-1))
                #Team 1 units between 0 and 2 in the x, one half of y
                pos_2 = (random.randint(0, (dimensions[1]//2)-1), random.randint(0, 2))  
                pos_3 = (random.randint(dimensions[1]//2, dimensions[1]-1), random.randint(0, 2))  
                
            manager.reset_map()
            unit1 = manager.place_unit(pos_1, 0)
            unit2 = manager.place_unit(pos_2, 1)
            unit3 = manager.place_unit(pos_3, 1)

            #print("GENOME EVALUATION: ")
            while (manager.game_result() == -1 and manager.turn_count < 7): #Turn Count limit may have to be modified
                move_list = manager.find_movement(unit1)
                
                input_list = list(np.zeros((6)))

                win_move = (0, 0)
                win_weight = -float("inf")
                for move in move_list:
                    #Cannot stay in the same spot: avoid getting trapped in a 'hole'
                    if (move.pos == unit1.pos):
                        pass
                    move_pos = move.pos
                    if move.is_attack:      #move is of Tile Class
                        move_pos = move.move_parent.pos
                        manager.sim_combat(unit1, move.unit_ref) 
                    
                    input_list[0] = (unit2.pos[0] - move.pos[0])/(dimensions[0]-1)  #maybe use move_pos later
                    input_list[1] = (unit2.pos[1] - move.pos[1])/(dimensions[1]-1)  #maybe use move_pos later
                    input_list[2] = (unit2.temp_hp / 100.0) #Current max hp is 100 
                    
                    input_list[3] = (unit3.pos[0] - move.pos[0])/(dimensions[0]-1)  #maybe use move_pos later
                    input_list[4] = (unit3.pos[1] - move.pos[1])/(dimensions[1]-1)  #maybe use move_pos later
                    input_list[5] = (unit3.temp_hp / 100.0) #Current max hp is 100 

                    #VERY IMPORTANT, must be executed after SIMULATED combat!!!
                    if (move.is_attack):
                        manager.reset_temp_hp() 

                    input_tup = tuple(input_list)
                    format_in = [ '%.2f' % elem for elem in input_tup ]
                    #print("input vector: " + str(format_in))
                    
                    output = net.activate(input_tup)
                    
                    format_out = [ '%.2f' % elem for elem in output ]
                    #print("output vector: " + str(format_out))

                    #print(len(output[0]))
                    #print(len(win_weight))
                    if (output[0] >= win_weight):
                        win_move = move.pos
                        win_weight = output[0]

                #print("unit 1 pos: " + str(unit1.pos))
                #print("unit 2 pos: " + str(unit2.pos))
                
                #print("move: " + str(move))
                manager.move_unit(unit1, win_move)

                #print("new_pos: "+ str(unit1.pos))
                manager.turn_count += 1
            
            #if (math.dist(pos1, pos2) == 0):
            #    wins += 1
            if (manager.game_result() == 0):
                wins +=1

            #dist_list.append(math.dist(pos1, pos2))
            
        
        #print("fitness: " + str(wins/games_run))
        genome.fitness = (wins / games_run)
        
        #print("")
        #print("END GENOME EVALUATION: ")

        #print("end unit 1 pos: " + str(pos1))
        #print("end unit 2 pos: " + str(pos2))
        #print("fitness: " + str(5.656854249492381 - (sum(dist_list)/games_run)))
        #print("")
        #print("")

        #genome.fitness = 5.656854249492381 - (sum(dist_list)/games_run)
        
        
def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to *generations* generations.
    winner = p.run(eval_genomes, 100)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    #for xi, xo in zip(xor_inputs, xor_outputs):
    #    output = winner_net.activate(xi)
    #    print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    #node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    #visualize.draw_net(config, winner, True, node_names=node_names)
    #visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    #visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)

    #What does this restore even do????
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    
    #p.run(eval_genomes, 10)
    
    return winner_net