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
        #genome.fitness = 1.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        games_run = 1
        wins = 0
        
        for i in range(games_run):
            dimensions = (4,4)
            Map = map_manager(dimensions)
            
            pos1 = (0, 0)
            unit1 = Map.place_unit(pos1, 0)
            
            pos2 = (0, 0) 
            while (unit1.pos == pos2):
                pos2 = (random.randint(0, Map.Map.shape[0]-1), random.randint(0, Map.Map.shape[1]-1)) 
            
            unit2 = Map.place_unit(pos2, 1)
            #print("unit 1 pos: " + str(unit1.pos))
            #print("unit 2 pos: " + str(unit2.pos))

            while (Map.Game_result() == -1 and Map.turn_count < 2): #Turn Count limit may have to be modified
                input_list = []
                input_list.append(pos1[0])
                input_list.append(pos1[1])
                #input_list.append(pos1[0]/(dimensions[0]-1))
                #input_list.append(pos1[1]/(dimensions[1]-1))

                for row in Map.Map:
                    for tile in row:
                        if (tile.unit_ref is not None and tile.unit_ref.Team == 1):
                            input_list.append(1.0)
                        else:
                            input_list.append(0.0)
                input_tup = tuple(input_list)
                format_in = [ '%.2f' % elem for elem in input_tup ]
                print("input vector: " + str(format_in))
                #find out how to dictate output dimensions

                #use output dimensions
                output = net.activate(input_tup)
                
                format_out = [ '%.2f' % elem for elem in output ]
                print("output vector: " + str(format_out))
                #move_index = np.argmax(output) 
                #print("move index: " + str(move_index))
                Map.find_movement(unit1)
                move = (min(int(output[0] * unit1.max_move), (dimensions[0]-1)),
                        min(int(output[1] * unit1.max_move), (dimensions[1]-1)))
                print("move: " + str(move))
                
                Map.move_unit(unit1, move)

                #Back to computer's turn
                Map.Turn()
                Map.Turn()
            
            if (Map.Game_result() == 0):
                wins += 1
            
        #Figure out Fitness function
        genome.fitness = (wins / games_run)

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
    winner = p.run(eval_genomes, 300)

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