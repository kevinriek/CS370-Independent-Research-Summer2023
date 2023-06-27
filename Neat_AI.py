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
        games_run = 10
        wins = 0
        
        #for i in range(games_run):
        dimensions = (5,5)
        Map = map_manager(dimensions)
        
        pos1 = (random.randint(0, Map.Map.shape[0]-1), random.randint(0, Map.Map.shape[1]-1))
        unit1 = Map.place_unit(pos1, 0)
        
        pos2 = (random.randint(0, Map.Map.shape[0]-1), random.randint(0, Map.Map.shape[1]-1))
        while (unit1.pos == pos2):
            pos2 = (random.randint(0, Map.Map.shape[0]-1), random.randint(0, Map.Map.shape[1]-1)) 
        
        unit2 = Map.place_unit(pos2, 1)
        #print("GENOME EVALUATION: ")

        
        #dist_list = []
        while (Map.Game_result() == -1 and Map.turn_count < 1): #Turn Count limit may have to be modified
            #print("unit 1 pos: " + str(unit1.pos))
            #print("unit 2 pos: " + str(unit2.pos))
            
            input_list = []
            input_list.append(unit1.pos[0]/(dimensions[0]-1))
            input_list.append(unit1.pos[1]/(dimensions[1]-1))

            input_list.append(unit2.pos[0]/(dimensions[0]-1))
            input_list.append(unit2.pos[1]/(dimensions[1]-1))

            #print("input pos: " + str(input_list))
            """
            for row in Map.Map:
                for tile in row:
                    if (tile.unit_ref is not None and tile.unit_ref.Team == 1):
                        input_list.append(1.0)
                    else:
                        input_list.append(0.0)
            """

            input_tup = tuple(input_list)
            format_in = [ '%.2f' % elem for elem in input_tup ]
            #print("input vector: " + str(format_in))
            #find out how to dictate output dimensions

            #use output dimensions
            output = net.activate(input_tup)
            
            format_out = [ '%.2f' % elem for elem in output ]
            #print("output vector: " + str(format_out))
            #move_index = np.argmax(output) 
            """
            print("move index: " + str(move_index))
            print("max move: " + str(unit1.max_move))
            print("dimensions: " + str(dimensions))
            print("adjusted move x:" + str(round((output[1] * unit1.max_move))))
            print("unit position  x:" + str(unit1.pos[1]))
            
            print("adjusted move y:" + str(round((output[0] * unit1.max_move))))
            print("unit position  y:" + str(unit1.pos[0]))
            """
            Map.find_movement(unit1)
            move = (max(min(round(output[0] * unit1.max_move), dimensions[0]-1-unit1.pos[0]), -unit1.pos[0]),
                    max(min(round(output[1] * unit1.max_move), dimensions[1]-1-unit1.pos[1]), -unit1.pos[1]))
            #print("move: " + str(move))

            Map.move_unit(unit1, move)
            #print("new_pos: "+ str(unit1.pos))

            #Back to computer's turn
            Map.Turn()
            Map.Turn()
            
            if (Map.Game_result() == 0):
                wins += 1
            #dist_list.append()
            
        #Figure out Fitness function
        #genome.fitness = (wins / games_run)
        
        
        #print("end unit 1 pos: " + str(unit1.pos))
        #print("end unit 2 pos: " + str(unit2.pos))
        #print("fitness: " + str(5.65685 - math.dist(unit1.pos, unit2.pos)))
        #print("")
        genome.fitness = 5.65685 - math.dist(unit1.pos, unit2.pos)
        
        
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
    winner = p.run(eval_genomes, 5000)

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