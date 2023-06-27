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
        
        games_run = 5
        dist_list = []
        dimensions = (5,5)
        move_dist = 4

        for i in range(games_run):
            pos_pick = random.randint(0, 3)
            pos_1 = [0, 0]
            pos_2 = [0, 0]
            if (pos_pick == 0):
                pos_1 = [0, 0]
                pos_2 = [4, 4]
            elif (pos_pick == 1):
                pos_1 = [4, 0]
                pos_2 = [0, 4]
            elif (pos_pick == 2):
                pos_1 = [0, 4]
                pos_2 = [4, 0]
            elif(pos_pick == 3):
                pos_1 = [4, 4]
                pos_2 = [0, 0]

            pos1 = tuple(pos_1)
            pos2 = tuple(pos_2)
            dist = math.dist(pos1, pos2)
            
            #print("GENOME EVALUATION: ")

            #dist_list = []
            turns = 0
            while (dist > 0.0 and turns < 2): #Turn Count limit may have to be modified
                #print("unit 1 pos: " + str(pos1))
                #print("unit 2 pos: " + str(pos2))
                
                input_list = []
                input_list.append(pos1[0]/(dimensions[0]-1))
                input_list.append(pos1[1]/(dimensions[1]-1))

                input_list.append(pos2[0]/(dimensions[0]-1))
                input_list.append(pos2[1]/(dimensions[1]-1))

                #print("input pos: " + str(input_list))

                input_tup = tuple(input_list)
                format_in = [ '%.2f' % elem for elem in input_tup ]
                #print("input vector: " + str(format_in))

                #use output dimensions
                output = net.activate(input_tup)
                
                format_out = [ '%.2f' % elem for elem in output ]
                #print("output vector: " + str(format_out))
                
                move = [round(output[0] * move_dist), round(output[1] * move_dist)]
                #move = (max(min(round(output[0] * unit1.max_move), dimensions[0]-1-unit1.pos[0]), -unit1.pos[0]),
                #        max(min(round(output[1] * unit1.max_move), dimensions[1]-1-unit1.pos[1]), -unit1.pos[1]))
                #print("move: " + str(move))
                new_pos = list(np.add(pos1, move))
                if (new_pos[0] >= dimensions[0]):
                    new_pos[0] = dimensions[0]-1
                if (new_pos[1] >= dimensions[1]):
                    new_pos[1] = dimensions[1]-1

                if (new_pos[0] < 0):
                    new_pos[0] = 0
                if (new_pos[1] < 0):
                    new_pos[1] = 0
                pos1 = tuple(new_pos)

                #print("new_pos: "+ str(pos1))
                turns += 1
            dist_list.append(math.dist(pos1, pos2))
            
            
        #genome.fitness = (wins / games_run)
        
        #print("")
        #print("END GENOME EVALUATION: ")

        #print("end unit 1 pos: " + str(pos1))
        #print("end unit 2 pos: " + str(pos2))
        #print("fitness: " + str(5.656854249492381 - (sum(dist_list)/games_run)))
        #print("")
        #print("")
        genome.fitness = 5.656854249492381 - (sum(dist_list)/games_run)
        
        
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