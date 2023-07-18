import numpy as np
import math 
import queue
import os
import neat
import visualize
import random
import matplotlib.pyplot as plt
import pickle

import AI_modules
from AI_modules import no_ai, rand_ai, script_ai, neat_ai, script_performance

import GameManager
from GameManager import map_manager, Tile, Unit, Team

Population = None

#This is the Eval Func branch
def eval_genomes(genomes, config):
    
    #Self-play
    # #Getting best genome
    # global Population
    # best_genome = Population.best_genome
    # if best_genome is None:
    #     print('defaulting to this genome')
    #     best_id, best_genome = genomes[0]
    
    for genome_id, genome in genomes:
        my_net = neat.nn.FeedForwardNetwork.create(genome, config)
        #op_net = neat.nn.FeedForwardNetwork.create(best_genome, config)

        games_run = 3
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

                #print("new_pos: "+ str(unit1.pos))
                
                #Next Turn
                manager.Turn()
            
            #if (math.dist(pos1, pos2) == 0):
            #    wins += 1
            if (manager.game_result() == 0):
                #print(manager)
                #print("{}, {}, {}, {}, {}, {}".format(pos_1, pos_2, pos_3, pos_4, pos_5, pos_6))
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
    run_name = 'Relative-Position-Eval-Func'

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    global Population
    Population = p

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    script = vs_script_reporter(20, (8, 8))
    p.add_reporter(script)

    #THIS ENABLES CHECKPOINTS

    if not os.path.exists('./checkpoints/{}'.format(run_name)):
        os.makedirs('./checkpoints/{}'.format(run_name))
    checkpoint = neat.Checkpointer(generation_interval=5, filename_prefix='./checkpoints/{}/neat-checkpoint-'.format(run_name))
    p.add_reporter(checkpoint)
    #can use restore_checkpoint to resume simulation

    # Run for up to *generations* generations.
    generations = 1
    winner = p.run(eval_genomes, generations)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    plot_stats(stats, run_name)
    plot_script_performance(script, run_name)
    
    winner.write_config('./best/{}-config'.format(run_name), config)
    with open('./best/{}-genome'.format(run_name), "wb") as f:
        pickle.dump(winner, f)
        f.close()
    
    return winner_net, stats

def load_net(config_path, path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    # Unpickle saved winner
    with open(path, "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return net

def plot_stats(stats_reporter, name):
    plt.plot(stats_reporter.get_fitness_stat(max))
    plt.title("{} Best Fitness vs. Generation".format(name))
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.show()

    plt.plot(stats_reporter.get_fitness_mean())
    plt.title("{} Mean Fitness vs. Generation".format(name))
    plt.xlabel("Generation")
    plt.ylabel("Mean Fitness")
    plt.show()

def plot_script_performance(script_reporter, name):
    plt.plot(script_reporter.winrate_list)
    plt.title("{} Best Genome Winrate vs. Generation".format(name))
    plt.xlabel("Generation")
    plt.ylabel("Best Genome Winrate vs. Script")
    plt.show()

class vs_script_reporter(neat.reporting.BaseReporter):
    def __init__(self, games_run, dimensions):
        self.winrate_list = []
        self.dimensions = dimensions
        self.games = games_run

    def post_evaluate(self, config, population, species, best_genome):
        gen_rate = script_performance(self.games, best_genome, config, self.dimensions)
        self.winrate_list.append(gen_rate)
        print("This generation's best genome's winrate vs. script: {}".format(round(gen_rate, 2)))
              

