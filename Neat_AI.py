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
    #Getting best genome
    global Population
    best_genome = Population.best_genome
    #best_genome = Population.reporters[1].best_genome
    if best_genome is None:
        print('defaulting to this genome')
        best_id, best_genome = genomes[0]

    #Population.reporters[1].best_genome    #stats
    
    for genome_id, genome in genomes:
        my_net = neat.nn.FeedForwardNetwork.create(genome, config)
        op_net = neat.nn.FeedForwardNetwork.create(best_genome, config)

        games_run = 10
        wins = 0
        dimensions = (7,7)
        units_per_side = 5
        manager = map_manager(dimensions)

        sum = 0.0

        for i in range(games_run):
            #also resets map
            manager.setup_even(dimensions, units_per_side)

            while (manager.game_joever() == -1 and manager.turn_count < 8): #Turn Count limit may have to be modified
                for unit in manager.Teams[manager.curr_team].live_units:
                    win_move = (0, 0)
                    if manager.curr_team == 0:
                        win_move = neat_ai(manager, unit, my_net)
                    elif manager.curr_team == 1:
                        win_move = script_ai(manager, unit)
                    manager.move_unit(unit, win_move)
                #print(manager)
                manager.Turn()
                
            sum += manager.game_feedback()
                # if (manager.game_joever() == 0):
                #     wins +=1
                # elif (manager.game_joever() == 1):
                #     wins -= 1

        #print('wins: ' + str(wins))
        #manager.game_feedback()
        feedback = (sum / games_run)
        genome.fitness = feedback
        
def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    run_name = 'Global-Position-Eval-Func_vsScript_difference_2layer'

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    global Population
    Population = p

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    script = vs_script_reporter(25, (7, 7))     #25 games, 7x7
    p.add_reporter(script)

    #THIS ENABLES CHECKPOINTS

    if not os.path.exists('./checkpoints/{}'.format(run_name)):
        os.makedirs('./checkpoints/{}'.format(run_name))
    checkpoint = neat.Checkpointer(generation_interval=5, filename_prefix='./checkpoints/{}/neat-checkpoint-'.format(run_name))
    p.add_reporter(checkpoint)
    #can use restore_checkpoint to resume simulation

    # Run for up to *generations* generations.
    generations = 100
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
              

