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
    op_net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    
    games_run = 10
    dimensions = (7,7)
    units_per_side = 5
    manager = map_manager(dimensions)

    for genome_id, genome in genomes:
        wins = 0
        my_net = neat.nn.FeedForwardNetwork.create(genome, config)    
        
        #sum = 0.0
        for j in range(games_run):
            #also resets map
            #Rand setup with pick i, also resets map
            manager.setup_rand(units_per_side)

            while (manager.game_joever() == -1 and manager.turn_count < 8): #Turn Count limit may have to be modified
                for unit in manager.Teams[manager.curr_team].live_units:
                    win_move = (0, 0)
                    if manager.curr_team == 0:
                        win_move = neat_ai(manager, unit, my_net)
                    elif manager.curr_team == 1:
                        win_move = neat_ai(manager, unit, op_net)
                    manager.move_unit(unit, win_move)
                #print(manager)
                manager.Turn()
                
            #sum += manager.game_feedback()
            if (manager.game_feedback() == 0):
                wins +=1
            # elif (manager.game_joever() == 1):
            #     wins -= 1

        #print('wins: ' + str(wins))
        #manager.game_feedback()
        genome.fitness = (wins / games_run)
        
def run(config_file, run_name):
    
    run_name = run_name.replace(' ', '-')
    if os.path.exists('./best/{}-genome'.format(run_name)):
        raise NameError('This name, {}, is already used. Delete the previous "./best" and "./checkpoint" files or pick a new run name.'.format(run_name))
    
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

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
    plot = plot_reporter(generation_interval=5, stats_reporter=stats, run_name=run_name,
                         save_file=False, script_reporter=script)
    p.add_reporter(plot)

    #THIS ENABLES CHECKPOINTS
    if not os.path.exists('./checkpoints/{}'.format(run_name)):
        os.makedirs('./checkpoints/{}'.format(run_name))
    checkpoint = neat.Checkpointer(generation_interval=5, filename_prefix='./checkpoints/{}/neat-checkpoint-'.format(run_name))
    p.add_reporter(checkpoint)
    #can use restore_checkpoint to resume simulation

    # Run for up to *generations* generations.
    generations = 10
    winner = p.run(eval_genomes, generations)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    plot_stats(stats, run_name, "./plots/")
    plot_script_performance(script, run_name, "./plots/")
    
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

def plot_stats(stats_reporter, name, path=""):
    plt.plot(stats_reporter.get_fitness_stat(max))
    plt.title("{} Best Fitness vs. Generation".format(name))
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")

    if path != "":
        filepath = path + "{} Best Fitness vs. Generation".format(name).replace(' ', '-') + '.jpg'
        if os.path.isfile(filepath):
            os.remove(filepath)
        plt.savefig(filepath)
    plt.show()

    plt.plot(stats_reporter.get_fitness_mean())
    plt.title("{} Mean Fitness vs. Generation".format(name))
    plt.xlabel("Generation")
    plt.ylabel("Mean Fitness")
    
    if path != "":
        filepath = path + "{} Mean Fitness vs. Generation".format(name).replace(' ', '-') + '.jpg'
        if os.path.isfile(filepath):
            os.remove(filepath)
        plt.savefig(filepath)
    plt.show()

def plot_script_performance(script_reporter, name, path=""):
    plt.plot(script_reporter.winrate_list)
    plt.title("{} Best Genome Winrate vs. Generation".format(name))
    plt.xlabel("Generation")
    plt.ylabel("Best Genome Winrate vs. Script")
    
    if path != "":
        filepath = path + "{} Best Genome Winrate vs. Script".format(name).replace(' ', '-') + '.jpg'
        if os.path.isfile(filepath):
            os.remove(filepath)
        plt.savefig(filepath)
    plt.show()


class plot_reporter(neat.reporting.BaseReporter):
    def __init__(self, generation_interval, stats_reporter, run_name="", save_file=False, script_reporter=None):
        self.gen_interval = generation_interval
        self.run_name = run_name
        self.filepath = "./plots/{}".format(run_name)
        self.script_reporter = script_reporter
        self.stats_reporter = stats_reporter
        self.save = save_file

        self.gen_count = 0

    def post_evaluate(self, config, population, species, best_genome):
        self.gen_count += 1
        if ((self.gen_count % self.gen_interval) == 0 and self.gen_count > 0):
            if self.save:
                plot_stats(self.stats_reporter, self.run_name, "./plots/")
            else:
                plot_stats(self.stats_reporter, self.run_name)
            if (self.script_reporter is not None):
                if self.save:
                    plot_script_performance(self.script_reporter, self.run_name, "./plots/")
                else:
                    plot_script_performance(self.script_reporter, self.run_name)

class vs_script_reporter(neat.reporting.BaseReporter):
    def __init__(self, games_run, dimensions):
        self.winrate_list = []
        self.games = games_run
        self.manager = map_manager(dimensions)

    def post_evaluate(self, config, population, species, best_genome):
        gen_rate = script_performance(self.manager, self.games, best_genome, config)
        self.winrate_list.append(gen_rate)
        print("This generation's best genome's winrate vs. script: {}".format(round(gen_rate, 2)))
              

