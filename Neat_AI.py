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
from AI_modules import no_ai, rand_ai, script_ai
from AI_modules import neat_ai, move_pick_ai, script_performance, neat_performance 

import GameManager
from GameManager import map_manager, Tile, Unit, Team

Population = None

#This is the Eval Func branch
def eval_genomes(genomes, config):
    
    #Self-play
    # Getting best genomes of past x generaations
    global best_genomes_reporter
    op_net = None 
    if len(best_genomes_reporter.eval_nets) > 1:
        op_net = best_genomes_reporter.eval_nets[len(best_genomes_reporter.eval_nets)-1]

    global manager
    games_run = len(manager.map_layouts) * 2 #one game for each layout, from each side (times 2)

    for genome_id, genome in genomes:
        wins = 0
        my_net = neat.nn.FeedForwardNetwork.create(genome, config)    
        
        #sum = 0.0
        for i in range(games_run):
            manager.apply_map_layout(i % len(manager.map_layouts))
            
            #Alternate which side starts
            if i > len(manager.map_layouts):
                #Note that this makes the turn count end earlier
                manager.curr_team = 1

            while (manager.game_joever() == -1 and manager.turn_count < 10): #Turn Count limit may have to be modified
                for unit in manager.Teams[manager.curr_team].live_units:               
                    win_move = (0, 0)
                    if manager.curr_team == 0:
                        #win_move = move_pick_ai(manager, unit, my_net)
                        win_move = neat_ai(manager, unit, my_net)
                    elif manager.curr_team == 1:
                        if op_net is None:
                            win_move = script_ai(manager, unit)
                        else:
                            win_move = neat_ai(manager, unit, op_net)
                        #win_move = move_pick_ai(manager, unit, op_net)
                    manager.move_unit(unit, win_move)
                    #print(manager)
                manager.Turn()
            
            #print(manager)
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
    global Stats
    Stats = stats
    p.add_reporter(stats)
    
    gen_interval = 5
    global best_genomes_reporter
    best_genomes_reporter = genome_reporter(generation_interval=gen_interval, run_name=run_name)
    eval = eval_reporter(25, (8, 8), best_genomes_reporter)     #50 (25 x 2)games, 8x8
    p.add_reporter(eval)
    p.add_reporter(best_genomes_reporter)

    plot = plot_reporter(generation_interval=gen_interval, stats_reporter=stats, run_name=run_name,
                         save_file=False, eval_reporter=eval)
    p.add_reporter(plot)

    #THIS ENABLES CHECKPOINTS
    if not os.path.exists('./checkpoints/{}'.format(run_name)):
        os.makedirs('./checkpoints/{}'.format(run_name))
    checkpoint = neat.Checkpointer(generation_interval=gen_interval,
        filename_prefix='./checkpoints/{}/neat-checkpoint-'.format(run_name))
    p.add_reporter(checkpoint)
    #can use restore_checkpoint to resume simulation

    # Run for up to *generations* generations.
    generations = 100

    #Global manager
    global manager
    dimensions = (8,8)
    manager = map_manager(dimensions)
    manager.setup_layouts_rand(layout_n=gen_interval, unit_count=5)

    winner = p.run(eval_genomes, generations)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    plot_stats(stats, gen_interval, run_name, "./plots/")
    plot_eval_performance(eval, run_name, "./plots/")
    
    winner.write_config('./best/{}-config'.format(run_name), config)
    with open('./best/{}-genome'.format(run_name), "wb") as f:
        pickle.dump(winner, f)
        f.close()
    
    return winner_net, winner, stats

def load_net(config_path, path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    # Unpickle saved winner
    with open(path, "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return net

def plot_stats(stats_reporter, generation_interval, name, path=""):
    plt.plot(stats_reporter.get_fitness_stat(max))
    plt.title("{} Best Fitness vs. Generation".format(name))
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    for x in range(1, len(stats_reporter.get_fitness_stat(max)) // generation_interval):
        plt.axvline(x=(x * generation_interval), color='k')

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
    for x in range(1, len(stats_reporter.get_fitness_stat(max)) // generation_interval):
        plt.axvline(x=(x * generation_interval), color='k')
    
    if path != "":
        filepath = path + "{} Mean Fitness vs. Generation".format(name).replace(' ', '-') + '.jpg'
        if os.path.isfile(filepath):
            os.remove(filepath)
        plt.savefig(filepath)
    plt.show()


def plot_eval_performance(eval_reporter, gen_interval, name, path=""):
    length = len(eval_reporter.eval_performance['script'])
    c = ['b', 'r', 'g', 'c', 'm', 'y']
    ci = 0
    for key, vals in eval_reporter.eval_performance.items():
        print('start: ' + str(ci * gen_interval))
        print('end: ' + str((ci * gen_interval) + len(vals)))
        
        plt.plot(list(range(ci * gen_interval, (ci * gen_interval) + len(vals))),
                vals, color=c[ci % len(c)], label=key)
        ci += 1

    # for x in range(1, len(stats_reporter.get_fitness_stat(max)) // generation_interval):
    #     plt.axvline(x=(x * generation_interval), color='k')
    
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
    def __init__(self, generation_interval, stats_reporter, run_name="", save_file=False, eval_reporter=None):
        self.gen_interval = generation_interval
        self.run_name = run_name
        self.filepath = "./plots/{}".format(run_name)
        self.eval_reporter = eval_reporter
        self.stats_reporter = stats_reporter
        self.save = save_file

        self.gen_count = 0

    def post_evaluate(self, config, population, species, best_genome):
        self.gen_count += 1
        if ((self.gen_count % self.gen_interval) == 0 and self.gen_count > 0):
            if self.save:
                plot_stats(self.stats_reporter, self.gen_interval, self.run_name, "./plots/")
            else:
                plot_stats(self.stats_reporter, self.gen_interval, self.run_name)
            if (self.eval_reporter is not None):
                if self.save:
                    plot_eval_performance(self.eval_reporter, self.gen_interval, self.run_name, "./plots/")
                else:
                    plot_eval_performance(self.eval_reporter, self.gen_interval, self.run_name)

class eval_reporter(neat.reporting.BaseReporter):
    def __init__(self, games_run, dimensions, genome_reporter):
        self.games = games_run
        self.manager = map_manager(dimensions)
        self.manager.setup_layouts_rand(layout_n=games_run, unit_count=5)
        
        self.eval_performance = {}
        self.genome_reporter = genome_reporter

    def post_evaluate(self, config, population, species, best_genome):
        my_net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        
        win_rate = script_performance(self.manager, my_net, config)
        if 'script' in self.eval_performance:
            self.eval_performance['script'].append(win_rate)
        else:
            self.eval_performance['script'] = [win_rate]
        print("Winrate vs. {} : {}".format('script', win_rate))
        
        for i in range(len(self.genome_reporter.eval_nets)):
            op_net = self.genome_reporter.eval_nets[i]

            win_rate = neat_performance(self.manager, my_net, op_net, config)
            if str(i + 1) not in self.eval_performance:
                self.eval_performance[str(i+1)] = [win_rate]
            else:
                self.eval_performance[str(i+1)].append(win_rate)
            print("Winrate vs. {} : {}".format(str(i+1), win_rate))

class genome_reporter(neat.reporting.BaseReporter):
    def __init__(self, generation_interval, run_name):
        self.gen_count = 0
        self.gen_interval = generation_interval
        self.run_name = run_name
        
        self.best_genome = None     #The best genome in each generation
        self.best_pop = None        #The population (dict, not class) with the best genome so far
        self.best_species = None
        self.eval_nets = []

    def post_evaluate(self, config, population, species, best_genome):
        self.gen_count += 1
        if (self.best_genome is None) or (best_genome.fitness > self.best_genome.fitness):
            #print("New best genome: {}".format(best_genome))
            self.best_genome = best_genome
            self.best_pop = population
            self.best_species = species

        if ((self.gen_count % self.gen_interval) == 0 and self.gen_count > 0):
            #print('interval!')
            self.eval_nets.append(neat.nn.FeedForwardNetwork.create(self.best_genome, config))
            #print(self.eval_nets)
            
            #Change population's population to this
            global Population
            #print('Old population: {}'.format(Population.population))
            
            Population.population = self.best_pop
            #print('New population: {}'.format(Population.population))
            Population.species = self.best_species

            self.best_genome = None 
            self.best_pop = None    
            self.best_species = None

        # if ((self.gen_count % self.gen_interval) == 0 and self.gen_count > 0):
        #     best_genome.write_config('./best/{}-config'.format(self.run_name), config)
        #     with open('./best/{}-genome'.format(self.run_name), "wb") as f:
        #         pickle.dump(best_genome, f)
        #         f.close()

    #Returns the best genome from the last n generations
    def return_best(self, n):
        return self.best_genomes[-n:]
