import matplotlib.pyplot as plt
import os
import neat

import AI_modules
from AI_modules import no_ai, rand_ai, script_ai
from AI_modules import neat_ai, move_pick_ai, script_performance, neat_performance 

import GameManager
from GameManager import map_manager, Tile, Unit, Team

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
        plt.plot(list(range(ci * gen_interval, (ci * gen_interval) + (gen_interval * len(vals)), gen_interval)),
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
    def __init__(self, generation_interval, stats_reporter, run_name="", save_file=False,
                  eval_reporter=None, genome_reporter=None):
        self.gen_interval = generation_interval
        self.run_name = run_name
        self.filepath = "./plots/{}".format(run_name)
        self.eval_reporter = eval_reporter
        self.stats_reporter = stats_reporter
        self.save = save_file
        self.gen_reporter = genome_reporter

    def post_evaluate(self, config, population, species, best_genome):
        if ((self.gen_reporter.gen_count % self.gen_interval) == 0 and 
                self.gen_reporter.gen_count > 0):
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
        if (self.genome_reporter.gen_count % self.genome_reporter.gen_interval == 0 and
            self.genome_reporter.gen_count > 0):
            my_net = neat.nn.FeedForwardNetwork.create(best_genome, config)
            
            win_rate = script_performance(self.manager, my_net, config)
            if 'script' in self.eval_performance:
                self.eval_performance['script'].append(win_rate)
            else:
                self.eval_performance['script'] = [win_rate]
            print("Best genome Winrate vs. {} : {}".format('script', win_rate))
            
            for i in range(len(self.genome_reporter.eval_nets)-1):
                op_net = self.genome_reporter.eval_nets[i]

                win_rate = neat_performance(self.manager, my_net, op_net, config)
                if str(i + 1) not in self.eval_performance:
                    self.eval_performance[str(i+1)] = [win_rate]
                else:
                    self.eval_performance[str(i+1)].append(win_rate)
                print("Best genome Winrate vs. {} : {}".format(str(i+1), win_rate))

class genome_reporter(neat.reporting.BaseReporter):
    def __init__(self, generation_interval, run_name, Population):
        self.gen_count = -1
        self.gen_interval = generation_interval
        self.run_name = run_name
        
        self.best_genome = None     #The best genome in each generation
        self.best_pop = None        #The population (dict, not class) with the best genome so far
        self.best_species = None
        self.eval_nets = []

        self.Population = Population
        self.rand_phase = True

    def post_evaluate(self, config, population, species, best_genome):
        self.gen_count += 1
        if (self.best_genome is None) or (best_genome.fitness > self.best_genome.fitness):
            #print("New best genome: {}".format(best_genome))
            self.best_genome = best_genome
            self.best_pop = population
            self.best_species = species

        #For random phase, of 10 iterations
        if (self.gen_count == 10 and self.rand_phase == True):
            self.gen_count = 0
            self.rand_phase = False

            self.Population.population = self.best_pop
            self.Population.species = self.best_species

            self.best_genome = None 
            self.best_pop = None    
            self.best_species = None

        if ((self.gen_count % self.gen_interval) == 0 and self.gen_count > 0):
            self.eval_nets.append(neat.nn.FeedForwardNetwork.create(self.best_genome, config))
            #print(self.eval_nets)
            
            #Change population's population to this
            self.Population.population = self.best_pop
            self.Population.species = self.best_species

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