import matplotlib.pyplot as plt
import os
import neat
from multiprocessing import Pool

from AI_modules import *
from GameManager import *

def plot_stats(stats_reporter, generation_intervals, name, path=""):
    plt.plot(stats_reporter.get_fitness_stat(max))
    plt.title("{} Best Fitness vs. Generation".format(name))
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    for x in generation_intervals:
        plt.axvline(x=x, color='k')

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
    for x in generation_intervals:
        plt.axvline(x=x, color='k')
    
    if path != "":
        filepath = path + "{} Mean Fitness vs. Generation".format(name).replace(' ', '-') + '.jpg'
        if os.path.isfile(filepath):
            os.remove(filepath)
        plt.savefig(filepath)
    plt.show()


def plot_eval_performance(eval_reporter, gen_intervals, name, path=""):
    c = ['b', 'r', 'g', 'c', 'm', 'y']
    ci = 0
    for key, vals in eval_reporter.eval_performance.items():
        # if ci == 0 or ci >= len(gen_intervals):
        #     x_vals = list(range(len(vals)))
        # else:
        #     x_vals = [x + gen_intervals[ci] for x in list(range(len(vals)))]
        if key == 'script':
            x_vals = list(range(len(vals)))
            plt.plot(x_vals, vals, color=c[(ci) % len(c)], label=key)
        else:
            x_vals = [x + gen_intervals[ci] for x in list(range(len(vals)))]
            plt.plot(x_vals, vals, color=c[(ci+1) % len(c)], label=key)
            #plt.plot(vals, color=c[(ci-1) % len(c)], label=key)
            ci += 1

    for x in gen_intervals:
        plt.axvline(x=x, color='k')
    
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
    def __init__(self, stats_reporter, run_name="", save_file=False,
                  eval_reporter=None, genome_reporter=None):
        self.run_name = run_name
        self.filepath = "./plots/{}".format(run_name)
        self.eval_reporter = eval_reporter
        self.stats_reporter = stats_reporter
        self.save = save_file
        self.gen_reporter = genome_reporter

    def post_evaluate(self, config, population, species, best_genome):
        if (self.gen_reporter.is_interval):
            if self.save:
                plot_stats(self.stats_reporter, self.gen_reporter.gen_intervals, self.run_name, "./plots/")
            else:
                plot_stats(self.stats_reporter, self.gen_reporter.gen_intervals, self.run_name)
            if (self.eval_reporter is not None):
                if self.save:
                    plot_eval_performance(self.eval_reporter, self.gen_reporter.gen_intervals, self.run_name, "./plots/")
                else:
                    plot_eval_performance(self.eval_reporter, self.gen_reporter.gen_intervals, self.run_name)

class eval_reporter(neat.reporting.BaseReporter):
    def __init__(self, games_run, dimensions, genome_reporter, thread_count):
        self.games = games_run
        self.manager = map_manager(dimensions)
        self.manager.setup_layouts_rand(layout_n=games_run, unit_count=12)
        
        self.eval_performance = {}
        self.eval_stats = {}
        self.genome_reporter = genome_reporter
        self.pool = Pool(thread_count)

    def post_evaluate(self, config, population, species, best_genome):
        #if (self.genome_reporter.is_interval and len(self.genome_reporter.gen_intervals) > 1):
        my_net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        
        win_rate, stats = eval_performance(self.pool, self.manager, my_net, None)
        if 'script' in self.eval_performance:
            self.eval_performance['script'].append(win_rate)
            self.eval_stats['script'].append(stats)
        else:
            self.eval_performance['script'] = [win_rate]
            self.eval_stats['script'] = [stats]
        print("Best genome Winrate vs. {} : {:0.2f}".format('script', win_rate))
        for key, val in stats.items():
            print("Best genome {} vs. {} : {:0.2f}".format(key, 'script', val))

        print("###############################################")
        for i in range(len(self.genome_reporter.eval_nets)):
            op_net = self.genome_reporter.eval_nets[i]

            win_rate, stats = eval_performance(self.pool, self.manager, my_net, op_net)
            if str(i + 1) not in self.eval_performance:
                self.eval_performance[str(i+1)] = [win_rate]
                self.eval_stats[str(i+1)] = [stats]
            else:
                self.eval_performance[str(i+1)].append(win_rate)
                self.eval_stats[str(i+1)].append(stats)
            print("Best genome Winrate vs. {} : {:0.2f}".format(str(i+1), win_rate))
            for key, val in stats.items():
                print("Best genome {} vs. {} : {:0.2f}".format(key, str(i+1), val))
            print("###############################################")

class genome_reporter(neat.reporting.BaseReporter):
    def __init__(self, max_generation_interval, run_name, Population, interval_fitness_threshold, networks_list):
        self.gen_count = -1
        self.run_name = run_name
        self.eval_nets = networks_list

        self.max_gen_interval = max_generation_interval
        self.int_fit_threshold = interval_fitness_threshold
        self.gen_intervals = []
        self.Population = Population

        self.is_interval = False

    def post_evaluate(self, config, population, species, best_genome):
        self.gen_count += 1
        self.is_interval = False

        if ( best_genome.fitness == 1.0 or  #For initial random phase, always want that to reach 100% fitness
            (best_genome.fitness > self.int_fit_threshold and len(self.gen_intervals) != 0) or #For any other phase, reach the fitness threshold
            self.gen_count == self.max_gen_interval): #If we reach the maximum number of generations in an interval
            
            self.is_interval = True

            # # >0 to ignore the first vs. random iteration 
            # if len(self.gen_intervals) > 0:
            self.eval_nets.append(neat.nn.FeedForwardNetwork.create(best_genome, config))
            # Keep the number of eval nets at 5
            if len(self.eval_nets) > 5:
                self.eval_nets.remove(self.eval_nets[0])

            if (len(self.gen_intervals) == 0):
                self.gen_intervals.append(self.gen_count)
            else:
                self.gen_intervals.append(self.gen_count + self.gen_intervals[-1])

            self.gen_count = 0

        # if ((self.gen_count % self.gen_interval) == 0 and self.gen_count > 0):
        #     best_genome.write_config('./best/{}-config'.format(self.run_name), config)
        #     with open('./best/{}-genome'.format(self.run_name), "wb") as f:
        #         pickle.dump(best_genome, f)
        #         f.close()

       

    #Returns the best genome from the last n generations
    def return_best(self, n):
        return self.best_genomes[-n:]