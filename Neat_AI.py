import numpy as np
import math 
import os
import neat
import visualize
import random
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool

import AI_modules
from AI_modules import *

import GameManager
from GameManager import *

import my_reporters
from my_reporters import *

# Global variables
thread_count = 8
dimensions = (8, 8)
Population = None
multiprocess_pool = None
managers = []

def thread_eval(config, op_net, genome, manager, is_rand):
    games_run = len(manager.map_layouts) * 2
    wins = 0
    my_net = neat.nn.FeedForwardNetwork.create(genome, config)    

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
                    win_move = move_pick_ai(manager, unit, my_net)
                    #win_move = neat_ai(manager, unit, my_net)
                elif manager.curr_team == 1:
                    if op_net is None:
                        if is_rand:
                            win_move = rand_ai(manager, unit)
                        else:
                            win_move = script_ai(manager, unit)
                    else:
                        #win_move = neat_ai(manager, unit, op_net)
                        win_move = move_pick_ai(manager, unit, op_net)
                manager.move_unit(unit, win_move)
                #print(manager)
            manager.Turn()
        
        #print(manager)
        #sum += manager.game_feedback()
        if (manager.game_feedback() == 0):
            wins +=1

    return (wins / games_run)

#This is the Eval Func branch
def eval_genomes(genomes, config):
    
    #Self-play
    # Getting best genomes of past x generaations
    global best_genomes_reporter

    global op_net
    op_net = None 
    if len(best_genomes_reporter.eval_nets) > 0:
        op_net = best_genomes_reporter.eval_nets[len(best_genomes_reporter.eval_nets)-1]

    global gl_config
    gl_config = config

    global managers
    input_ls = []
    for genome_id, genome in genomes:
        #(config, op_net, genomes, manager, gen_count):
        #change managers[0]
        if len(best_genomes_reporter.gen_intervals) == 0:
            input_ls.append((config, op_net, genome, managers[0], True))
        else:
            input_ls.append((config, op_net, genome, managers[0], False))

    ret_fitness = multiprocess_pool.starmap(thread_eval, input_ls)
    #print("Returned fitness: {}".format(ret_fitness))
    #multiprocess_pool.close()
    #multiprocess_pool.join()

    index = 0
    for genome_id, genome in genomes:
        genome.fitness = ret_fitness[index]
        index += 1

        
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

    global multiprocess_pool
    multiprocess_pool = Pool(thread_count)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    global Stats
    Stats = stats
    p.add_reporter(stats)
    
    max_gen_interval = 100
    int_fit_threshold = 0.95
    global best_genomes_reporter
    best_genomes_reporter = genome_reporter(max_generation_interval=max_gen_interval, 
        run_name=run_name, Population=p, interval_fitness_threshold=int_fit_threshold)
    p.add_reporter(best_genomes_reporter)

    eval = eval_reporter(20, dimensions, best_genomes_reporter)     #40 (20 x 2)games, 8x8
    p.add_reporter(eval)    

    plot = plot_reporter(stats_reporter=stats, run_name=run_name, save_file=False, eval_reporter=eval,
                          genome_reporter=best_genomes_reporter)
    p.add_reporter(plot)

    #THIS ENABLES CHECKPOINTS
    #Note: checkpoints currently not working with other reporters in custom my_reporters file

    if not os.path.exists('./checkpoints/{}'.format(run_name)):
        os.makedirs('./checkpoints/{}'.format(run_name))
    checkpoint = neat.Checkpointer(generation_interval=25,
        filename_prefix='./checkpoints/{}/neat-checkpoint-'.format(run_name))
    p.add_reporter(checkpoint)
    #can use restore_checkpoint to resume simulation

    # Run for up to *generations* generations.
    generations = 500

    #Global manager
    global managers
    #root manager
    root_manager = map_manager(dimensions)
    root_manager.setup_layouts_rand(layout_n=20, unit_count=5)

    #set up each thread with a unique manager that has a copy of the same map layouts
    for i in range(thread_count):
        thread_manager = map_manager(dimensions)
        thread_manager.copy_setup(root_manager)
        managers.append(thread_manager)
        
    winner = p.run(eval_genomes, generations)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    plot_stats(stats, best_genomes_reporter.gen_intervals, run_name, "./plots/")
    plot_eval_performance(eval, run_name, "./plots/")
    
    winner.write_config('./best/{}-config'.format(run_name), config)
    with open('./best/{}-genome'.format(run_name), "wb") as f:
        pickle.dump(winner, f)
        f.close()

    multiprocess_pool.terminate()

    return winner_net, winner, stats

def load_net(config_path, path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    # Unpickle saved winner
    with open(path, "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return net