import numpy as np
import math 
import queue
import os
import neat
import visualize
import random
import matplotlib.pyplot as plt
import pickle
import threading
import copy

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

def thread_eval(manager, games_run, genomes, config, op_net):
    for genome_id, genome in genomes:
        wins = 0
        my_net = neat.nn.FeedForwardNetwork.create(genome, config)    
        
        # for eval_num in range((len(best_genomes_reporter.eval_nets) + 1)):
        #     op_net = None
        #     if eval_num != len(best_genomes_reporter.eval_nets):
        #         op_net = best_genomes_reporter.eval_nets[eval_num]

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
                            if Population.generation < 10:
                                win_move = rand_ai(manager, unit)
                            else:
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

        #print('wins: ' + str(wins))
        #manager.game_feedback()
        genome.fitness = (wins / games_run)

#This is the Eval Func branch
def eval_genomes(genomes, config):
    
    #Self-play
    # Getting best genomes of past x generaations
    global best_genomes_reporter
    op_net = None 
    if len(best_genomes_reporter.eval_nets) > 0:
        op_net = best_genomes_reporter.eval_nets[len(best_genomes_reporter.eval_nets)-1]

    global managers
    #one game for each layout, from each side (times 2)
    # games_run = (len(best_genomes_reporter.eval_nets) + 1) * len(manager.map_layouts) * 2 
    games_run = len(managers[0].map_layouts) * 2 
    genomes_per_thread = len(genomes) // thread_count
    threads = []

    for i in range(thread_count):
        start = i * genomes_per_thread
        end = (i+1) * genomes_per_thread
        if i == thread_count - 1:
            end = len(genomes)
        
        print('start: {}'.format(start))
        print('end: {}'.format(end))
        thread_genomes = genomes[start : end]
        t = threading.Thread(target=thread_eval, args=[managers[i], games_run, thread_genomes, config, op_net])
        t.start()
        threads.append(t)
    
    for thread in threads:
        thread.join()
        print('thread done!')
        
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
    
    gen_interval = 20
    global best_genomes_reporter
    best_genomes_reporter = genome_reporter(generation_interval=gen_interval, run_name=run_name, Population=p)
    p.add_reporter(best_genomes_reporter)

    eval = eval_reporter(20, dimensions, best_genomes_reporter)     #40 (20 x 2)games, 8x8
    p.add_reporter(eval)    

    plot = plot_reporter(generation_interval=gen_interval, stats_reporter=stats, run_name=run_name,
                         save_file=False, eval_reporter=eval, genome_reporter=best_genomes_reporter)
    p.add_reporter(plot)

    #THIS ENABLES CHECKPOINTS
    if not os.path.exists('./checkpoints/{}'.format(run_name)):
        os.makedirs('./checkpoints/{}'.format(run_name))
    checkpoint = neat.Checkpointer(generation_interval=gen_interval,
        filename_prefix='./checkpoints/{}/neat-checkpoint-'.format(run_name))
    p.add_reporter(checkpoint)
    #can use restore_checkpoint to resume simulation

    # Run for up to *generations* generations.
    generations = 250

    #Global manager
    global managers
    managers = []

    #root manager
    root_manager = map_manager(dimensions)
    root_manager.setup_layouts_rand(layout_n=20, unit_count=5)

    #set up each thread with a unique manager that has a copy of the same map layouts
    for i in range(thread_count):
        thread_manager = map_manager(dimensions)
        thread_manager.map_layouts = copy.deepcopy(root_manager.map_layouts)
        managers.append(thread_manager)
        
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