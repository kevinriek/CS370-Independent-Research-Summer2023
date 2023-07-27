# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:32:18 2023

@author: kevin
"""

import numpy as np
import math 
import queue
import os
import neat
import visualize
import random
import matplotlib.pyplot as plt
import pickle

import GameManager
from GameManager import *

import Neat_AI
from Neat_AI import *

import AI_modules
from AI_modules import *

import my_reporters
from my_reporters import *

local_dir = os.path.abspath('')
config_path = os.path.join(local_dir, 'config-feedforward')

random.seed(4123) #This is to set the random generated board starting positions
if __name__ == '__main__':
    win_net, win_genome, stats = run(config_path, "New Self-Play test")