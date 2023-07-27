import numpy as np
import time

def createandsort (n):
 rand = np.random.RandomState(42) #Give a seed to reproduce results
 a = rand.rand(n) #Generate an array of size n
 return a.sort() #Sort the array

def Summation(val):
    return val + 1

def wait(n):
    time.sleep(1)  # Simulating some work with a 1 second delay
    return