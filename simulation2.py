# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import warnings
from functions import plot_trial, plot_sim, save_csv, save_csv2
from scipy import stats
import time


warnings.filterwarnings("ignore")

def increasing(x):
    
    dx = np.diff(x)
    
    return  np.all(dx >= 0)

normed_tan = lambda x: np.tan( x*np.pi/2) + 1

util = lambda scale, x: np.tanh(scale*x)



class accounting:
    
    def __init__(self, init_cash = 0, init_holdings = 0, init_pot = 0):
        
        self.cash = init_cash
        self.holdings = init_holdings
        self.pot = init_pot
        
class simulation:
    
    def __init__(self, )

