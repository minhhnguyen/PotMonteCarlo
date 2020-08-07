# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from random import random

ftse_100_daily_return = 0.094/252
ftse_100_daily_sd = 0.1041/np.sqrt(252)

dt = 1/252
initial_investment = 10000
mu = ftse_100_daily_return
sigma = ftse_100_daily_sd

inverse_etf = np.zeros((1,252)).flatten()
inverse_etf[0] = initial_investment

sim_etf = initial_investment*np.exp( np.cumsum((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*np.random.normal(0,1,252)))

d_sim_etf = (sim_etf[1:] - sim_etf[:-1])/sim_etf[:-1]

for i,ret in enumerate(d_sim_etf):
    
    inverse_etf[i+1] = (1  - ret)*inverse_etf[i]

plt.plot(sim_etf)
plt.plot(inverse_etf)
plt.legend(['ETF','Inverse ETF'])



