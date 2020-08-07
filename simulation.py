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

class user_change():
    
    def __init__(self,
                 total_users,
                 daily_return,
                 daily_sd,
                 return_factor,
                 sd_factor,
                 util_scale,
                 base_user_change_rate,
                 eq_counter_cyclical_rate,
                 eq_retaining_rate):
        
#        change_rate = util(util_scale, np.abs(daily_return) )
        
        change_rate = np.random.beta(1000*base_user_change_rate*return_factor*sd_factor, 1000*(1 - base_user_change_rate))
    
        cyclical_change = math.ceil(change_rate*total_users)
        counter_cyclical_change = int(np.random.beta(eq_counter_cyclical_rate*200,200*(1 - eq_counter_cyclical_rate)*return_factor*sd_factor)*cyclical_change)
        
        if daily_return >= 0:
            
            in_change = return_factor*cyclical_change
            
            out_change = sd_factor*counter_cyclical_change
        
        elif daily_return < 0:
            
            in_change = sd_factor*counter_cyclical_change
            
            out_change = (1 - eq_retaining_rate)*cyclical_change/return_factor
            
        else:
            
            in_change = np.random.beta(sd_factor*100,100)*cyclical_change
            
            out_change = np.random.beta(sd_factor*100,100)*cyclical_change
    
        self.inflow = int(in_change)
        self.outflow = int(out_change)



class day():
    
    def __init__(self,
                 total_pot,
                 total_users,
                 daily_return,
                 daily_sd,
                 util_scale,
                 transaction_fee,
                 wealth_effect_bias,
                 income_effect_bias,
                 eq_retaining_rate,
                 eq_counter_cyclical_rate,
                 base_user_change_rate,
                 eq_investment_rate,
                 eq_collection_rate,
                 eq_avg_investment,
                 eq_avg_collection_pct):
        
        sd_factor = normed_tan(wealth_effect_bias*daily_sd)
        return_factor = normed_tan(income_effect_bias*np.abs(daily_return))
        
        
        user_count_change = user_change(total_users = total_users,
                                        daily_return = daily_return,
                                        daily_sd = daily_sd,
                                        sd_factor = sd_factor,
                                        return_factor = return_factor,
                                        util_scale = util_scale,
                                        eq_retaining_rate = eq_retaining_rate,
                                        base_user_change_rate = base_user_change_rate,
                                        eq_counter_cyclical_rate = eq_counter_cyclical_rate)
        
        avg_pot = total_pot/total_users
        
        action_rate = util(util_scale, daily_return*sd_factor)
        
        if daily_return >= 0:
            
            self.investment_rate = action_rate
            self.avg_investment_amt = np.random.gamma(return_factor,sd_factor*eq_avg_investment) 
            
            self.collection_rate = np.random.beta(eq_collection_rate*sd_factor*1000,1000*(1 - eq_collection_rate)*return_factor) 
            self.avg_collection_amt = np.random.beta(return_factor*sd_factor*eq_avg_collection_pct*1000,1000*(1 - eq_avg_collection_pct))*avg_pot 
        
        else:
                        
            self.investment_rate = np.random.beta(eq_investment_rate*1000,1000*(1 - eq_investment_rate)*return_factor*sd_factor) 
            self.avg_investment_amt = np.random.gamma(1,sd_factor*eq_avg_investment/return_factor) 
            
            self.collection_rate = -action_rate
            self.avg_collection_amt = np.random.beta(return_factor*sd_factor*eq_avg_collection_pct*1000,1000*(1 - eq_avg_collection_pct))*avg_pot 
        
        
        self.in_transactions = math.ceil(self.investment_rate*total_users) + user_count_change.inflow
        
        
        self.out_transactions = math.ceil(self.collection_rate*(total_users)) +\
                            math.ceil( user_count_change.outflow )
        
        
        self.in_value = self.in_transactions*self.avg_investment_amt 
        self.out_value = math.ceil(total_users*self.collection_rate)*self.avg_collection_amt + math.ceil( user_count_change.outflow )*avg_pot
        self.outstanding = self.in_value - self.out_value
                
        self.inflow = user_count_change.inflow
        self.outflow = user_count_change.outflow
        
        self.total_users = total_users + self.inflow - self.outflow
        
        self.pot = total_pot + self.outstanding
        
        self.revenue = transaction_fee*(self.in_value + self.out_value)


class first_day():
    
    def __init__(self,
                 total_pot,
                 total_users,
                 daily_return,
                 daily_sd,
                 util_scale,
                 income_effect_bias,
                 wealth_effect_bias,
                 transaction_fee,
                 eq_investment_rate,
                 eq_avg_investment,
                 eq_avg_collection_pct):
        
        sd_factor = normed_tan(wealth_effect_bias*daily_sd)
        return_factor = normed_tan(income_effect_bias*np.abs(daily_return))
        
        investment_rate = util(util_scale, return_factor*eq_investment_rate/sd_factor)

        #wealth effect
        self.investment_rate = np.random.beta(return_factor*sd_factor*investment_rate*1000,1000*(1 - investment_rate)) 
        # greater volatility introduces greater variance in investmnet amount
        self.avg_investment_amt = np.random.gamma(return_factor,sd_factor*eq_avg_investment) 
        
        self.avg_collection_pct = eq_avg_collection_pct
        
        self.in_transactions = math.ceil(self.investment_rate*total_users)


        self.in_value = self.in_transactions*self.avg_investment_amt

        self.pot = total_pot + self.in_value
        
        self.revenue = transaction_fee*(self.in_value)
    


class Pot():
    
    def __init__(self,
                 initial_pot,
                 initial_users,
                 mu,
                 sigma,
                 cash_return,
                 transaction_fee = 0.5,
                 premium_fee = 5,
                 premium_share = 0.1,
                 n_trials = 1,
                 sd_window = 10,
                 observe_window = 3,
                 user_lookback = 5,
                 initial_cash = 10000,
                 initial_holdings = 10000,
                 util_scale = 30,
                 wealth_effect_bias = 0.3,
                 income_effect_bias = 0.7,
                 eq_retaining_rate = 0.25,
                 eq_counter_cyclical_rate = 0.5,
                 base_user_change_rate = 0.01,
                 eq_investment_rate = 0.075,
                 eq_collection_rate = 0.03,
                 eq_avg_investment = 20,
                 eq_avg_collection_pct = 0.5):
        
        assert initial_holdings >= initial_pot, 'ETF vallue held cannot be less than the total Pot value.'
        
        self.initial_pot = initial_pot
        self.initial_users = initial_users
        
        self.mu = mu
        self.sigma = sigma
        self.cash_return = cash_return
        
        self.transaction_fee = transaction_fee
        self.premium_fee = premium_fee
        
        self.premium_share = premium_share
        
        self.n_trials = n_trials
        
        self.dates  = pd.date_range(start = '2/1/2021', end = '1/31/2024')
        self.n_days = len(self.dates.tolist())
        
        self.sd_window = sd_window
        self.observe_window = observe_window
        self.user_lookback = user_lookback
        
        self.initial_cash = initial_cash
        self.initial_holdings = initial_holdings
        
        self.util_scale = util_scale
        
        self.wealth_effect_bias = wealth_effect_bias
        self.income_effect_bias = income_effect_bias
        
        self.eq_retaining_rate = eq_retaining_rate
        self.eq_counter_cyclical_rate = eq_counter_cyclical_rate
        self.base_user_change_rate = base_user_change_rate
        self.eq_investment_rate = eq_investment_rate
        self.eq_collection_rate = eq_collection_rate
        self.eq_avg_investment = eq_avg_investment
        self.eq_avg_collection_pct = eq_avg_collection_pct
    
    def get_sd(self, returns):
        
        sd = np.zeros((len(returns),1)).flatten()
        
        sd[0] = self.sigma/np.sqrt(252)
        sd[1] = self.sigma/np.sqrt(252)
        
        lookback = self.sd_window
        
        for i,ret in enumerate(returns):
            
            if 1 < i < lookback:
                
                sd[i ] = np.std(returns[:i])
                
            elif i >= lookback:
                
                sd[i ] = np.std(returns[i-lookback:i])
        
        return sd
    
    
    def create_table(self):
        
        etf = self.generate_etf()
        
        returns = (etf[1:] - etf[:-1])/etf[:-1]
        
        
        table = pd.DataFrame(np.zeros((self.n_days, 24)), columns =
                             ['etf','returns', 'sd','cash_payment_share',
                              'pot','holdings','cash','users','premium_users',
                              'inflow','outflow','holdings_transactions',
                              'in_amount','out_amount','outstanding',
                              'in_transactions','out_transactions',
                              'investment_rate','collection_rate',
                              'revenue','premium_revenue','transaction_ratio',
                              'debt_days', 'payment_proc_cost'])
    
    
        table.etf = etf
        table.returns = np.hstack((self.mu/252,returns))
        table.sd = self.get_sd(table.returns)

        return table
    
    def buy(self, amount):
        
        self.cash -= amount
        self.holdings += amount
        
        self.holdings_transactions += 1
        
    def sell(self, amount):
        
        self.cash += amount
        self.holdings -= amount
        
        self.holdings_transactions +=1
    
    
    def generate_etf(self):
        
        T = self.n_days
        dt = 1/T
        
        simulated_etf = np.exp(np.cumsum( (self.mu - 0.5*self.sigma**2 )*dt + self.sigma*np.sqrt(dt)*np.random.normal(0,1,T) ))

        return simulated_etf
    
    
    def settle_outstanding(self,day , day_results):
        
        self.cash += day_results.revenue 
        
        if self.outstanding < 0:
                                
                if (self.cash + self.outstanding)/self.pot >= 0.2:
                    
                    self.cash += self.outstanding
                    
                    self.cash_payment += 1
                
                else:
                    
                    amount = -self.outstanding
                    
                    self.sell(amount)
                    
                    self.cash -= amount
                    
                    self.holdings_payment += 1
                    
                    
        elif day_results.pot + self.outstanding >= self.holdings:
            
            self.cash += self.outstanding
            
            amount = np.exp(10*self.table.sd[day])*self.outstanding
            
            if amount > 0.15*self.cash:
                
                amount = self.outstanding
            
            self.buy(amount)
                 
    
        else:
        
            self.cash += self.outstanding
                
    
    def rebalance_portfolio(self,day, total_pot ):
        
        threshold = 0.9*self.holdings
        
        exceedence = self.holdings - total_pot
        
        window = self.observe_window
        
        if total_pot < threshold and total_pot < np.min(self.table.loc[day - window:day,'pot']):
            
            if self.cash >= 0.3*self.holdings:
                
                if increasing(self.table.loc[day - window:day,'returns']):
                    
                    amount = (1 + 10*np.sum(self.table.loc[day - window:day,'returns']))*exceedence*0.5
                
                else:
                
                    amount = None
                
            elif  0.3*self.holdings > self.cash >= 0.1*self.holdings:
                
                
                if increasing(-self.table.loc[day-2:day,'returns']) :
                
                    amount = (1 + 10*self.table.loc[day,'returns'])*exceedence*0.5
                    
                else:
                    
                    amount = None
                    
            else:
                
                amount = threshold - total_pot
            
            if amount and (self.holdings - amount) > self.initial_holdings:
            
                self.sell(amount)
                
        
        elif total_pot >= threshold and increasing(self.table.loc[day-window:day,'pot']):
            
            amount = np.sum(np.abs(self.table.loc[day-21:day,'outstanding']))
            
            if amount >= 0.3*self.cash:
            
                amount = np.exp(100*self.table.loc[day,'sd'])*(1 + 100*self.table.loc[day,'returns'])*exceedence
            
            self.buy(amount)
        
        threshold = 0.05*self.holdings
        
        amount = 0.5*exceedence
        
        if self.cash <= threshold and (self.holdings - amount) >= self.initial_holdings:
            
            self.sell(amount)
    
    def trial(self):
        
        self.table = self.create_table()
        
        returns = self.table.returns
        sd = self.table.sd
        
        self.holdings = self.initial_holdings
        self.cash = self.initial_cash
        
        self.pot = self.initial_pot
        self.total_users = self.initial_users
        
        self.cash_payment = 0
        self.holdings_payment = 0
        self.total_transactions = 0
        
        self.holdings_transactions = 0
        
        self.debt_days = 0
        
        daily_return = returns[0]
        daily_sd = sd[0]
        
        day_results = first_day(total_pot = self.pot,
                        total_users = self.total_users,
                        daily_return = daily_return,
                        daily_sd = daily_sd,
                        util_scale = self.util_scale,
                        income_effect_bias = self.income_effect_bias,
                        wealth_effect_bias = self.wealth_effect_bias,
                        transaction_fee = self.transaction_fee,
                        eq_investment_rate = self.eq_investment_rate,
                        eq_avg_investment = self.eq_avg_investment,
                        eq_avg_collection_pct = self.eq_avg_collection_pct)
        
        self.pot = day_results.pot
        self.total_transactions += day_results.in_transactions
        
        self.outstanding = day_results.in_value
        
        self.settle_outstanding(day = 0, day_results = day_results)
            
        if self.cash < 0:
            
            self.debt_days +=1
            
        premium_fees = self.premium_share*self.total_users*self.premium_fee/30
        
        if self.holdings_transactions >0:
            
            trans_ratio = self.total_transactions/self.holdings_transactions
        
        else:
            
            trans_ratio = 0
        
        self.table.loc[0,'cash_payment_share':] = [0, self.pot , self.holdings, self.cash, self.total_users,np.round(self.premium_share*self.total_users), 
                                    0,0, self.holdings_transactions,
                                    day_results.in_value,0, self.outstanding,
                                    day_results.in_transactions, 0,
                                    day_results.investment_rate,self.eq_collection_rate,
                                    day_results.revenue , premium_fees, trans_ratio,
                                    self.debt_days, 0.005*(day_results.in_value)]
        
        self.cash += day_results.revenue
        
        self.loops = np.zeros((4,len(self.table)))
        
        for i in range(1,len(returns)):
            
            
            daily_return = returns[i-1]
            daily_sd = sd[i-1]
            
            shifted_return = (1 + daily_return)
                
            if i > self.user_lookback:
                
                return_factor = normed_tan(self.wealth_effect_bias*np.sum(returns[i-self.user_lookback:i]))
            
            else:
               
                return_factor = normed_tan(self.wealth_effect_bias*np.sum(returns[:self.user_lookback]))
            
                
            sd_factor = normed_tan(self.wealth_effect_bias*daily_sd)
        
            
            self.cash *= (1 + self.cash_return)
            self.pot *= shifted_return
            self.holdings *= shifted_return
        
            
            day_results = day(total_pot = self.pot,
                              total_users = self.total_users,
                              daily_return = daily_return,
                              daily_sd = daily_sd,
                              transaction_fee = self.transaction_fee,
                              util_scale = self.util_scale,
                              wealth_effect_bias = self.wealth_effect_bias,
                              income_effect_bias = self.income_effect_bias,
                              base_user_change_rate = self.base_user_change_rate,
                              eq_retaining_rate = self.eq_retaining_rate,
                              eq_counter_cyclical_rate = self.eq_counter_cyclical_rate,
                              eq_investment_rate = self.eq_investment_rate,
                              eq_collection_rate = self.eq_collection_rate,
                              eq_avg_investment = self.eq_avg_investment,
                              eq_avg_collection_pct = self.eq_avg_collection_pct)
            
            self.total_users = day_results.total_users
            
            self.pot = day_results.pot
            
            self.total_transactions += day_results.in_transactions + day_results.out_transactions
            
            if day_results.total_users == 0:
                
                break
            
            self.outstanding = day_results.outstanding
            self.settle_outstanding(day = i, day_results = day_results)
            
            if self.cash_payment + self.holdings_payment > 0:
            
                cash_payment_share  = self.cash_payment/(self.cash_payment + self.holdings_payment)
                
            else: 
                
                cash_payment_share = None
                
            if cash_payment_share:
                
                current_cash_payment_share = cash_payment_share
                
            else:
                
                current_cash_payment_share = 0
                
            self.rebalance_portfolio(day = i,
                                     total_pot = day_results.pot)
            
            if self.cash < 0:
            
                self.debt_days +=1
                
            premium_users = np.round(np.random.beta(100*return_factor/self.premium_share,1000*sd_factor/self.premium_share)*self.total_users)
                
            premium_fees = premium_users*self.premium_fee/30
            
            if self.holdings_transactions > 0:
            
                trans_ratio = self.total_transactions/self.holdings_transactions
        
            else:
                
                trans_ratio = 0
                
            self.cash += premium_fees
                
            self.table.loc[i,'cash_payment_share':] = [
                         current_cash_payment_share, self.pot, 
                         self.holdings, self.cash + premium_fees, self.total_users, premium_users,
                         day_results.inflow, day_results.outflow, self.holdings_transactions,
                         day_results.in_value,day_results.out_value, self.outstanding,
                         day_results.in_transactions, day_results.out_transactions,
                         day_results.investment_rate,day_results.collection_rate,
                         day_results.revenue + premium_fees , premium_fees, trans_ratio,
                         self.debt_days, 0.005*(day_results.in_value + day_results.out_value)]
    

    def simulate(self):
        
        self.simulated_users = np.zeros((self.n_days,self.n_trials))
        self.simulated_p_users = np.zeros((self.n_days,self.n_trials))
        
        self.simulated_holdings_transactions = np.zeros((self.n_days,self.n_trials))
        
        self.simulated_cash_payment_share = np.zeros((self.n_days,self.n_trials))
        
        self.simulated_pots = np.zeros((self.n_days,self.n_trials))
        self.simulated_etf = np.zeros((self.n_days,self.n_trials))
        self.simulated_cash = np.zeros((self.n_days,self.n_trials))
        self.simulated_holdings = np.zeros((self.n_days,self.n_trials))
        
        self.simulated_cushion = np.zeros((self.n_days,self.n_trials))
        
        
        self.simulated_sd = np.zeros((self.n_days,self.n_trials))
        
        self.simulated_in_transactions = np.zeros((self.n_days,self.n_trials))
        self.simulated_out_transactions = np.zeros((self.n_days,self.n_trials))
        
        self.simulated_in_amount = np.zeros((self.n_days,self.n_trials))
        self.simulated_out_amount = np.zeros((self.n_days,self.n_trials))
        
        self.simulated_outstanding = np.zeros((self.n_days,self.n_trials))
        
        self.simulated_revenue = np.zeros((self.n_days,self.n_trials))
        self.simulated_p_revenue = np.zeros((self.n_days,self.n_trials))
        self.simulated_cum_revenue = np.zeros((self.n_days,self.n_trials))
        
        self.simulated_transaction_ratio = np.zeros((self.n_days,self.n_trials))
        
        self.simulated_debt_days = np.zeros((self.n_days,self.n_trials))
        self.simulated_payment_proc_cost = np.zeros((self.n_days,self.n_trials))

        
        
        for i in range(self.n_trials):
            
            self.trial()
            
            self.simulated_users[:,i] = self.table.loc[:,'users']
            self.simulated_p_users[:,i] = self.table.loc[:,'premium_users']
            
            self.simulated_holdings_transactions[:,i] = self.table.loc[:,'holdings_transactions']
            
            self.simulated_cash_payment_share[:,i] = self.table.loc[:,'cash_payment_share']
            
            self.simulated_pots[:,i] = self.table.loc[:,'pot']
            self.simulated_etf[:,i] = self.table.loc[:,'etf']
            self.simulated_cash[:,i] = self.table.loc[:,'cash']
            self.simulated_holdings[:,i] = self.table.loc[:,'holdings']
            
            self.simulated_cushion[:,i] = self.simulated_cash[:,i]/self.simulated_holdings[:,i]
            
            self.simulated_sd[:,i] = self.table.loc[:,'sd']
            
            self.simulated_in_transactions[:,i] = self.table.loc[:,'in_transactions']
            self.simulated_out_transactions[:,i] = self.table.loc[:,'out_transactions']
            
            self.simulated_in_amount[:,i] = self.table.loc[:,'in_amount']
            self.simulated_out_amount[:,i] = self.table.loc[:,'out_amount']
            
            self.simulated_outstanding[:,i] = self.table.loc[:,'outstanding']
            
            self.simulated_revenue[:,i] = self.table.loc[:,'revenue']
            self.simulated_p_revenue[:,i] = self.table.loc[:,'premium_revenue']
            self.simulated_cum_revenue = np.cumsum(self.simulated_revenue, axis = 0)
            
            self.simulated_transaction_ratio[:,i] = self.table.loc[:,'transaction_ratio']
            
            self.simulated_debt_days[:,i] = self.table.loc[:,'debt_days']
            self.simulated_payment_proc_cost[:,i] = self.table.loc[:,'payment_proc_cost']
            
            if i%50 == 0:
                
                print(f'Finished {i} trials.')
        
        self.simulated_etf *= 1000
                


if __name__ == '__main__':
    
    ftse_100_return = 0.0994
    ftse_100_sd = 0.1041
    treasury_daily_yield = (1 + 0.0239)**(1/252) - 1
    
    a = Pot(initial_pot=150,
            initial_users = 10,
            mu = ftse_100_return,
            sigma = ftse_100_sd,
            cash_return = treasury_daily_yield,
            transaction_fee = 0.03,
            premium_fee = 10,
            n_trials = 100,
            sd_window = 10,
            observe_window = 5,
            user_lookback = 5,
            initial_cash = 10000,
            initial_holdings = 10000,
            util_scale = 10,
            wealth_effect_bias = 5,
            income_effect_bias = 5,
            base_user_change_rate = 0.06,
            eq_retaining_rate = 0.25,
            eq_counter_cyclical_rate = 0.5,
            eq_investment_rate = 1/14,
            eq_collection_rate = 1/30,
            eq_avg_investment = 25,
            eq_avg_collection_pct = 0.2)
    
    
    a.simulate()
    
    plot_sim(a)
    save_csv2(a)    



