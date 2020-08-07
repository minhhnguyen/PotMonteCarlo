import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_sim(simulation):
    
    plt.figure()
    plt.plot(simulation.dates,simulation.simulated_users)
    plt.xlabel('Date')
    plt.ylabel('Users')
    plt.xticks(rotation = 30)
    
    plt.figure()
    plt.plot(simulation.dates,simulation.simulated_cash_payment_share)
    plt.xlabel('Date')
    plt.ylabel('Cash Payment Share')
    plt.xticks(rotation = 30)
    
    plt.figure()
    plt.plot(simulation.dates, simulation.simulated_etf)
    plt.xlabel('Date')
    plt.ylabel('ETF Value')
    plt.xticks(rotation = 30)
    
    plt.figure()
    plt.plot(simulation.dates, simulation.simulated_pots)
    plt.xlabel('Date')
    plt.ylabel('Total Pot')
    plt.xticks(rotation = 30)
    
    plt.figure()
    plt.plot(simulation.dates, simulation.simulated_cash)
    plt.xlabel('Date')
    plt.ylabel('Total Cash')
    plt.xticks(rotation = 30)
    
    plt.figure()
    plt.plot(simulation.dates, simulation.simulated_holdings)
    plt.xlabel('Date')
    plt.ylabel('Total Holdings')
    plt.xticks(rotation = 30)
    
    plt.figure()
    plt.plot(simulation.dates, simulation.simulated_debt_days)
    plt.xlabel('Date')
    plt.ylabel('Debt Days')
    plt.xticks(rotation = 30)
    
    plt.figure()
    plt.plot(simulation.dates, simulation.simulated_revenue)
    plt.xlabel('Date')
    plt.ylabel('Daily Revenue')
    plt.xticks(rotation = 30)
    
    
    cash_payment_share_75 = np.quantile(simulation.simulated_cash_payment_share,0.75, axis = 1)
    cash_payment_share_50 = np.quantile(simulation.simulated_cash_payment_share,0.5, axis = 1)
    cash_payment_share_25 = np.quantile(simulation.simulated_cash_payment_share,0.25, axis = 1)
    cash_payment_share_mean = np.mean(simulation.simulated_cash_payment_share, axis = 1)
    
    plt.figure()
    plt.plot(simulation.dates, cash_payment_share_75, color = 'red')
    plt.plot(simulation.dates, cash_payment_share_50, color = 'blue')
    plt.plot(simulation.dates, cash_payment_share_25, color = 'green')
    plt.plot(simulation.dates, cash_payment_share_mean, color = 'yellow')
    plt.xlabel('Date')
    plt.ylabel('Cash Payment Share')
    plt.legend(['$75^{th}$ percentile','$50^{th}$ percentile','$25^{th}$ percentile','mean'])
    plt.xticks(rotation = 30)
    
    cushion_75 = np.quantile(simulation.simulated_cushion,0.75, axis = 1)
    cushion_50 = np.quantile(simulation.simulated_cushion,0.5, axis = 1)
    cushion_25 = np.quantile(simulation.simulated_cushion,0.25, axis = 1)
    cushion_mean = np.mean(simulation.simulated_cushion, axis = 1)
    
    plt.figure()
    plt.plot(simulation.dates, cushion_75, color = 'red')
    plt.plot(simulation.dates, cushion_50, color = 'blue')
    plt.plot(simulation.dates, cushion_25, color = 'green')
    plt.plot(simulation.dates, cushion_mean, color = 'yellow')
    plt.xlabel('Date')
    plt.ylabel('Cushion')
    plt.legend(['$75^{th}$ percentile','$50^{th}$ percentile','$25^{th}$ percentile','mean'])
    plt.xticks(rotation = 30)
    
    
    cash_75 = np.quantile(simulation.simulated_cash,0.75, axis = 1)
    cash_50 = np.quantile(simulation.simulated_cash,0.50, axis = 1)
    cash_25 = np.quantile(simulation.simulated_cash,0.25, axis = 1)
    cash_mean = np.mean(simulation.simulated_cash, axis = 1)
    holdings_75 = np.quantile(simulation.simulated_holdings,0.75, axis = 1)
    holdings_50 = np.quantile(simulation.simulated_holdings,0.50, axis = 1)
    holdings_25 = np.quantile(simulation.simulated_holdings,0.25, axis = 1)
    holdings_mean = np.mean(simulation.simulated_holdings, axis = 1)
    pot_75 = np.quantile(simulation.simulated_pots,0.75, axis = 1)
    pot_50 = np.quantile(simulation.simulated_pots,0.5, axis = 1)
    pot_25 = np.quantile(simulation.simulated_pots,0.25, axis = 1)
    pot_mean = np.mean(simulation.simulated_pots, axis = 1)

    
    plt.figure()
    plt.plot(simulation.dates, cash_75, color = 'red')
    plt.plot(simulation.dates, cash_50, color = 'blue')
    plt.plot(simulation.dates, cash_25, color = 'green')
#    plt.plot(simulation.dates, cash_mean, color = 'yellow')
    plt.plot(simulation.dates, holdings_75, linestyle = ':', color = 'red')
    plt.plot(simulation.dates, holdings_50, linestyle = ':', color = 'blue')
    plt.plot(simulation.dates, holdings_25, linestyle = ':', color = 'green')
#    plt.plot(simulation.dates, holdings_mean, linestyle = ':', color = 'yellow')
#    plt.plot(simulation.dates, pot_75, linestyle = '--', color = 'red')
#    plt.plot(simulation.dates, pot_50, linestyle = '--', color = 'blue')
#    plt.plot(simulation.dates, pot_25, linestyle = '--', color = 'green')
#    plt.plot(simulation.dates, pot_mean, linestyle = '--', color = 'yellow')
    plt.xlabel('Date')
    plt.ylabel('Amount')
#    plt.legend(['$75^{th}$ cash','$50^{th}$ cash','$25^{th}$ cash','cash mean',
#                '$75^{th}$ holdings','$50^{th}$ holdings','$25^{th}$ holdings','holdings mean',
#                '$75^{th}$ pot','$50^{th}$ pot','$25^{th}$ pot','pot mean'], bbox_to_anchor=(1.05, 0.6) )
    plt.xticks(rotation = 30)
    

    
    outstanding_75 = np.quantile(simulation.simulated_outstanding,0.75, axis = 1)
    outstanding_50 = np.quantile(simulation.simulated_outstanding,0.5, axis = 1)
    outstanding_25 = np.quantile(simulation.simulated_outstanding,0.25, axis = 1)
    outstanding_mean = np.mean(simulation.simulated_outstanding, axis = 1)
        
    
    plt.figure()
    plt.plot(simulation.dates, outstanding_75)
    plt.plot(simulation.dates, outstanding_50)
    plt.plot(simulation.dates, outstanding_25)
    plt.plot(simulation.dates, outstanding_mean)
    plt.xlabel('Date')
    plt.ylabel('Outstanding')
    plt.legend(['$75^{th}$ percentile','$50^{th}$ percentile','$25^{th}$ percentile','mean'])
    plt.xticks(rotation = 30)
    
    
    
    etf_75 = np.quantile(simulation.simulated_etf,0.75, axis = 1)
    etf_50 = np.quantile(simulation.simulated_etf,0.5, axis = 1)
    etf_25 = np.quantile(simulation.simulated_etf,0.25, axis = 1)
    etf_mean = np.mean(simulation.simulated_etf, axis = 1)
        
    
    plt.figure()
    plt.plot(simulation.dates, etf_75)
    plt.plot(simulation.dates, etf_50)
    plt.plot(simulation.dates, etf_25)
    plt.plot(simulation.dates, etf_mean)
    plt.xlabel('Date')
    plt.ylabel('ETF Value')
    plt.legend(['$75^{th}$ percentile','$50^{th}$ percentile','$25^{th}$ percentile','mean'])
    plt.xticks(rotation = 30)
    
    estimated_users_75 = np.quantile(simulation.simulated_users,0.75, axis = 1)
    estimated_users_50 = np.quantile(simulation.simulated_users,0.5, axis = 1)
    estimated_users_25 = np.quantile(simulation.simulated_users,0.25, axis = 1)
    estimated_users_mean = np.mean(simulation.simulated_users, axis = 1)
    
    
    plt.figure()
    plt.plot(simulation.dates, estimated_users_75)
    plt.plot(simulation.dates, estimated_users_50)
    plt.plot(simulation.dates, estimated_users_25)
    plt.plot(simulation.dates, estimated_users_mean)
    plt.xlabel('Date')
    plt.ylabel('Users')
    plt.legend(['$75^{th}$ percentile','$50^{th}$ percentile','$25^{th}$ percentile','mean'])
    plt.xticks(rotation = 30)
    
    
    revenue_75 = np.quantile(simulation.simulated_revenue,0.75, axis = 1)
    revenue_50 = np.quantile(simulation.simulated_revenue,0.5, axis = 1)
    revenue_25 = np.quantile(simulation.simulated_revenue,0.25, axis = 1)
    revenue_mean = np.mean(simulation.simulated_revenue, axis = 1)
        
    
    plt.figure()
    plt.plot(simulation.dates, revenue_75)
    plt.plot(simulation.dates, revenue_50)
    plt.plot(simulation.dates, revenue_25)
    plt.plot(simulation.dates, revenue_mean)
    plt.xlabel('Date')
    plt.ylabel('Daily Revenue')
    plt.legend(['$75^{th}$ percentile','$50^{th}$ percentile','$25^{th}$ percentile','mean'])
    plt.xticks(rotation = 30)
    
    cum_revenue_75 = np.quantile(simulation.simulated_cum_revenue,0.75, axis = 1)
    cum_revenue_50 = np.quantile(simulation.simulated_cum_revenue,0.5, axis = 1)
    cum_revenue_25 = np.quantile(simulation.simulated_cum_revenue,0.25, axis = 1)
    cum_revenue_mean = np.mean(simulation.simulated_cum_revenue, axis = 1)
        
    
    plt.figure()
    plt.plot(simulation.dates, cum_revenue_75)
    plt.plot(simulation.dates, cum_revenue_50)
    plt.plot(simulation.dates, cum_revenue_25)
    plt.plot(simulation.dates, cum_revenue_mean)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Revenue')
    plt.legend(['$75^{th}$ percentile','$50^{th}$ percentile','$25^{th}$ percentile','mean'])
    plt.xticks(rotation = 30)
    
    
    trans_ratio_75 = np.quantile(simulation.simulated_transaction_ratio,0.75, axis = 1)
    trans_ratio_50 = np.quantile(simulation.simulated_transaction_ratio,0.5, axis = 1)
    trans_ratio_25 = np.quantile(simulation.simulated_transaction_ratio,0.25, axis = 1)
    trans_ratio_mean = np.mean(simulation.simulated_transaction_ratio, axis = 1)
        
    
    plt.figure()
    plt.plot(simulation.dates, trans_ratio_75)
    plt.plot(simulation.dates, trans_ratio_50)
    plt.plot(simulation.dates, trans_ratio_25)
    plt.plot(simulation.dates, trans_ratio_mean)
    plt.xlabel('Date')
    plt.ylabel('Transaction Ratio')
    plt.legend(['$75^{th}$ percentile','$50^{th}$ percentile','$25^{th}$ percentile','mean'])
    plt.xticks(rotation = 30)
    
    payment_proc_cost_75 = np.quantile(simulation.simulated_payment_proc_cost,0.75, axis = 1)
    payment_proc_cost_50 = np.quantile(simulation.simulated_payment_proc_cost,0.5, axis = 1)
    payment_proc_cost_25 = np.quantile(simulation.simulated_payment_proc_cost,0.25, axis = 1)
    payment_proc_cost_mean = np.mean(simulation.simulated_payment_proc_cost, axis = 1)
        
    
    plt.figure()
    plt.plot(simulation.dates, payment_proc_cost_75)
    plt.plot(simulation.dates, payment_proc_cost_50)
    plt.plot(simulation.dates, payment_proc_cost_25)
    plt.plot(simulation.dates, payment_proc_cost_mean)
    plt.xlabel('Date')
    plt.ylabel('PP Revenue Ratio')
    plt.legend(['$75^{th}$ percentile','$50^{th}$ percentile','$25^{th}$ percentile','mean'])
    plt.xticks(rotation = 30)
    
    
    sim = np.random.randint(0,len(simulation.simulated_holdings[0]))
    
    beginning = 600
    length = beginning + 100
    
    fig, ax1 = plt.subplots()
    
    
    
    ax1.plot(simulation.dates[beginning:length], simulation.simulated_holdings[beginning:length,sim])
    ax1.plot(simulation.dates[beginning:length], simulation.simulated_pots[beginning:length,sim])
    ax1.plot(simulation.dates[beginning:length], simulation.simulated_cash[beginning:length,sim])
    ax1.set_ylabel('Value')
    
    
    ax2 = ax1.twinx()
    ax2.plot(simulation.dates[beginning:length], simulation.simulated_etf[beginning:length,sim], color = 'red')
    ax2.yaxis.label.set_color('red')
    ax2.set_ylabel('ETF Value')
    
    fig.legend(['Holdings','Pot','Cash','ETF'],loc='center left', bbox_to_anchor=(1.05, 0.5) )
    plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
    
    
    fig.show()

        
def plot_trial(a, start, end):
    
        fig, ax1 = plt.subplots()
    
        ax1.plot(a.dates[start:end], a.table.holdings[start:end])
        ax1.plot(a.dates[start:end], a.table.pot[start:end])
        ax1.plot(a.dates[start:end], a.table.cash[start:end])
        ax1.set_ylabel('Value')
        
        ax2 = ax1.twinx()
        ax2.plot(a.dates[start:end], a.table.etf[start:end], color = 'red')
        ax2.yaxis.label.set_color('red')
        ax2.set_ylabel('ETF Value')
        
        fig.legend(['Holdings','Pot','Cash','ETF'])
        
        plt.figure()
    
        plt.plot(a.dates[start:end], a.table.in_transactions[start:end])
        plt.plot(a.dates[start:end], a.table.out_transactions[start:end])
        plt.ylabel('No. of Transactions')
        plt.legend(['In Trans','Out Trans'])
        
        plt.figure()
        plt.plot(a.dates[start:end], a.table.in_amount[start:end],color = 'red')
        plt.plot(a.dates[start:end], a.table.out_amount[start:end],color = 'blue')
        plt.ylabel('Value')
        
        plt.legend(['In Amt','Out Amt'])
    
        
        fig, ax1 = plt.subplots()
    
        plt.plot(a.dates, np.cumsum(a.table.revenue))
        plt.plot(a.dates, a.table.holdings)
        plt.plot(a.dates, a.table.pot)
        plt.plot(a.dates, a.table.cash)
        ax1.set_ylabel('Value')
        
        ax2 = ax1.twinx()
        ax2.plot(a.dates, a.table.etf, color = 'black')
        ax2.yaxis.label.set_color('black')
        ax2.set_ylabel('ETF Value')
    
        fig.legend(['Revenue','Holdings','Pot','Cash','ETF'])
        
        
        plt.figure()
        plt.plot(a.dates, np.cumsum(a.table.revenue)/a.table.holdings)
        plt.plot(a.dates, a.table.cash/a.table.holdings)
        plt.axhline(y = 0.25, color = 'red')
        plt.legend(['Revenue','Cash'])
        
        plt.figure()
        plt.plot(a.dates, a.table.users)
        plt.ylabel('Users')
        
        plt.figure()
        plt.plot(a.dates, a.table.cash_payment_share)
        plt.ylabel('Cash Payment Share')
        
        
        
def save_csv(a):
    
    simulation_results = pd.DataFrame(np.vstack([
                                             np.quantile(a.simulated_etf, 0, axis = 1),
                                             np.quantile(a.simulated_pots, 0, axis = 1),
                                             np.quantile(a.simulated_cash, 0, axis = 1),
                                             np.quantile(a.simulated_holdings, 0, axis = 1),
                                             np.quantile(a.simulated_users, 0, axis = 1),
                                             np.quantile(a.simulated_p_users, 0, axis = 1),
                                             np.quantile(a.simulated_in_transactions, 0, axis = 1),
                                             np.quantile(a.simulated_out_transactions, 0, axis = 1),
                                             np.quantile(a.simulated_in_amount, 0, axis = 1),
                                             np.quantile(a.simulated_out_amount, 0, axis = 1),
                                             np.quantile(a.simulated_revenue, 0, axis = 1),
                                             np.quantile(a.simulated_p_revenue, 0, axis = 1),
                                             np.quantile(a.simulated_holdings_transactions, 0, axis = 1),
                                             np.quantile(a.simulated_payment_proc_cost, 0, axis = 1),
                                             np.quantile(a.simulated_etf, 0.25, axis = 1),
                                             np.quantile(a.simulated_pots, 0.25, axis = 1),
                                             np.quantile(a.simulated_cash, 0.25, axis = 1),
                                             np.quantile(a.simulated_holdings, 0.25, axis = 1),
                                             np.quantile(a.simulated_users, 0.25, axis = 1),
                                             np.quantile(a.simulated_p_users, 0.25, axis = 1),
                                             np.quantile(a.simulated_in_transactions, 0.25, axis = 1),
                                             np.quantile(a.simulated_out_transactions, 0.25, axis = 1),
                                             np.quantile(a.simulated_in_amount, 0.25, axis = 1),
                                             np.quantile(a.simulated_out_amount, 0.25, axis = 1),
                                             np.quantile(a.simulated_revenue, 0.25, axis = 1),
                                             np.quantile(a.simulated_p_revenue, 0.25, axis = 1),
                                             np.quantile(a.simulated_holdings_transactions, 0.25, axis = 1),
                                             np.quantile(a.simulated_payment_proc_cost, 0.25, axis = 1),
                                             np.quantile(a.simulated_etf, 0.5, axis = 1),
                                             np.quantile(a.simulated_pots, 0.5, axis = 1),
                                             np.quantile(a.simulated_cash, 0.5, axis = 1),
                                             np.quantile(a.simulated_holdings, 0.5, axis = 1),
                                             np.quantile(a.simulated_users, 0.5, axis = 1),
                                             np.quantile(a.simulated_p_users, 0.5, axis = 1),
                                             np.quantile(a.simulated_in_transactions, 0.5, axis = 1),
                                             np.quantile(a.simulated_out_transactions, 0.5, axis = 1),
                                             np.quantile(a.simulated_in_amount, 0.5, axis = 1),
                                             np.quantile(a.simulated_out_amount, 0.5, axis = 1),
                                             np.quantile(a.simulated_revenue, 0.5, axis = 1),
                                             np.quantile(a.simulated_p_revenue, 0.5, axis = 1),
                                             np.quantile(a.simulated_holdings_transactions, 0.5, axis = 1),
                                             np.quantile(a.simulated_payment_proc_cost, 0.5, axis = 1),
                                             np.quantile(a.simulated_etf, 0.75, axis = 1),
                                             np.quantile(a.simulated_pots, 0.75, axis = 1),
                                             np.quantile(a.simulated_cash, 0.75, axis = 1),
                                             np.quantile(a.simulated_holdings, 0.75, axis = 1),
                                             np.quantile(a.simulated_users, 0.75, axis = 1),
                                             np.quantile(a.simulated_p_users, 0.75, axis = 1),
                                             np.quantile(a.simulated_in_transactions, 0.75, axis = 1),
                                             np.quantile(a.simulated_out_transactions, 0.75, axis = 1),
                                             np.quantile(a.simulated_in_amount, 0.75, axis = 1),
                                             np.quantile(a.simulated_out_amount, 0.75, axis = 1),
                                             np.quantile(a.simulated_revenue, 0.75, axis = 1),
                                             np.quantile(a.simulated_p_revenue, 0.75, axis = 1),
                                             np.quantile(a.simulated_holdings_transactions, 0.75, axis = 1),
                                             np.quantile(a.simulated_payment_proc_cost, 0.75, axis = 1),
                                             np.mean(a.simulated_etf, axis = 1),
                                             np.mean(a.simulated_pots, axis = 1),
                                             np.mean(a.simulated_cash, axis = 1),
                                             np.mean(a.simulated_holdings, axis = 1),
                                             np.mean(a.simulated_users, axis = 1),
                                             np.mean(a.simulated_p_users, axis = 1),
                                             np.mean(a.simulated_in_transactions, axis = 1),
                                             np.mean(a.simulated_out_transactions, axis = 1),
                                             np.mean(a.simulated_in_amount, axis = 1),
                                             np.mean(a.simulated_out_amount, axis = 1),
                                             np.mean(a.simulated_revenue, axis = 1),
                                             np.mean(a.simulated_p_revenue, axis = 1),
                                             np.mean(a.simulated_holdings_transactions, axis = 1),
                                             np.mean(a.simulated_payment_proc_cost, axis = 1)]
                                              ).T,
                                             columns = ['etf_0','pot_0','cash_0','holdings_0','users_0','p_users_0','in_transactions_0','out_transactions_0','in_amount_0','out_amount_0','revenue_0','p_revenue_0','holdings_transactions_0','payment_proc_cost_0',
                                                        'etf_25','pot_25','cash_25','holdings_25','users_25','p_users_25','in_transactions_25','out_transactions_25','in_amount_25','out_amount_25','revenue_25','p_revenue_25','holdings_transactions_25','payment_proc_cost_25',
                                                        'etf_50','pot_50','cash_50','holdings_50','users_50','p_users_50','in_transactions_50','out_transactions_50','in_amount_50','out_amount_50','revenue_50','p_revenue_50','holdings_transactions_50','payment_proc_cost_50',
                                                        'etf_75','pot_75','cash_75','holdings_75','users_75','p_users_75','in_transactions_75','out_transactions_75','in_amount_75','out_amount_75','revenue_75','p_revenue_75','holdings_transactions_75','payment_proc_cost_75',
                                                        'etf_mean','pot_mean','cash_mean','holdings_mean','users_mean','p_users_mean','in_transactions_mean','out_transactions_75','in_amount_mean','out_amount_mean','revenue_mean','p_revenue_mean','holdings_transactions_mean',
                                                        'payment_proc_cost_mean'], index = a.dates)

    monthly_results = simulation_results.loc[:,['users_0','users_25','users_50','users_75','users_mean',
                                                'p_users_0','p_users_25','p_users_50','p_users_75','p_users_mean',
                                                'etf_0','etf_25','etf_50','etf_75','etf_mean',
                                                'pot_0','pot_25','pot_50','pot_75','pot_mean',
                                                'cash_0','cash_25','cash_50','cash_75','cash_mean',
                                                'holdings_0','holdings_25','holdings_50','holdings_75','holdings_mean',
                                                'holdings_transactions_0','holdings_transactions_25','holdings_transactions_50','holdings_transactions_75','holdings_transactions_mean'
                                                ]].loc[simulation_results.index.is_month_end]
    
    transaction_results = simulation_results.groupby(pd.Grouper(freq='M')).sum().drop(['users_0','users_25','users_50','users_75','users_mean',
                                                'p_users_0','p_users_25','p_users_50','p_users_75','p_users_mean',
                                                'etf_0','etf_25','etf_50','etf_75','etf_mean',
                                                'pot_0','pot_25','pot_50','pot_75','pot_mean',
                                                'cash_0','cash_25','cash_50','cash_75','cash_mean',
                                                'holdings_0','holdings_25','holdings_50','holdings_75','holdings_mean',
                                                'holdings_transactions_0','holdings_transactions_25','holdings_transactions_50','holdings_transactions_75','holdings_transactions_mean'], axis = 1)
                                              
    
    full_results = pd.concat([monthly_results,transaction_results], axis = 1).round()
    
    full_results.to_csv('full_results.csv', encoding='utf-8')
    

def get_pct_indexes(a):
    
    users = a.simulated_users
    indexes = np.zeros((len(users),4))
    
    
    for i in range(len(users)):
        
        users_0 = np.quantile(a.simulated_users[i,:], 0.001, interpolation = 'nearest')
        users_25 = np.quantile(a.simulated_users[i,:], 0.25, interpolation = 'nearest')
        users_50 = np.quantile(a.simulated_users[i,:], 0.5, interpolation = 'nearest')
        users_75 = np.quantile(a.simulated_users[i,:], 0.75, interpolation = 'nearest')
        
        indexes[i,0] = np.where(users[i,:] == users_0)[0][0]
        indexes[i,1] = np.where(users[i,:] == users_25)[0][0]
        indexes[i,2] = np.where(users[i,:] == users_50)[0][0]
        indexes[i,3] = np.where(users[i,:] == users_75)[0][0]
        
    return indexes.astype(int)

def get_means(a, indexes):
    
    sim = np.zeros((len(indexes),14))
    
    sim[:,0] = np.mean(a.simulated_etf, axis = 1)
    sim[:,1] = np.mean(a.simulated_pots, axis = 1)
    sim[:,2] = np.mean(a.simulated_cash, axis = 1)
    sim[:,3] = np.mean(a.simulated_holdings, axis = 1)
    sim[:,4] = np.mean(a.simulated_users, axis = 1)
    sim[:,5] = np.mean(a.simulated_p_users, axis = 1)
    sim[:,6] = np.mean(a.simulated_in_transactions, axis = 1)
    sim[:,7] = np.mean(a.simulated_out_transactions, axis = 1)
    sim[:,8] = np.mean(a.simulated_in_amount, axis = 1)
    sim[:,9] = np.mean(a.simulated_out_amount, axis = 1)
    sim[:,10] = np.mean(a.simulated_revenue, axis = 1)
    sim[:,11] = np.mean(a.simulated_p_revenue, axis = 1)
    sim[:,12] = np.mean(a.simulated_holdings_transactions, axis = 1)
    sim[:,13] = np.mean(a.simulated_payment_proc_cost, axis = 1)
    
    return sim


def get_simulated_ts(a, indexes, pct):
    
    sim = np.zeros((len(indexes),14))
    
    if pct == 0:
        i = 0
    elif pct == 0.25:
        i = 1
    elif pct == 0.5:
        i = 2
    else:
        i = 3
    
    for day in range(len(indexes)):
        
        sim[day,0] = a.simulated_etf[day,indexes[day,i]]
        sim[day,1] = a.simulated_pots[day,indexes[day,i]]
        sim[day,2] = a.simulated_cash[day, indexes[day,i]]
        sim[day,3] = a.simulated_holdings[day, indexes[day,i]]
        sim[day,4] = a.simulated_users[day, indexes[day,i]]
        sim[day,5] = a.simulated_p_users[day, indexes[day,i]]
        sim[day,6] = a.simulated_in_transactions[day, indexes[day,i]]
        sim[day,7] = a.simulated_out_transactions[day, indexes[day,i]]
        sim[day,8] = a.simulated_in_amount[day, indexes[day,i]]
        sim[day,9] = a.simulated_out_amount[day, indexes[day,i]]
        sim[day,10] = a.simulated_revenue[day, indexes[day,i]]
        sim[day,11] = a.simulated_p_revenue[day, indexes[day,i]]
        sim[day,12] = a.simulated_holdings_transactions[day, indexes[day,i]]
        sim[day,13] = a.simulated_payment_proc_cost[day, indexes[day,i]]
        
    return sim
    
def save_csv2(a):
    
    indexes = get_pct_indexes(a)
    
    simulation_results = pd.DataFrame(np.hstack([
            get_simulated_ts(a, indexes, 0),
            get_simulated_ts(a, indexes, 0.25),
            get_simulated_ts(a, indexes, 0.5),
            get_simulated_ts(a, indexes, 0.75),
            get_means(a, indexes)
        ]), 
        columns = ['etf_0','pot_0','cash_0','holdings_0','users_0','p_users_0','in_transactions_0','out_transactions_0','in_amount_0','out_amount_0','revenue_0','p_revenue_0','holdings_transactions_0','payment_proc_cost_0',
                   'etf_25','pot_25','cash_25','holdings_25','users_25','p_users_25','in_transactions_25','out_transactions_25','in_amount_25','out_amount_25','revenue_25','p_revenue_25','holdings_transactions_25','payment_proc_cost_25',
                   'etf_50','pot_50','cash_50','holdings_50','users_50','p_users_50','in_transactions_50','out_transactions_50','in_amount_50','out_amount_50','revenue_50','p_revenue_50','holdings_transactions_50','payment_proc_cost_50',
                   'etf_75','pot_75','cash_75','holdings_75','users_75','p_users_75','in_transactions_75','out_transactions_75','in_amount_75','out_amount_75','revenue_75','p_revenue_75','holdings_transactions_75','payment_proc_cost_75',
                   'etf_mean','pot_mean','cash_mean','holdings_mean','users_mean','p_users_mean','in_transactions_mean','out_transactions_75','in_amount_mean','out_amount_mean','revenue_mean','p_revenue_mean','holdings_transactions_mean','payment_proc_cost_mean'
                   ], 
        index = a.dates)

    monthly_results = simulation_results.loc[:,['users_0','users_25','users_50','users_75','users_mean',
                                                'p_users_0','p_users_25','p_users_50','p_users_75','p_users_mean',
                                                'etf_0','etf_25','etf_50','etf_75','etf_mean',
                                                'pot_0','pot_25','pot_50','pot_75','pot_mean',
                                                'cash_0','cash_25','cash_50','cash_75','cash_mean',
                                                'holdings_0','holdings_25','holdings_50','holdings_75','holdings_mean',
                                                'holdings_transactions_0','holdings_transactions_25','holdings_transactions_50','holdings_transactions_75','holdings_transactions_mean'
                                                ]].loc[simulation_results.index.is_month_end]
    
    transaction_results = simulation_results.groupby(pd.Grouper(freq='M')).sum().drop(['users_0','users_25','users_50','users_75','users_mean',
                                                'p_users_0','p_users_25','p_users_50','p_users_75','p_users_mean',
                                                'etf_0','etf_25','etf_50','etf_75','etf_mean',
                                                'pot_0','pot_25','pot_50','pot_75','pot_mean',
                                                'cash_0','cash_25','cash_50','cash_75','cash_mean',
                                                'holdings_0','holdings_25','holdings_50','holdings_75','holdings_mean',
                                                'holdings_transactions_0','holdings_transactions_25','holdings_transactions_50','holdings_transactions_75','holdings_transactions_mean'], axis = 1)
    full_results = pd.concat([monthly_results,transaction_results], axis = 1).round()
    
    full_results.to_csv('full_results2.csv', encoding='utf-8')
    

