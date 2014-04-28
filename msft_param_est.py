#!/usr/bin/env python

import pandas as pd
import numpy as np

MSG_FILENAME = 'MSFT_2012-06-21_34200000_57600000_message_5.csv'
OB_FILENAME = 'MSFT_2012-06-21_34200000_57600000_orderbook_5.csv'

DEPTH = 5

LIMIT_ORDER_SUBMITTED = 1
PARTIAL_CANCEL = 2
FULL_CANCEL = 3
VISIBLE_ORDER_EXEC = 4
HIDDEN_ORDER_EXEC = 5

ASK_PRICE_COL = 'ask_price_'
ASK_SIZE_COL = 'ask_size_'
BID_PRICE_COL = 'bid_price_'
BID_SIZE_COL = 'bid_size_'

# Both in seconds
TRADING_ST_TIME = 9.5*60*60
TRADING_END_TIME = 16*60*60

# Total minutes of trading
trading_time = (TRADING_END_TIME - TRADING_ST_TIME)/60.0

BUY_DIR = 1
SELL_DIR = -1

msg_df = pd.read_csv(MSG_FILENAME)
ob_df = pd.read_csv(OB_FILENAME)

msg_df = msg_df.ix[ (msg_df.index >= TRADING_ST_TIME) &
                    (msg_df.index <= TRADING_END_TIME) ]
ob_df = ob_df.ix[ (msg_df.index >= TRADING_ST_TIME) &
                  (msg_df.index <= TRADING_END_TIME) ]

"""
msg_df['price'] = msg_df['price']/10000.0
for i in range(DEPTH):
    ob_df[ASK_PRICE_COL + str(i+1)] = ob_df[ASK_PRICE_COL + str(i+1)]/10000.0
    ob_df[BID_PRICE_COL + str(i+1)] = ob_df[BID_PRICE_COL + str(i+1)]/10000.0
"""

limit_order_df = msg_df.ix[msg_df['type'] == LIMIT_ORDER_SUBMITTED]
mkt_order_df = msg_df.ix[(msg_df['type'] == VISIBLE_ORDER_EXEC) |
                         (msg_df['type'] == HIDDEN_ORDER_EXEC)]
cancel_order_df = msg_df.ix[msg_df['type'] == FULL_CANCEL]

avg_limit_order_size = np.mean( limit_order_df['size'] )
avg_mkt_order_size = np.mean( mkt_order_df['size'] )
avg_cancel_order_size = np.mean( cancel_order_df['size'] )

avg_lim_buy_size = np.mean(limit_order_df.ix[limit_order_df['direction'] == BUY_DIR]['size'])
avg_lim_sell_size = np.mean(limit_order_df.ix[limit_order_df['direction'] == SELL_DIR]['size'])

avg_mkt_buy_size = np.mean(mkt_order_df.ix[mkt_order_df['direction'] == BUY_DIR]['size'])
avg_mkt_sell_size = np.mean(mkt_order_df.ix[mkt_order_df['direction'] == SELL_DIR]['size'])

# Mu is easy.
n_mkt_buys = len(mkt_order_df.ix[mkt_order_df['direction'] == BUY_DIR])
n_mkt_sells = len(mkt_order_df.ix[mkt_order_df['direction'] == SELL_DIR])
mu_hat_ask = n_mkt_sells/trading_time * (avg_mkt_sell_size/avg_lim_sell_size)
mu_hat_bid = n_mkt_buys/trading_time * (avg_mkt_buy_size/avg_lim_buy_size)

# Want to calculate the distance between the order and the bid/ask

lim_sell_idx = ((msg_df['direction'] == SELL_DIR).values &
                (msg_df['type'] == LIMIT_ORDER_SUBMITTED).values)
lim_buy_idx = ((msg_df['direction'] == BUY_DIR).values &
               (msg_df['type'] == LIMIT_ORDER_SUBMITTED).values)

cancel_sell_idx = ((msg_df['direction'] == SELL_DIR).values &
                   (msg_df['type'] == FULL_CANCEL).values)
cancel_buy_idx = ((msg_df['direction'] == BUY_DIR).values &
                  (msg_df['type'] == FULL_CANCEL).values)

avg_cancel_sell_size = np.mean(msg_df.ix[cancel_sell_idx]['size'])
avg_cancel_buy_size = np.mean(msg_df.ix[cancel_buy_idx]['size'])

lim_sell_dist_ask = pd.Series(msg_df.ix[lim_sell_idx]['price'].values - 
                              ob_df.ix[lim_sell_idx][ASK_PRICE_COL + '1'].values)
lim_buy_dist_bid = pd.Series(ob_df.ix[lim_buy_idx][BID_PRICE_COL + '1'].values -
                             msg_df.ix[lim_buy_idx]['price'].values)

cancel_sell_dist_ask = pd.Series(msg_df.ix[cancel_sell_idx]['price'].values -
                                 ob_df.ix[cancel_sell_idx][ASK_PRICE_COL + '1'].values)
cancel_buy_dist_bid = pd.Series(ob_df.ix[cancel_buy_idx][BID_PRICE_COL + '1'].values-
                                msg_df.ix[cancel_buy_idx]['price'].values)

lim_sell_dist_ask = lim_sell_dist_ask.round(decimals=2)
lim_buy_dist_bid = lim_buy_dist_bid.round(decimals=2)
cancel_sell_dist_ask = cancel_sell_dist_ask.round(decimals=2)
cancel_buy_dist_bid = cancel_buy_dist_bid.round(decimals=2)

# Calculate number of limit sell orders i ticks from the ask

n_la = []
n_lb = []
n_ca = []
n_cb = []
for i in range(1, DEPTH+1):
    n_la.append(len(lim_sell_dist_ask.ix[lim_sell_dist_ask == float(i)*100]))
    n_lb.append(len(lim_buy_dist_bid.ix[lim_buy_dist_bid == float(i)*100]))
    n_ca.append(len(cancel_sell_dist_ask.ix[cancel_sell_dist_ask == float(i)*100]))
    n_cb.append(len(cancel_buy_dist_bid.ix[cancel_buy_dist_bid == float(i)*100]))
    
lam_hat_ask = [ x/trading_time for x in n_la ]
lam_hat_bid = [ x/trading_time for x in n_lb ]

q_ia = []
q_ib = []
for i in range(DEPTH):
    bid_size_change_df = pd.concat({'bid_volume':ob_df[BID_SIZE_COL+str(i+1)], 
                                    'bid_volume_1':ob_df[BID_SIZE_COL+str(i+1)].shift()}, 
                                   axis=1)
    ask_size_change_df = pd.concat({'ask_volume':ob_df[ASK_SIZE_COL+str(i+1)], 
                                    'ask_volume_1':ob_df[ASK_SIZE_COL+str(i+1)].shift()}, 
                                   axis=1)
    bid_size_delta = (bid_size_change_df['bid_volume'][1:] -
                      bid_size_change_df['bid_volume_1'][1:])
    ask_size_delta = (ask_size_change_df['ask_volume'][1:] - 
                      ask_size_change_df['ask_volume_1'][1:])
    q_ia.append(np.mean(bid_size_delta.ix[bid_size_delta > 0])/avg_lim_sell_size)
    q_ib.append(np.mean(ask_size_delta.ix[ask_size_delta > 0])/avg_lim_buy_size)

theta_hat_ask = [ ((n_ca[i] * 
                    (avg_cancel_sell_size /
                     avg_lim_sell_size)) /
                   (trading_time*q_ia[i])) for i in range(len(n_ca)) ]
theta_hat_bid = [ ((n_cb[i] * 
                    (avg_cancel_buy_size /
                     avg_lim_sell_size)) /
                   (trading_time*q_ib[i])) for i in range(len(n_cb)) ]


##### Calc of 'service' times and emp prob of either service occurring #####

mkt_buy_idx = ((msg_df['direction'] == BUY_DIR).values &
               ((msg_df['type'] == HIDDEN_ORDER_EXEC).values |
                (msg_df['type'] == VISIBLE_ORDER_EXEC).values))

mkt_sell_idx = ((msg_df['direction'] == SELL_DIR).values &
                ((msg_df['type'] == HIDDEN_ORDER_EXEC).values |
                 (msg_df['type'] == VISIBLE_ORDER_EXEC).values))

mkt_buy_lag_df = pd.concat({'unlagged':msg_df[mkt_buy_idx]['timestamp'], 
                            'lagged':msg_df[mkt_buy_idx]['timestamp'].shift()},
                           axis=1)

mkt_sell_lag_df = pd.concat({'unlagged':msg_df[mkt_sell_idx]['timestamp'],
                             'lagged':msg_df[mkt_sell_idx]['timestamp'].shift()},
                            axis=1)

mkt_buy_ia_time = mkt_buy_lag_df['unlagged'] - mkt_buy_lag_df['lagged']
mkt_sell_ia_time = mkt_sell_lag_df['unlagged'] - mkt_sell_lag_df['lagged']

avg_mkt_buy_service_time = np.mean(mkt_buy_ia_time.ix[mkt_buy_ia_time > 0])
avg_mkt_sell_service_time = np.mean(mkt_sell_ia_time.ix[mkt_sell_ia_time > 0])

n_mkt_buys = len(msg_df[mkt_buy_idx])
n_mkt_sells = len(msg_df[mkt_sell_idx])

avg_cancel_buy_svc_time = []
avg_cancel_sell_svc_time = []

for i in range(1,DEPTH+1):
    buy_idx = ((msg_df['type'] == FULL_CANCEL).values &
               (msg_df['direction'] == BUY_DIR).values &
               ((ob_df[BID_PRICE_COL+str(i)].values - 
                 msg_df['price'].values).round(2) == float(i)*100))

    sell_idx = ((msg_df['type'] == FULL_CANCEL).values &
                (msg_df['direction'] == SELL_DIR).values &
                ((msg_df['price'].values - 
                  ob_df[ASK_PRICE_COL+str(i)].values).round(2) == float(i)*100))

    if i == 1:
        n_cancel_buys_best_prc = len(msg_df[buy_idx])
        n_cancel_sells_best_prc = len(msg_df[sell_idx])

    cancel_buy_lag_df = pd.concat({'unlagged':msg_df[buy_idx]['timestamp'], 
                                   'lagged':msg_df[buy_idx]['timestamp'].shift()},
                                  axis=1)    
    cancel_sell_lag_df = pd.concat({'unlagged':msg_df[sell_idx]['timestamp'],
                                    'lagged':msg_df[sell_idx]['timestamp'].shift()},
                                   axis=1)
    
    cancel_buy_ia_time = cancel_buy_lag_df['unlagged'] - cancel_buy_lag_df['lagged']
    cancel_sell_ia_time = cancel_sell_lag_df['unlagged'] - cancel_sell_lag_df['lagged']
    
    avg_cancel_buy_svc_time.append(np.mean(cancel_buy_ia_time.ix[cancel_buy_ia_time > 0]))
    avg_cancel_sell_svc_time.append(np.mean(cancel_sell_ia_time.ix[cancel_sell_ia_time > 0]))

# Probability of a market order being the service vehicle
p_mkt_buy_svc = float(n_mkt_buys)/(n_cancel_buys_best_prc+n_mkt_buys)
p_mkt_sell_svc = float(n_mkt_sells)/(n_cancel_sells_best_prc+n_mkt_sells)

print '##### Discrete Simulation Parameters (rates are per second) #####'
print r'\hat{\lambda_a} =', lam_hat_ask
print r'\hat{\lambda_b} =', lam_hat_bid
print r'\hat{\mu_a} =', mu_hat_ask
print r'\hat{\mu_b} =', mu_hat_bid
print r'\hat{\theta_a} =', theta_hat_ask
print r'\hat{\theta_b} =', theta_hat_bid
print ''
print '##### Continuous Time Parameters #####'
print 'Avg Mkt Buy Interarrival Time:', avg_mkt_buy_service_time, 'seconds'
print 'Avg Mkt Sell Interarrival Time:', avg_mkt_sell_service_time, 'seconds'
print 'Avg Cancel Buy Interarrival Times (by queue depth #):', avg_cancel_buy_svc_time, 'seconds'
print 'Avg Cancel Sell Interarrival Times (by queue depth #):', avg_cancel_sell_svc_time, 'seconds'
print ''
print '##### Exponential Mixture Model Weights #####'
print 'Market Buy Orders:', p_mkt_buy_svc
print 'Market Sell Orders:', p_mkt_sell_svc
print 'Cancel Buy Orders:', (1 - p_mkt_buy_svc)
print 'Cancel Sell Orders:', (1 - p_mkt_sell_svc)
