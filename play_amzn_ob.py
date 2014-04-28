#!/user/bin/env python

import pandas as pd
import numpy as np

MSG_FILENAME = 'MSFT_2012-06-21_34200000_57600000_message_5.csv'
OB_FILENAME = 'MSFT_2012-06-21_34200000_57600000_orderbook_5.csv'

LIMIT_ORDER_SUBMITTED = 1
PARTIAL_CANCEL = 2
FULL_CANCEL = 3
VISIBLE_ORDER_EXEC = 4
HIDDEN_ORDER_EXEC = 5

LEVEL = 10

# Both in seconds
TRADING_ST_TIME = 9.5*60*60
TRADING_END_TIME = 16*60*60

# Read in the message file
msft_msg_df = pd.read_csv(MSG_FILENAME, index_col = 0)
msft_ob_df = pd.read_csv(OB_FILENAME)

"""
Returns the mid price given a bid/ask pair
"""
def mid_price(ask, bid):
    return (ask+bid)/2

# Want to eliminate entries that correspond to orders outside of
# the exchange trading hours

msft_msg_df = msft_msg_df.ix[ (msft_msg_df.index >= TRADING_ST_TIME) &
                              (msft_msg_df.index <= TRADING_END_TIME) ]

msft_msg_df[ 'price' ] = msft_msg_df[ 'price' ]/10000.00

type_count_df = msft_msg_df.groupby('type').size()

num_arrivals = type_count_df[LIMIT_ORDER_SUBMITTED]
num_departures = (type_count_df[HIDDEN_ORDER_EXEC] + 
                  type_count_df[VISIBLE_ORDER_EXEC])
num_cancels = type_count_df[FULL_CANCEL]

# Now calculate empirical arrival rate of limit orders

minutes = (TRADING_END_TIME - TRADING_ST_TIME)/60.0

limit_order_df = msft_msg_df.ix[ msft_msg_df['type'] == LIMIT_ORDER_SUBMITTED ]
mkt_order_df = msft_msg_df.ix[ (msft_msg_df['type'] == VISIBLE_ORDER_EXEC) | 
                               (msft_msg_df['type'] == HIDDEN_ORDER_EXEC) ]
cancel_order_df = msft_msg_df.ix[ msft_msg_df['type'] == FULL_CANCEL ]

# Calculate long run averages of behavior
avg_limit_order_size = np.mean( limit_order_df['size'] )
avg_mkt_order_size = np.mean( mkt_order_df['size'] )
avg_cancel_order_size = np.mean( cancel_order_df['size'] )

# Math for parameter estimation from Cont et al (2010)
lambda_hat = num_arrivals/minutes
mu_hat = (num_departures/minutes)*(avg_mkt_order_size/avg_limit_order_size)
theta_hat = num_cancels/minutes

print r'\hat{\lambda} = ' + str(lambda_hat)
print r'\hat{\mu} = ' + str(mu_hat)
print r'\hat{\theta} = ' + str(theta_hat)

