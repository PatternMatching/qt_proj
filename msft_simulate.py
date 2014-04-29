#!/usr/bin/env python

import numpy as np
import pylab as plt
import pandas as pd

DEPTH = 5

SIM_STEPS = 1000000

def calc_svc_rate(m,t,n):
    return m + t

def prob_incr(l, m, t, n):
    return l/(l + m + (t))

def simulate(l, m, t, n_steps):
    orders = 0
    incr = 0
    states = []
    for i in range(n_steps):
        p_incr = prob_incr(l, m, t, orders)
        r = np.random.rand()
        if (r < p_incr):
            orders += 1
            incr += 1
        else:
            orders = max(orders - 1, 0)

        states.append(orders)
    states = pd.Series(states)
    print 'Prob of increment:',float(incr)/n_steps
    return states

def t_pn(l, m, t, n):
    return (theor_p0(l,m,t,n) * 
            (l**n)/np.product([calc_svc_rate(m,t,x) for x in range(1,n+1)]))

def emp_p0(states):
    return len(states.ix[states == 0])/float(len(states))
    
def emp_pn(states, n):
    return len(states.ix[states == n])/float(len(states))

def theor_p0(l, m, t, n):
    total = 0
    for i in range(1,n+1):
        num = l**i
        den = np.product([calc_svc_rate(m,t,x) for x in range(1,i+1)])
        total += num/den
        #print 'theor_p0: i =',i,'num =',num,'den =',den
    return 1/(1 + total)

if __name__ == "__main__":
    fake_data = False
    if fake_data:
    	lam = 1.85
    	mu = 0.94
    	theta = 2
    else:
        lam = 4.7498
        mu = 0.1177
        theta = 10.9278
    
    orders = 0
    states = []

    plot = False

    states = simulate(lam, mu, theta, SIM_STEPS)

    print 'Empirical p_0 =', emp_p0(states)
    print 'Theoretical p_0 =', theor_p0(lam,mu,theta,12)

    if plot:
        plt.figure()
        plt.plot(states[:2000])
        plt.show()

"""
class BestPriceQueue(object):
    def __init__(self, lam, mu, theta):
        self.states = []
        self.limit_orders = 0
        self.lam = lam
        self.mu = mu
        self.theta = theta
        # self.p_incr = lam/(lam + mu + self.limit_orders*theta)
        # print self.p_incr
        # self.p_decr = 1 - self.p_incr
        
    def update_probs():
        self.p_incr = self.lam / (self.lam + 
                                  self.mu + 
                                  self.limit_orders*self.theta)

    def step():
        r = np.random.rand()
        if r < self.p_incr:
            self.limit_orders += 1
        else:
            self.limit_orders -= 1
        self.states.append(self.limit_orders)
        self.update_probs()

    def multi_step(num_steps):
        for i in range(num_steps):
            self.step()

    def print_states():
        print self.states

    def reset():
        self.limit_orders = 0
        self.states = []
"""
