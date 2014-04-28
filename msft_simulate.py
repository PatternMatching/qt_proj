#!/usr/bin/env python

import numpy as np
import pylab as plt
import pandas as pd

DEPTH = 5

SIM_STEPS = 100000

def prob_incr(l, m, t, n):
    return l/(l + m + (n*t))

def simulate(l, m, t, n_steps):
    orders = 0
    states = []
    for i in range(n_steps):
        p_incr = prob_incr(l, m, t, orders)
        r = np.random.rand()
        if (r < p_incr) | (orders == 0):
            orders += 1
        else:
            orders -= 1

        states.append(orders)
    states = pd.Series(states)
    return states

if __name__ == "__main__":
    lam =  1.85
    mu = 0.94
    theta = 0.71
    orders = 0
    states = []

    states = simulate(lam, mu, theta, SIM_STEPS)

    plt.figure()
    plt.plot(states[:200])
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
