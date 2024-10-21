#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 16:55:43 2024

@author: kevinyin
"""

import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
import os

from numba import njit




#%% initialize

# directory
directory = "/Users/kevinyin/Documents/Research/PhD/sov_debt_proj"
os.chdir(directory) 

# set (odd) number of discrete possible bond values (N), income values (M)
N = 301
M = 21

# parameters
r = 0.0117 # interest (US 5 year bond quarterly yield)
#r = 0.06 # interest (US 5 year bond quarterly yield)
#sigma = 2 # risk aversion
sigma = 4
rho = 0.945 # 
eta = 0.025 # 

beta = 1 # impatience
delta = 0.7
#theta = 0.282  # re-entry probability
theta = 0.1  # re-entry probability

# NEW PARAMS
#gmin_vec = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36]
gmin_vec = [0.238]
#, 0.2, 0.15, 0.1, 0.05] # minimum government spending
tau = 0.3 # tax rate


# initialize discretized asset values
b_grid = np.linspace(-1, 1, N)


# define y-markov process
mc = qe.tauchen(n=M, rho=rho, sigma=eta)
y_grid = np.exp(mc.state_values)
y_grid_def = y_grid
#y_grid_def = np.where(0.969 * np.mean(y_grid) < y_grid, 0.969 * np.mean(y_grid), y_grid)
P = mc.P




#%% functions

# store means for each gmin
b_mean_vec = []
q_mean_vec = []


# loop over varied parameter
for gmin in gmin_vec:

    '''
    Consumption
    '''
    # consumption function defined by budget constraint
    @njit
    def consume(y,b_0,b_1,q,d):
        if d == 1:
            c = tau * y - q * b_1 + b_0
        if d == 0:
            c = tau * y
        return c
    
    
    '''
    Utility
    '''
    # utility function
    @njit
    def utility(c, sigma):
        u = (np.power(c - gmin, 1-sigma)) / (1-sigma)
        return u
    
    
    '''
    Updates
    '''
    @njit
    def update_C(y_grid, b_grid, price, sigma):
        '''
        Returns:
            1) Good standing consumption matrix (N*M x N)
            2) Bad standing consumption matrix (N*M x 1)
            3) Good standing utility matrix (N*M x N)
            4) Bad standing utility matrix (N*M x 1)
        '''
        # initialize C matrix
        C_mat_G = np.zeros((N*M,N)) # in good standing, B' is a choice variable
        C_mat_B = np.zeros((N*M)) # in bad standing, you cannot choose B' so there is no dimension to optimize over
        
        # the n-th row of the consumption matrix is N * i + j, where i is the y-index and j is the bond-index
        for i in range(len(y_grid)):
            for j in range(len(b_grid)): # initial bond purchases (row)
                for k in range(len(b_grid)): # b prime (column)
                    C_mat_G[N*i+j][k] = consume(y_grid[i],b_grid[j],b_grid[k],price[N*i+k],1) # tried k instead of j in the price
                    C_mat_B[N*i+j] = consume(y_grid_def[i],b_grid[j],b_grid[k],price[N*i+k],0)
            
        #C_mat_G = np.where(C_mat_G<=0, 0.00001, C_mat_G)
        #C_mat_B = np.where(C_mat_B<=0, 0.00001, C_mat_B)
        
        # fill U matrix
        U_mat_G = utility(C_mat_G, sigma)
        U_mat_B = utility(C_mat_B, sigma)
        
        return C_mat_G, C_mat_B, U_mat_G, U_mat_B 
    
    
    #@njit
    def update_V(V_old_O, V_old_B, U_mat_G, U_mat_B, P, beta, delta, theta, sigma):
        '''
        Returns:
            1) Good standing consumption matrix (N*M x N)
            2) Bad standing consumption matrix (N*M x 1)
            3) Good standing utility matrix (N*M x N)
            4) Bad standing utility matrix (N*M x 1)
        '''
        # initialize
        exp_V_mat_O = np.zeros((M*N,N))
        exp_V_mat_B = np.zeros((M*N))
        
        # create a list of initial (y,b) specific indices
        index_list = []
        for b in range(N):
            y_b_index = []
            for y in range(M):
                y_b_index.append(y*N+b)
            index_list.append(y_b_index)
        
        # for each initial (y,b) combination, compute the expected value of the O-VF and bad-standing B-VF
        for y in range(M):
            for idx in index_list: # for each initial bond value
                V_mat_O = np.repeat(V_old_O[idx].reshape(1,M), repeats=N, axis=0)
                exp_V_mat_O[y*N:y*N+N,idx[0]] = V_mat_O @ P[y] 
                exp_V_mat_B[y*N:y*N+N] = V_old_B[idx] @ P[y]
                
        # update entries of continue-value functions in good and bad standing for bond value (b) positions only
        W_new_G = np.max(U_mat_G + beta * delta * exp_V_mat_O, axis=1) 
        V_new_G = np.max(U_mat_G + delta * exp_V_mat_O, axis=1) 
        W_new_B = U_mat_B + beta * delta * (theta * exp_V_mat_O[:,N//2] + (1-theta) * exp_V_mat_B)
        V_new_B = U_mat_B + delta * (theta * exp_V_mat_O[:,N//2] + (1-theta) * exp_V_mat_B)
        
        # choose whether or not to default
        V_new_O = np.where(W_new_G > W_new_B, V_new_G, V_new_B) # if W_G > W_B, value is V_G, otherwise V_B

        # policy indices
        b_choice = np.argmax(U_mat_G + beta * delta * exp_V_mat_O, axis=1)
        
        # track of whether or not the sovereign defaults for a given value of Y and B (0 indicates repay)
        d_choice = np.argmax(np.stack((W_new_G, W_new_B)).transpose(),axis=1)
        
        return V_new_G, V_new_B, V_new_O, b_choice, d_choice
    
    
    #@njit
    def update_Q(d_choice, P, r):
        '''
        Default Probabilities and Bond Prices
        '''
        # update price function
        price_new = np.zeros((N*M))
        arr_delta = np.zeros((M,N))
        
        # calculate ex ante default probabilities
        for b in range(N):
            
            # create list of indices for all the b-indexed incomes
            y_b_index = []
            for y in range(M):
                y_b_index.append(y*N+b)
            
            arr_delta[:,b] = P @ d_choice[y_b_index]
            price_new[y_b_index] = (1 - arr_delta[:,b]) / (1+r)
        
        return price_new, arr_delta
    
    
    # simulation
    def simulate(T, b_choice, y_grid, b_grid):
        '''
        Simulate Income Process and Policy Responses
        '''
        y_sim = []
        b_sim = []
        d_sim = []
        q_sim = []
        
        # simulate income process
        y_sim = np.exp(mc.simulate(T))
        
        # initialize debt value at zero
        b_t = b_grid[int(N/2)]
        d_t = 0
        
        for t in range(T):
            
            # yesterday's debt level and income don't depend on today's default
            y_idx = np.argmin(np.abs(y_sim[t] - y_grid))
            b1_idx = np.argmin(np.abs(b_t - b_grid))
            
            # if access to financial markets
            if d_t == 0:
                default = d_choice[N * y_idx + b1_idx]
                
                if default == 0: 
                    b2_idx = b_choice[N * y_idx + b1_idx]
                    
                else:
                    b2_idx = int(N/2)
                    d_t = 1
            
            # if barred from financial markets
            else:
                b2_idx = int(N/2)
                
            # transition out of autarky with probability theta
            if d_t == 1:
                if np.random.uniform() > theta:
                    d_t = 1
                else:
                    d_t = 0
            
            b_t = b_grid[b2_idx]
            b_sim.append(b_t)
            d_sim.append(d_t)
            q_sim.append(price_new[y_idx*N+b2_idx])
            
        return y_sim, b_sim, d_sim, q_sim
    
    
    
    
    
    #%% function iteration 
    
    # initialize guess for V_new
    price_norm = 10
    sup_norm = 10
    counter = 0
    V_old_G = np.zeros(N*M)
    V_old_B = np.zeros(N*M)
    V_old_O = np.ones(M*N) 
    
    
    # initialize guess for price function Q * 1 / (1+r)
    price_old = np.ones(N*M) 
    
    # create trace list
    V_trace_G = []
    V_trace_B = []
    V_trace_O = []
    price_trace = []
    
    # solver
    while sup_norm > 0.05:
        
        price_trace.append(price_old)
            
        '''
        Consumption and Utility
        '''
        C_mat_G, C_mat_B, U_mat_G, U_mat_B = update_C(y_grid, b_grid, price_old, sigma)
        
        # utility for positive net consumption (c - gmin) should always be negative
        U_mat_G[U_mat_G > 0] = -100000000000000000000
        U_mat_B[U_mat_B > 0] = -100000000000000000000
    
        '''
        Value Functions and Policy Functions
        '''
        V_new_G, V_new_B, V_new_O, b_choice, d_choice = update_V(V_old_O, V_old_B, U_mat_G, U_mat_B, P, beta, delta, theta, sigma)
    
        '''
        Tracking and Updating
        '''
        if counter % 10 == 0:
            V_trace_G.append(V_new_G)
            V_trace_B.append(V_new_B)
            V_trace_O.append(V_new_O)
        counter += 1
        sup_norm = np.linalg.norm(V_new_O - V_old_O)
        #np.max(np.abs(V_new_G - V_old_G)) + np.max(np.abs(V_new_B - V_old_B))
        print("Distance between VF:", sup_norm)
        
        # re-initialize value functions
        V_old_G = V_new_G
        V_old_B = V_new_B
        V_old_O = V_new_O
        
        '''
        Defaults and Prices
        '''
        price_new, arr_delta = update_Q(d_choice, P, r)
        price_old = price_new
        
        '''
        TEMP: Plot price and VF
        '''
        # define low, medium, and high income states
        y_L = np.argmin(np.abs(y_grid - 0.95 * np.mean(y_grid)))
        y_M = np.argmin(np.abs(y_grid - np.mean(y_grid)))
        y_H = np.argmin(np.abs(y_grid - 1.05 * np.mean(y_grid)))
    
        # select price functions for each income level
        for y in [y_L,y_H]:
                
            # create list of indices for all the b-indexed incomes
            y_b_index = []
            for b in range(N):
                y_b_index.append(y*N+b)
            
            price_y = price_new[y_b_index]
            plt.plot(b_grid,price_y)
        
        #plt.xlabel("Assets")
        #plt.ylabel("Price")
        #plt.show()
    
    
            
            
    
    #%% plots
    
    # define low, medium, and high income states
    y_L = np.argmin(np.abs(y_grid - 0.95 * np.mean(y_grid)))
    y_M = np.argmin(np.abs(y_grid - np.mean(y_grid)))
    y_H = np.argmin(np.abs(y_grid - 1.05 * np.mean(y_grid)))
    
    
    
    '''
    Price Function Plot w/ Arellano Bounds
    '''
    # select price functions for each income level
    for y in [y_L,y_H]:
            
        # create list of indices for all the b-indexed incomes
        y_b_index = []
        for b in range(N):
            y_b_index.append(y*N+b)
        
        price_y = price_new[y_b_index]
        plt.plot(b_grid,price_y)
    
    plt.title("Price Function, G-Min=" + str(gmin) + ", beta=" + str(beta) + ", sigma=" + str(sigma))
    plt.xlabel("Assets")
    plt.ylabel("Debt Policy")
    plt.show()
    
    
    # compute policy function
    for y in [y_L,y_H]:
            
        # create list of indices for all the b-indexed incomes
        y_b_index = []
        for b in range(N):
            y_b_index.append(y*N+b)
        
        b_choice_y = b_choice[y_b_index]
        #plt.plot(b_grid,b_choice_y)
    
    #plt.xlim(-0.35, 0)
    #plt.xlabel("Assets")
    #plt.ylabel("Debt Policy")
    #plt.show()
    
    
    #for i in range(len(price_trace)):
    #    plt.plot(b_grid, price_trace[i][3010:3311], linewidth=0.5, color='red')
    #plt.title('Value Function (H) Convergence (every 10 iters)')
    #plt.xlabel('B')
    #plt.ylabel('V_G', rotation=0, labelpad=15)
    #plt.show()\
    
    
    
    '''
    Value Function Plot w/ Arellano Bounds
    '''
    # select value functions for each income level
    for y in [y_L,y_H]:
            
        # create list of indices for all the b-indexed incomes
        y_b_index = []
        for b in range(N):
            y_b_index.append(y*N+b)
        
        value_y = V_new_O[y_b_index]
        #plt.plot(b_grid,value_y)
    
    #plt.xlim(-0.4, 0.4)
    #plt.show()
    
    
    
    '''
    Default Probability Plot
    '''
    fig, ax = plt.subplots(figsize=(10, 6.5))
    hm = ax.pcolormesh(b_grid[:160], y_grid, arr_delta[:,0:160], cmap='plasma')
    cax = fig.add_axes([.92, .1, .02, .8])
    fig.colorbar(hm, cax=cax)
    ax.set_xlabel('Debt', fontsize=16)
    ax.set_ylabel('Income', fontsize=16)
    ax.set_title("Default Probabilities, G-Min=" + str(gmin) + ", beta=" + str(beta) + ", sigma=" + str(sigma), fontsize=18)
    plt.show()
    
    
    
    #%% simulation
    
    T = 500
    
    y_sim, b_sim, d_sim, q_sim = simulate(T, b_choice, y_grid, b_grid)
    
    plt.plot(y_sim)
    plt.title("Output TS, G-Min=" + str(gmin) + ", beta=" + str(beta) + ", sigma=" + str(sigma))
    plt.xlabel("Time")
    plt.ylabel("Output")
    for t in range(T):
        d_t = d_sim[t]
        if d_t == 1:
            plt.axvline(x=t, color='r', linewidth=2, alpha=0.4)
    plt.show()
    
    
    plt.plot(b_sim)
    plt.title("Asset Position TS, G-Min=" + str(gmin) + ", beta=" + str(beta) + ", sigma=" + str(sigma))
    plt.xlabel("Time")
    plt.ylabel("Foreign Assets")
    for t in range(T):
        d_t = d_sim[t]
        if d_t == 1:
            plt.axvline(x=t, color='r', linewidth=2, alpha=0.4)
    plt.show()
    


    #%% variable means

    y_means = []
    b_means = []
    q_means = []    
    
    # run 100 simulations of the model
    for i in range(0,101):
        y_sim, b_sim, d_sim, q_sim = simulate(T, b_choice, y_grid, b_grid)
        y_means.append(np.mean(y_sim))
        b_means.append(np.mean(b_sim))
        q_means.append(np.mean(q_sim))
        
    # calculate mean of means
    #print("G-Min=" + str(gmin) + ", Median Ouput:", np.median(y_means))
    print("G-Min=" + str(gmin) + ", Median Assets:", np.median(b_means))
    
    b_mean_vec.append(np.median(b_means))
    q_mean_vec.append(np.median(q_means))

    
    
#%%

# plot how the mean changes with gmin

plt.plot(gmin_vec, b_mean_vec)
plt.title("Savings and Minimum Spending")
plt.xlabel("G-Min")
plt.ylabel("Mean Assets (Savings)")
plt.show()
        
plt.plot(gmin_vec, q_mean_vec)
plt.title("Average Bond Price and Minimum Spending")
plt.xlabel("G-Min")
plt.ylabel("Price")
plt.show()
            
    
    
    
    
    
    
    