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

import warnings
warnings.filterwarnings("ignore")




#%% initialize

# directory
directory = "/Users/kevinyin/Documents/Research/PhD/sov_debt_proj"
os.chdir(directory) 

# set (odd) number of discrete possible bond values (N), income values (M)
N = 301
M = 11

# parameters
r = 0.0117 # interest (US 5 year bond quarterly yield)
sigma = 2 # risk aversion
rho = 0.945 # 0.945
eta = 0.025 # 0.025

p_d = 0 # probability of disaster
p_s = 0.2 # probability of staying in a disaster
d_size = 0.70 # disaster loses 20% of GDP

beta = 0.953 # impatience
theta = 0.282  # re-entry probability

# NEW PARAMS
gmin_vec = [0]
gmin = 0
tau = 1 # tax rate (no taxes in this model)

# LTD params
kappa = 0.03
Lambda = 0.05


# initialize discretized asset values
b_grid = np.linspace(-0.6, 0.6, N)


plt.rcParams.update({"font.family": "Helvetica"})





#%% functions

#p_d_vec = [0, 0.0025, 0.005, 0.0075, 0.01]
p_d_vec = [0, 0.005, 0.01, 0.015, 0.02]
sigma_vec = [2, 2.5, 3, 3.5, 4]

beta_vec = [0.95, 0.96, 0.97]

p_s_vec = [0, 0.25, 0.5, 0.75, 0.99]
p_s_vec = [0.15]

# store means for each parameter value
b_mean_vec = []
q_mean_vec = []
eq_prem_mat = []
b_mat = []


    
# loop over varied parameter
for beta in beta_vec:
    
    eq_prem_vec = []
    b_mean_vec = []
    q_mean_vec = []
    for p_d, sigma in zip(p_d_vec,sigma_vec):
        
        # name parameters for figures
        param = p_d
        param_name = "Disaster Probability"
        param_vec = p_d_vec
        
    
        # initialize values for loop
        M = 21
        mc = qe.tauchen(n=M, rho=rho, sigma=eta)
        y_grid = np.exp(mc.state_values)
        P = mc.P # column index tells you the starting state, row tells you the end state
        
        # TEMP: create a separate y_grid for pure arellano with no disaster risk
        mc_arr = qe.tauchen(n=M, rho=rho, sigma=eta)
        y_grid_arr = np.exp(mc.state_values)
        P_arr = mc_arr.P # column index tells you the starting state, row tells you the end state
    
        # create new y_grid and new probability matrix
        y_grid_disast = y_grid * d_size
        y_grid = np.vstack((y_grid.reshape(-1,1), y_grid_disast.reshape(-1,1)))
        y_grid = y_grid.flatten()
        
    
        # calculate transition probabilities to disasters
        P_s2d = P * p_d
        P_s2s = P * (1-p_d)
        P_d2s = P * (1-p_s)
        P_d2d = P * p_s
    
        # concatenate probability matrices
        P_s = np.hstack((P_s2s, P_s2d))
        P_d = np.hstack((P_d2s, P_d2d))
        P = np.vstack((P_s, P_d))
    
        M = len(y_grid)
        
        # NOTE: Which average you take matters a lot! You cannot use y_grid AFTER you'd added disaster risk
        y_grid_def = np.where(0.969 * np.mean(y_grid_arr) < y_grid, 0.969 * np.mean(y_grid_arr), y_grid)
        #y_grid_def = np.where(0.969 * np.mean(y_grid) < y_grid, 0.969 * np.mean(y_grid), y_grid)
        
        '''
        Consumption
        '''
        # consumption function defined by budget constraint
        @njit
        def consume(y,b_0,b_1,q,d):
            if d == 1:
                c = tau * y - q * (b_1 - (1 - Lambda) * b_0) + (Lambda + (1-Lambda) * kappa) * b_0
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
            
            # crucial for decimal values of sigma
            C_mat_G = np.where(C_mat_G<=0, 0.0000001, C_mat_G)
            C_mat_B = np.where(C_mat_B<=0, 0.0000001, C_mat_B)
            
            # fill U matrix
            U_mat_G = utility(C_mat_G, sigma)
            U_mat_B = utility(C_mat_B, sigma)
            
            return C_mat_G, C_mat_B, U_mat_G, U_mat_B 
        
        
        #@njit
        def update_V(V_old_O, V_old_B, U_mat_G, U_mat_B, P, beta, theta, sigma):
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
                    
                    # TEMP: compare distance between value functions  
                    #V_mat_O_arr = np.repeat(V_old_O[0:N*M1][idx].reshape(1,M1), repeats=N, axis=0)
                    #print(np.linalg.norm(V_mat_O - V_mat_O_arr))
                    
                    exp_V_mat_O[y*N:y*N+N,idx[0]] = V_mat_O @ P[y] 
                    exp_V_mat_B[y*N:y*N+N] = V_old_B[idx] @ P[y]
                    
                    # TEMP: compare distance between expectations
                    #exp_V_mat_O[y*N:y*N+N,idx[0]] = V_mat_O @ P_arr[y] 
                    #exp_V_mat_B[y*N:y*N+N] = V_old_B[idx] @ P_arr[y]
                  
                    
            # update entries of continue-value functions in good and bad standing for bond value (b) positions only
            V_new_G = np.max(U_mat_G + beta * exp_V_mat_O, axis=1) 
            V_new_B = U_mat_B + beta * (theta * exp_V_mat_O[:,N//2] + (1-theta) * exp_V_mat_B)
            
            # FIXME: Compare expected value matrices between the no-disaster and disaster cases
            # did this, they are different 
            
            # choose whether or not to default
            V_new_O = np.maximum(V_new_G, V_new_B)
            
            # policy indices
            b_choice = np.argmax(U_mat_G + beta * exp_V_mat_O, axis=1)
            
            # track of whether or not the sovereign defaults for a given value of Y and B (0 indicates repay)
            d_choice = np.argmax(np.stack((V_new_G, V_new_B)).transpose(),axis=1)
            
            return V_new_G, V_new_B, V_new_O, b_choice, d_choice
        
        
        #@njit
        def update_Q(d_choice, price_old, P, r):
            '''
            Default Probabilities and Bond Prices
            '''
            # update price function
            price_new = np.zeros((N*M))
            delta = np.zeros((M,N))
            
            # calculate ex ante default probabilities
            for b in range(N):
                
                # create list of indices for all the b-indexed incomes
                y_b_index = []
                for y in range(M):
                    y_b_index.append(y*N+b)
                
                delta[:,b] = P @ d_choice[y_b_index]
                price_new[y_b_index] = (1 - delta[:,b]) * (Lambda + (1-Lambda) * (kappa + price_old[y_b_index])) / (1+r) #LTD
            
            return price_new, delta
        
        
        # simulate a Markov chain given P and Y
        def simulate_mc(P, Y, T):
            """
            Simulates a Markov chain.
            
            Parameters:
            P (np.ndarray): An N x M transition matrix where rows are end states, columns are starting states.
            Y (np.ndarray): An M x 1 state vector, representing the initial state probabilities.
            steps (int): The number of steps to simulate.
        
            Returns:
            np.ndarray: A vector of states after the simulation.
            """
            current_state = Y[int(M/4)] # initialize at 100% of GDP
            states = [current_state]
        
            for t in range(T):
                current_idx = np.argmin(np.abs(Y - current_state))
                next_state = np.random.choice(Y, p=P[current_idx])
                states.append(next_state)
                current_state = next_state
        
            return np.array(states)
        
        
        
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
            y_sim = simulate_mc(P,y_grid,T)
            # FIX ME: need to have the simulation account for disasters
            
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
                b_sim.append(b_t * 100)
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
            V_new_G, V_new_B, V_new_O, b_choice, d_choice = update_V(V_old_O, V_old_B, U_mat_G, U_mat_B, P, beta, theta, sigma)
            '''
            Tracking and Updating
            '''
            if counter % 10 == 0:
                V_trace_G.append(V_new_G)
                V_trace_B.append(V_new_B)
                V_trace_O.append(V_new_O)
            counter += 1
            sup_norm = np.linalg.norm(V_new_O - V_old_O)
            print("Distance to Convergence:", sup_norm)

            # re-initialize value functions
            V_old_G = V_new_G
            V_old_B = V_new_B
            V_old_O = V_new_O
            
            '''
            Defaults and Prices
            '''
            price_new, delta = update_Q(d_choice, price_old, P, r)
            price_old = price_new
            
            '''
            TEMP: Plot price and VF
            '''
            # define low, medium, and high income states
            y_L = np.argmin(np.abs(y_grid - 0.95 * np.mean(y_grid_arr)))
            y_M = np.argmin(np.abs(y_grid - np.mean(y_grid_arr)))
            y_H = np.argmin(np.abs(y_grid - 1.05 * np.mean(y_grid_arr)))
        
        
    
                
        
        #%% plots
        
        # define low, medium, and high income states
        y_L = np.argmin(np.abs(y_grid - 0.95 * np.mean(y_grid_arr)))
        y_M = np.argmin(np.abs(y_grid - np.mean(y_grid_arr)))
        y_H = np.argmin(np.abs(y_grid - 1.05 * np.mean(y_grid_arr)))
        
        
        
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
        
        plt.title("Price Function, Disas. Prb=" + str(p_d) + ", beta=" + str(beta) + ", sigma=" + str(sigma))
        plt.xlabel("Assets")
        plt.ylabel("Price")
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
        hm = ax.pcolormesh(b_grid[:160], y_grid, delta[:,0:160], cmap='plasma')
        cax = fig.add_axes([.92, .1, .02, .8])
        fig.colorbar(hm, cax=cax)
        ax.set_xlabel('Debt', fontsize=16)
        ax.set_ylabel('Income', fontsize=16)
        ax.set_title("Default Probabilities, Disas. Prb=" + str(p_d) + ", beta=" + str(beta) + ", sigma=" + str(sigma), fontsize=18)
        plt.show()
        
        
        
        
        #%% simulation
        
        T = 500
        
        y_sim, b_sim, d_sim, q_sim = simulate(T, b_choice, y_grid, b_grid)
        
        plt.plot(y_sim)
        plt.title("Output TS, p_d" + param_name + '=' + str(param) + ", β=" + str(beta) + ", σ=" + str(sigma))
        plt.xlabel("Time")
        plt.ylabel("Output")
        for t in range(T):
            d_t = d_sim[t]
            if d_t == 1:
                plt.axvline(x=t, color='r', linewidth=2, alpha=0.4)
        plt.show()
        
        
        plt.plot(b_sim)
        plt.title("Asset Position TS," + param_name + '=' + str(param) + ", σ=" + str(beta) + ", σ=" + str(sigma))
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
        #print("Disas. Prb=" + str(p_d) + ", Median Ouput:", np.median(y_means))
        print(param_name + ": " + str(param) + ", Mean Assets:", np.median(b_means))
        
        # calculate the Barro equity premium from parameters
        EP = (sigma * eta**2 + (1-(1-p_d)**4) * ((d_size)**(-sigma) - (d_size)**(1-sigma) + (1-d_size))) * 100
        
        b_mean_vec.append(np.median(b_means))
        q_mean_vec.append(np.median(q_means))
        eq_prem_vec.append(EP)
    
    #eq_prem_mat.append(eq_prem_vec) # new graph for each beta
    b_mat.append(b_mean_vec) # new graph for each beta (beta only affects borrowing, not EP)


    
#%%

# create lists of colors and linestyles
color_list = ['red', 'blue', 'green', 'black']
lstyle_list = ['solid', 'dotted', 'dashed', 'dashdot']

# plot how the mean changes with disaster probability
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax2.plot(sigma_vec, b_mat[0], linewidth=3, color='red')
for (i,beta) in zip(range(len(beta_vec)), beta_vec):
    ax1.plot(param_vec, b_mat[i], linewidth=3, color=color_list[i], linestyle=lstyle_list[i], label='β=' + str(beta_vec[i]))
fig.text(0.5, -0.05, "*Probability of staying in a disaster is 0.2 quarterly, or 0.0016 annually.", ha='center')
#plt.axhline(y = 0, color = 'grey', linestyle = 'dashed') 
ax1.set_title("Gov. Balance Sheet vs. Parameter Values", pad=10)
ax1.set_xlabel(param_name + ", Quarterly", labelpad=7.5)
ax2.set_xlabel("CRRA", labelpad=7.5)
ax1.legend()
ax1.set_ylabel("Mean Assets (% of GDP)")
plt.show()
fig.savefig('param_vs_savings.pdf', format='pdf', bbox_inches='tight')



fig = plt.figure()
ax1 = fig.add_subplot(111)
for (i,beta) in zip(range(len(beta_vec)), beta_vec):
    ax1.plot(eq_prem_vec, b_mat[i], linewidth=3, color=color_list[i], linestyle=lstyle_list[i], label='β=' + str(beta_vec[i]))
fig.text(0.5, -0.05, "*Probability of staying in a disaster is 0.2 quarterly, or 0.0016 annually.", ha='center')
plt.title("Gov. Balance Sheet vs. Equity Premium", pad=10)
plt.xlabel("Implied Equity Premium (%)")
plt.ylabel("Mean Assets (% of GDP)")
plt.legend()
plt.show()
fig.savefig('equityprem_vs_savings.pdf', format='pdf', bbox_inches='tight')
        
        
plt.plot(param_vec, q_mean_vec)
plt.title("Bond Price vs. Parameter Value")
plt.xlabel(param_name)
plt.ylabel("Price")
plt.show()
            
    
    
    
    
    
    
    