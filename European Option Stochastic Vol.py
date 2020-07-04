# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 19:26:29 2020

@author: Justin Yu, M.S. Financial Engineering, Stevens Institute of Technology

Monte Carlo Option pricing for European options with stochastic volatility
Simulations use standard Euler discretization.
Price of the underlying asset follows Geometric Brownian Motion
"""
import numpy as np

def heston_MC(S, K, T, r, V, q, rho, kappa, theta, sigma, CallPut, n, m):
    '''
    Monte Carlo option pricing for European options with stochastic volatility following the Heston 
    model
    
    Args:
        S - initial price of underlying asset                   kappa - mean reversion              
        K - strike price                                        theta - long-run variance
        T - time to maturity                                    sigma - vol of vol
        r - risk-free rate                                      CallPut - call or put
        V - initial volatility of underlying asset              n - number of time steps in each path
        q - dividend rate                                       m - number of paths for the simulation
        rho - Brownian motions of price and volatility are
              correlated by this value
    '''
    sims = list(np.zeros(m))
    dt = T/n    
    
    for i in range(m):        
        W1 = np.random.standard_normal(size = n)*np.sqrt(dt)
        W2 = np.random.standard_normal(size = n)*np.sqrt(dt)     
        Z = rho*W1 + np.sqrt(1-rho**2)*W2     
        St = S; Vt = V
        
        for j in range(1,n):
            St = St + (r*St*dt + np.sqrt(V)*St*W1[j])
            Vt = Vt + kappa*(theta - Vt)*dt + sigma*np.sqrt(Vt)*Z[j]     
        sims[i] = St
    
    if CallPut == 'Call':      
        payoff = np.array(sims) - K
        payoff[payoff < 0] = 0       
    elif CallPut == 'Put':
        payoff = np.array(sims)*(-1) + K
        payoff[payoff < 0] = 0    
    
    return np.mean(payoff)*np.exp(-r*T)

