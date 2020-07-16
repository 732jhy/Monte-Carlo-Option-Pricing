# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:12:30 2020

@author: Justin Yu, M.S. Financial Engineering, Stevens Institute of Technology

Monte Carlo simulations for European option pricing (simple, AV, and CV)
"""
import numpy as np
import scip.stats as si

def monte_carlo(S,K,T,r,q,sigma,CallPut,n,m):
    '''
    Monte Carlo option pricing algorithm for European Calls and Puts with Euler discretization
    
    Args:
        S - Initial Stock Price         sigma - Volatility
        K - Strike Price                CallPut - Option type specification (Call or Put)
        T - Time to Maturity            n - number of time steps for each MC path
        r - Risk-free interest rate     m - number of simulated paths
        q - Dividend rate
        
    Returns the simulated price of a European call or put with standard deviation and standard error
    associated with the simulation
    '''
    
    def GBM_sim(mu,sigma,T,S,N, M):
        '''Simulates M paths of Geometric Brownian Motion with N time steps'''
        sims = np.zeros(M) 
        dt = T/N
        t = np.linspace(0, T, N)
        for i in range(M):
            W = [0]+np.random.standard_normal(size=N)
            sims[i] = np.sum(W)*np.sqrt(dt) #We only are concerned with the terminal value for European options
        
        St = S*np.exp((mu-0.5*sigma**2)*t + sigma*sims)     
        return St
   
    paths = GBM_sim(r-q,sigma,T,S,n,m)
    
    if CallPut == 'Call':
        paths = paths - K
        paths[paths < 0] = 0    
    
    elif CallPut == 'Put':
        paths = K - paths
        paths[paths < 0] = 0
    
    c_hat = np.mean(paths)*np.exp(-r*T)
    
    sd = np.sqrt(np.sum((paths*np.exp(-r*T) - c_hat)**2)/(m-1))
    se = sd/np.sqrt(m)       
    return c_hat, sd, se


def monte_carlo_AV(S,K,T,r,q,sigma,CallPut,n,m):
    '''
    Monte Carlo option pricing algorithm for European Calls and Puts with Antithetic Variance Reduction
    with Euler discretization
    
    Args:
        S - Initial Stock Price         sigma - Volatility
        K - Strike Price                CallPut - Option type specification (Call or Put)
        T - Time to Maturity            n - number of time steps for each MC path
        r - Risk-free interest rate     m - number of simulated paths
        q - Dividend rate
        
    Returns the simulated price of a European call or put with standard deviation and standard error
    associated with the simulation
    '''
    def GBM_sim(mu,sigma,T,S,N, M):
        '''Simulates M paths of Geometric Brownian Motion with N time steps incorporating antithetic variance reduction'''
        sims = np.zeros(M)
        dt = T/N
        t = np.linspace(0, T, N)
        for i in range(M):
            W = [0]+np.random.standard_normal(size=N)
            sims[i] = np.sum(W)*np.sqrt(dt) #we are concerned with only the final value of the path
        
        St = S*np.exp((mu-0.5*sigma**2)*t + sigma*sims)
        Sta = S*np.exp((mu-0.5*sigma**2)*t - sigma*sims)    
        return np.array([St, Sta])
        
    paths = GBM_sim(r-q,sigma,T,S,n,m)
    
    if CallPut == 'Call':
        paths = paths - K
        paths[paths < 0] = 0
    
    elif CallPut == 'Put':
        paths = K - paths
        paths[paths < 0] = 0
    
    cT = np.mean(paths, axis=0)
    c_hat = np.mean(paths)*np.exp(-r*T)
    
    sd = np.sqrt(np.sum((cT*np.exp(-r*T) - c_hat)**2)/(m-1))
    se = sd/np.sqrt(m)
    return c_hat, sd, se


def monte_carlo_CV(S,K,T,r,q,sigma,CallPut,n,m,beta1):
    '''
    Monte Carlo European option pricing algorithm with Delta-based control variate with Euler discretization
    
    Args:
        S - Initial Stock Price         sigma - Volatility
        K - Strike Price                CallPut - Option type specification (Call or Put)
        T - Time to Maturity            n - number of time steps for each MC path
        r - Risk-free interest rate     m - number of simulated paths
        q - Dividend rate
    
    Returns the simulated price of a European call or put with standard deviation and standard error
    associated with the simulation
    '''
    def delta(S, K, T, r, sigma, c_p):
        '''Black-Scholes delta'''
        d1 = (np.log(S/K) + np.sqrt(T)*(r + 0.5*sigma*sigma))/(sigma*np.sqrt(T))
        if c_p == 'Call':    
            return si.norm.cdf(d1, 0.0, 1.0)
        elif c_p == 'Put':
            return si.norm.cdf(d1, 0.0, 1.0) - 1
        
    def GBM_sim(mu,sigma,T,S,N, M, cp,K):
        '''Simulates M paths of Geometric Brownian Motion with N time steps incorporating antithetic variance reduction'''
        sims = list(np.zeros(M));
        t = np.linspace(0,T,N);        dt = T/N
        ttm = np.flip(t) 
        erddt = np.exp((r-q)*dt)      
        for i in range(M):
            W = [0]+np.random.standard_normal(size=N)
            W = np.cumsum(W)*np.sqrt(dt)
            St = S*np.exp((mu-0.5*sigma**2)*t + sigma*W)
            d = delta(St[0:N-1],K,ttm[0:N-1],mu,sigma,cp)                
            cv = np.sum(d*(St[1:N] - (St[0:N-1]*erddt))*np.exp(mu*ttm[1:N]))
            sims[i] = [St[-1], cv]
        return np.array(sims).T
    beta1 = -1.0 
    paths = GBM_sim(r-q,sigma,T,S,n,m,CallPut,K)

    if CallPut == 'Call':
        paths[0] = paths[0] - K
        paths[0][paths[0] < 0] = 0
        paths = paths[0] + beta1*paths[1]
        
    elif CallPut == 'Put':
        paths[0] = K - paths[0]
        paths[0][paths[0] < 0] = 0
        paths = paths[0] + beta1*paths[1]
    
    c_hat = np.mean(paths)*np.exp(-r*T)    
    sd = np.sqrt(np.sum((paths*np.exp(-r*T) - c_hat)**2)/(m-1))
    se = sd/np.sqrt(m)
      
    return c_hat, sd, se


def monte_carlo_AVCV(S,K,T,r,q,sigma,CallPut,n,m):
    '''
    Monte Carlo European option pricing algorithm with Delta-based control variate and Antithetic Variates
    with Euler discretization.
    
    Args:
        S - Initial Stock Price         sigma - Volatility
        K - Strike Price                CallPut - Option type specification (Call or Put)
        T - Time to Maturity            n - number of time steps for each MC path
        r - Risk-free interest rate     m - number of simulated paths
        q - Dividend rate
    
    Returns the simulated price of a European call or put with standard deviation and standard error
    associated with the simulation
    '''
    def delta(S, K, T, r, sigma, c_p):
        '''Black-Scholes delta'''
        d1 = (np.log(S/K) + np.sqrt(T)*(r + 0.5*sigma*sigma))/(sigma*np.sqrt(T))
        if c_p == 'Call':    
            return si.norm.cdf(d1, 0.0, 1.0)
        elif c_p == 'Put':
            return si.norm.cdf(d1, 0.0, 1.0) - 1
        
    def GBM_sim(mu,sigma,T,S,N, M, cp,K):
        '''Simulates M paths of Geometric Brownian Motion with N time steps incorporating antithetic variance reduction'''
        sims = list(np.zeros(M));
        t = np.linspace(0,T,N);        dt = T/N
        ttm = np.flip(t)  
        erddt = np.exp((r-q)*dt)    

        for i in range(M):         
            W = [0]+np.random.standard_normal(size=N)
            W = np.cumsum(W)*np.sqrt(dt)
            St = S*np.exp((mu-0.5*sigma**2)*t + sigma*W);            Sta = S*np.exp((mu-0.5*sigma**2)*t - sigma*W)
            
            d = delta(St[0:N-1],K,ttm[0:N-1],mu,sigma,cp);            da = delta(Sta[0:N-1],K,ttm[0:N-1],mu,sigma,cp)
            
            cv = np.sum(d*(St[1:N] - (St[0:N-1]*erddt))*np.exp(mu*ttm[1:N]))
            cva = np.sum(da*(Sta[1:N] - (Sta[0:N-1]*erddt))*np.exp(mu*ttm[1:N]))
            sims[i] = [St[-1], Sta[-1], cv, cva]
        
        return np.array(sims).T
    beta1 = -1.0
    paths = GBM_sim(r-q,sigma,T,S,n,m,CallPut,K)

    if CallPut == 'Call':
        paths[0] = paths[0] - K; paths[1] = paths[1] - K
        paths[0][paths[0] < 0] = 0; paths[1][paths[1] < 0] = 0
        paths[0] = paths[0] + beta1*paths[2]
        paths[1] = paths[1] + beta1*paths[3]
        
    elif CallPut == 'Put':
        paths[0] = K - paths[0]; paths[1] = K - paths[1]
        paths[0][paths[0] < 0] = 0; paths[1][paths[1] < 0] = 0
        paths[0] = paths[0] + beta1*paths[2]
        paths[1] = paths[1] + beta1*paths[3]
    
    cT = np.mean(paths[0:1], axis=0)
    c_hat = np.mean(paths[0:1])*np.exp(-r*T)
    
    sd = np.sqrt(np.sum((cT*np.exp(-r*T) - c_hat)**2)/(m-1))
    se = sd/np.sqrt(m)
       
    return c_hat, sd, se


