# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:12:30 2020

@author: Justin Yu, M.S. Financial Engineering, Stevens Institute of Technology

Monte Carlo simulations for European option pricing
Pricing algorithms include:
    - Simple
    - Antithetic Variate only
    - Delta-based Control Variate only
    - Delta- and Gamma-based Control Variates
    - Delta-based Control Variates and Antithetic Variates
"""

import numpy as np
import scipy.stats as si

def monte_carlo(S,K,T,r,q,sigma,CallPut,n,m):
    '''
    Simple Monte Carlo option pricing algorithm for European Calls and Puts with Euler discretization
    
    Args:
        S - Initial Stock Price         sigma - Volatility
        K - Strike Price                CallPut - Option type specification (Call or Put)
        T - Time to Maturity            n - number of time steps for each MC path
        r - Risk-free interest rate     m - number of simulated paths
        q - Dividend rate
        
    Returns the simulated price of a European call or put with standard deviation and standard error
    associated with the simulation
    '''
    
    def path_sim(mu,sigma,T,S,N, M):
        '''Simulates M paths of Geometric Brownian Motion with N time steps'''
        sims = np.zeros(M) 
        dt = T/N

        for i in range(M):
            W = [0]+np.random.standard_normal(size=N)
            sims[i] = np.sum(W)*np.sqrt(dt) #We only are concerned with the terminal value for European options
        
        St = S*np.exp((mu-0.5*sigma**2)*T + sigma*sims)     
        return St
   
    paths = path_sim(r-q,sigma,T,S,n,m)
    
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
    def path_sim(mu,sigma,T,S,N, M):
        '''Simulates M paths of Geometric Brownian Motion with N time steps incorporating antithetic variance reduction'''
        sims = np.zeros(M)
        dt = T/N

        for i in range(M):
            W = [0]+np.random.standard_normal(size=N)
            sims[i] = np.sum(W)*np.sqrt(dt) #we are concerned with only the final value of the path
        
        St = S*np.exp((mu-0.5*sigma**2)*T + sigma*sims)
        Sta = S*np.exp((mu-0.5*sigma**2)*T - sigma*sims)    
        return np.array([St, Sta])
        
    paths = path_sim(r-q,sigma,T,S,n,m)
    
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


def monte_carlo_deltCV(S,K,T,r,q,sigma,CallPut,n,m):
    '''
    Monte Carlo European option pricing algorithm with Delta-based Control Variate with Euler discretization
    
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
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        if c_p == 'Call':    
            return si.norm.cdf(d1, 0.0, 1.0)
        elif c_p == 'Put':
            return si.norm.cdf(d1, 0.0, 1.0) - 1
        
    def path_sim(mu,sigma,T,S,N, M, cp,K):
        '''Simulates M paths of Geometric Brownian Motion with N time steps incorporating antithetic variance reduction'''
        sims = list(np.zeros(M));
        t = np.linspace(0,T,N);        dt = T/N
        ttm = np.flip(t) 
        erddt = np.exp((r-q)*dt)      
        for i in range(M):
            # GBM paths:
            W = [0]+np.random.standard_normal(size=N)
            W = np.cumsum(W)*np.sqrt(dt)
            St = S*np.exp((mu-0.5*sigma**2)*t + sigma*W)
            
            # Delta CV:
            d = delta(St[0:N-1],K,ttm[0:N-1],mu,sigma,cp)                
            cv = np.sum(d*(St[1:N] - (St[0:N-1]*erddt))*np.exp(mu*ttm[1:N]))
            sims[i] = [St[-1], cv]
        return np.array(sims).T

    paths = path_sim(r-q,sigma,T,S,n,m,CallPut,K)

    if CallPut == 'Call':
        # Option payoff:
        paths[0] = paths[0] - K
        paths[0][paths[0] < 0] = 0
        
        # Control Variate:
        paths = paths[0] - paths[1]
        
    elif CallPut == 'Put':
        # Option payoff:
        paths[0] = K - paths[0]
        paths[0][paths[0] < 0] = 0
        
        # Control Variate:
        paths = paths[0] - paths[1]
    
    c_hat = np.mean(paths)*np.exp(-r*T)    
    sd = np.sqrt(np.sum((paths*np.exp(-r*T) - c_hat)**2)/(m-1))
    se = sd/np.sqrt(m)
      
    return c_hat, sd, se


def monte_carlo_delt_gamCV(S,K,T,r,q,sigma,CallPut,n,m):
    '''
    Monte Carlo European option pricing algorithm with Delta-based and Gamma-based Control Variates with Euler discretization
    
    Args:
        S - Initial Stock Price         sigma - Volatility
        K - Strike Price                CallPut - Option type specification (Call or Put)
        T - Time to Maturity            n - number of time steps for each MC path
        r - Risk-free interest rate     m - number of simulated paths
        q - Dividend rate
    
    Returns the simulated price of a European call or put with standard deviation and standard error
    associated with the simulation
    '''
    def delta(S, K, T, r, sigma, CallPut):
        '''Black-Scholes delta'''
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        if CallPut == 'Call':    
            return si.norm.cdf(d1, 0.0, 1.0)
        elif CallPut == 'Put':
            return si.norm.cdf(d1, 0.0, 1.0) - 1
    
    def gamma(S, K, T, r, q, sigma):
        '''Black-Scholes gamma'''
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        return (np.exp(-q*T)/(S*sigma*np.sqrt(T))) * ((2*np.pi)**(-0.5)) * np.exp(-0.5*d1**2)
    
    def path_sim(r,q,sigma,T,S,N, M,CallPut,K):
        '''Simulates paths of Geometric Brownian motion while implementing the control variates'''
        sims = list(np.zeros(M));
        t = np.linspace(0,T,N);        dt = T/N
        ttm = np.flip(t) 
        erddt = np.exp((r-q)*dt)   
        egamma = np.exp((2*(r-q) + sigma**2)*dt) - 2*erddt + 1

        for i in range(M):
            # GBM path:
            W = [0]+np.random.standard_normal(size=N)
            W = np.cumsum(W)*np.sqrt(dt)
            St = S*np.exp((r-q-0.5*sigma**2)*t + sigma*W)
            
            # Delta CV:
            d = delta(St[0:N-1],K,ttm[0:N-1],r-q,sigma,CallPut)                
            cv_d = np.sum(d*(St[1:N] - (St[0:N-1]*erddt))*np.exp((r-q)*ttm[1:N]))
            
            # Gamma CV:
            g = gamma(St[0:N-1], K, ttm[0:N-1], r,q, sigma)     
            cv_g = np.sum(g*(((St[1:N] - St[0:N-1])**2) - egamma*St[0:N-1]**2)*np.exp((r-q)*ttm[1:N]))
            sims[i] = [St[-1], cv_d, cv_g]
        return np.array(sims).T
        
    paths = path_sim(r,q,sigma,T,S,n,m,CallPut,K)

    if CallPut == 'Call':
        # Option payoff:
        paths[0] = paths[0] - K
        paths[0][paths[0] < 0] = 0
        
        # Control Variates:
        paths = paths[0] - paths[1]- 0.5*paths[2]
        
    elif CallPut == 'Put':
        # Option payoff:
        paths[0] = K - paths[0]
        paths[0][paths[0] < 0] = 0
        
        # Control Variates:
        paths = paths[0] - paths[1] - 0.5*paths[2]
    
    c_hat = np.mean(paths)*np.exp(-r*T)    
    sd = np.sqrt(np.sum((paths*np.exp(-r*T) - c_hat)**2)/(m-1))
    se = sd/np.sqrt(m)
      
    return c_hat, sd, se



def monte_carlo_deltCV_AV(S,K,T,r,q,sigma,CallPut,n,m):
    '''
    Monte Carlo European option pricing algorithm with Delta-based Control Variate and Antithetic Variate
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
        d1 = (np.log(S/K) + T*(r + 0.5*sigma**2))/(sigma*np.sqrt(T))
        if c_p == 'Call':    
            return si.norm.cdf(d1, 0.0, 1.0)
        elif c_p == 'Put':
            return si.norm.cdf(d1, 0.0, 1.0) - 1
        
    def path_sim(mu,sigma,T,S,N, M, cp,K):
        '''Simulates M paths of Geometric Brownian Motion with N time steps incorporating antithetic variance reduction'''
        sims = list(np.zeros(M));
        t = np.linspace(0,T,N);        dt = T/N
        ttm = np.flip(t)  
        erddt = np.exp((r-q)*dt)    

        for i in range(M):         
            # GBM paths:
            W = [0]+np.random.standard_normal(size=N)
            W = np.cumsum(W)*np.sqrt(dt)
            St = S*np.exp((mu-0.5*sigma**2)*t + sigma*W);            Sta = S*np.exp((mu-0.5*sigma**2)*t - sigma*W)
            
            # Delta CV:
            d = delta(St[0:N-1],K,ttm[0:N-1],mu,sigma,cp);            da = delta(Sta[0:N-1],K,ttm[0:N-1],mu,sigma,cp)
            cv = np.sum(d*(St[1:N] - (St[0:N-1]*erddt))*np.exp(mu*ttm[1:N]))
            cva = np.sum(da*(Sta[1:N] - (Sta[0:N-1]*erddt))*np.exp(mu*ttm[1:N]))
            sims[i] = [St[-1], Sta[-1], cv, cva]
        
        return np.array(sims).T
    
    paths = path_sim(r-q,sigma,T,S,n,m,CallPut,K)

    if CallPut == 'Call':
        # Option payoff:
        paths[0] = paths[0] - K; paths[1] = paths[1] - K
        paths[0][paths[0] < 0] = 0; paths[1][paths[1] < 0] = 0
        
        # Control Variate:
        paths[0] = paths[0] - paths[2]
        paths[1] = paths[1] - paths[3]
        
    elif CallPut == 'Put':
        # Option payoff:
        paths[0] = K - paths[0]; paths[1] = K - paths[1]
        paths[0][paths[0] < 0] = 0; paths[1][paths[1] < 0] = 0
        
        # Control Variate:
        paths[0] = paths[0] + beta1*paths[2]
        paths[1] = paths[1] + beta1*paths[3]
    
    cT = np.mean(paths[0:2], axis=0)
    c_hat = np.mean(paths[0:2])*np.exp(-r*T)
    
    sd = np.sqrt(np.sum((cT*np.exp(-r*T) - c_hat)**2)/(m-1))
    se = sd/np.sqrt(m)
       
    return c_hat, sd, se



def monte_carlo_delt_gamCV_AV(S,K,T,r,q,sigma,CallPut,n,m):
    '''
    Monte Carlo European option pricing algorithm with Delta and Gamma-based Control Variate and Antithetic Variate.
    
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
        d1 = (np.log(S/K) + T*(r + 0.5*sigma*sigma))/(sigma*np.sqrt(T))
        if c_p == 'Call':    
            return si.norm.cdf(d1, 0.0, 1.0)
        elif c_p == 'Put':
            return si.norm.cdf(d1, 0.0, 1.0) - 1
        
    def gamma(S, K, T, r, q, sigma):
        '''Black-Scholes gamma'''
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        return (np.exp(-q*T)/(S*sigma*np.sqrt(T))) * ((2*np.pi)**(-0.5)) * np.exp(-0.5*d1**2)
    
        
    def path_sim(mu,sigma,T,S,N, M, cp,K):
        sims = list(np.zeros(M));
        t = np.linspace(0,T,N);        dt = T/N
        ttm = np.flip(t)  
        erddt = np.exp((r-q)*dt)    
        egamma = np.exp((2*(r-q) + sigma**2)*dt) - 2*erddt + 1

        for i in range(M):         
            # GBM paths
            W = [0]+np.random.standard_normal(size=N)
            W = np.cumsum(W)*np.sqrt(dt)
            St = S*np.exp((mu-0.5*sigma**2)*t + sigma*W);            Sta = S*np.exp((mu-0.5*sigma**2)*t - sigma*W)
            
            # Delta-based control variate
            d = delta(St[0:N-1],K,ttm[0:N-1],mu,sigma,cp);            da = delta(Sta[0:N-1],K,ttm[0:N-1],mu,sigma,cp)
            cv_d = np.sum(d*(St[1:N] - (St[0:N-1]*erddt))*np.exp(mu*ttm[1:N]))
            cva_d = np.sum(da*(Sta[1:N] - (Sta[0:N-1]*erddt))*np.exp(mu*ttm[1:N]))
            
            # Gamma-based control variate
            g = gamma(St[0:N-1], K, ttm[0:N-1], r,q, sigma);        ga = gamma(Sta[0:N-1],K,ttm[0:N-1],r,q,sigma)
            cv_g = np.sum(g*(((St[1:N] - St[0:N-1])**2) - egamma*St[0:N-1]**2)*np.exp((r-q)*ttm[1:N]))           
            cva_g = np.sum(ga*(((Sta[1:N] - Sta[0:N-1])**2) - egamma*Sta[0:N-1]**2)*np.exp((r-q)*ttm[1:N]))
            
            sims[i] = [St[-1], Sta[-1], cv_d, cva_d, cv_g, cva_g]
        return np.array(sims).T

    paths = path_sim(r-q,sigma,T,S,n,m,CallPut,K)

    if CallPut == 'Call':
        # Option payoff
        paths[0] = paths[0] - K;            paths[1] = paths[1] - K
        paths[0][paths[0] < 0] = 0;         paths[1][paths[1] < 0] = 0
        
        # Control Variates:
        paths[0] = paths[0] - paths[2] - 0.5*paths[4]
        paths[1] = paths[1] - paths[3] - 0.5*paths[5]
        
    elif CallPut == 'Put':
        # Option payoff
        paths[0] = K - paths[0];            paths[1] = K - paths[1]
        paths[0][paths[0] < 0] = 0;         paths[1][paths[1] < 0] = 0
        
        # Control Variates:
        paths[0] = paths[0] - paths[2] - 0.5*paths[4]
        paths[1] = paths[1] - paths[3] - 0.5*paths[5]
    
    cT = np.mean(paths[0:2], axis=0)
    c_hat = np.mean(paths[0:2])*np.exp(-r*T)

    sd = np.sqrt(np.sum((cT*np.exp(-r*T) - c_hat)**2)/(m-1))
    se = sd/np.sqrt(m)
       
    return c_hat, sd, se
