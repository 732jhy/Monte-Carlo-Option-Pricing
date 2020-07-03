# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:18:19 2020

@author: Justin Yu, M.S. Financial Engineering, Stevens Institute of Technology

Monte Carlo pricing algorithms for Asian-style options
"""
import numpy as np
import scipy.stats as si


def arith_asian(S,K,T,r,sigma,CallPut,M,N):
    '''
    Simple Monte Carlo pricing for arithmetic Asian call option
    
    Args:
        S - Initial Stock Price         sigma - Volatility
        K - Strike Price                CallPut - Option type specification (Call or Put)
        T - Time to Maturity            N - number of time steps for each MC path
        r - Risk-free interest rate     M - number of simulated paths
    
    Returns the simulated price of an arithmetic Asian-style call or put option and the associated 
    95% confidence interval of the estimate.
    '''
    sims = np.zeros(M) # vector of simulated values
    t = np.linspace(0,T,N+1)
    dt = T/N
    
    for i in range(M):      
        W = [0] + list(np.random.standard_normal(size=N))
        W = np.cumsum(W)*np.sqrt(dt)
        St = S*np.exp((r-0.5*sigma**2)*t - sigma*W)
        sims[i] = np.sum(St)
        
    sims = sims/(N+1)
    if CallPut == 'Call':
        sims = sims - K;    sims[sims < 0] = 0
    elif CallPut == 'Put':
        sims = K - sims;    sims[sims < 0] = 0
    
    sims = sims*np.exp(-r*T)
    mu_hat = np.mean(sims)
    sigma_hat = np.sqrt(np.sum((sims-mu_hat)**2)/(M-1))    
    CI = [mu_hat - 1.96*sigma_hat/np.sqrt(M), mu_hat + 1.96*sigma_hat/np.sqrt(M)]
    return mu_hat, CI


def geom_asian(S,K,T,r,sigma,CallPut,M,N):
    '''
    Simple Monte Carlo pricing for geometric Asian call option
    
    Args:
        S - Initial Stock Price         sigma - Volatility
        K - Strike Price                CallPut - Option type specification (Call or Put)
        T - Time to Maturity            N - number of time steps for each MC path
        r - Risk-free interest rate     M - number of simulated paths
    
    Returns the simulated price of an geometric Asian-style call or put option and the associated 
    95% confidence interval of the estimate.
    '''
    sims = np.zeros(M)
    t = np.linspace(0,T,N+1)
    dt = T/N
    
    for i in range(M):
        W = [0] + list(np.random.standard_normal(size=N))
        W = np.cumsum(W)*np.sqrt(dt)
        St = S*np.exp((r-0.5*sigma**2)*t - sigma*W)
        sims[i] = np.product(St)**(1/(N+1))
    
    if CallPut == 'Call':
        sims = sims - K
        sims[sims < 0] = 0
        
    elif CallPut == 'Put':
        sims = K - sims
        sims[sims < 0] = 0
        
    sims = sims*np.exp(-r*T)
    est = np.mean(sims)   
    sigma_hat = np.sqrt(np.sum((sims-est)**2)/(M-1))
    CI = [est - 1.96*sigma_hat/np.sqrt(M), est + 1.96*sigma_hat/np.sqrt(M)]
    
    return est, CI    



def arith_asian_CV(S,K,T,r,sigma,M,N):
    '''
    Monte Carlo for arithmetic Asian call option with control variate
    The control variate is implemented using the Geometric Asian option's closed-form Black-Scholes pricing
    formula
    
    Args:
        S - Initial Stock Price         sigma - Volatility
        K - Strike Price                CallPut - Option type specification (Call or Put)
        T - Time to Maturity            N - number of time steps for each MC path
        r - Risk-free interest rate     M - number of simulated paths
    
    Returns the simulated price of the arithmetic Asian-style call or put option and the associated 95%
    confidence interval of the simulation.
    '''
    def geom_asian_black_scholes(S,K,T,r,sigma,N,CallPut):
        '''
        Black-Scholes implementation for pricing geometric Asian options
        
        Args:
            S - Initial Stock Price         sigma - Volatility
            K - Strike Price                CallPut - Option type specification (Call or Put)
            T - Time to Maturity            n - number of time steps for each MC path
            r - Risk-free interest rate     m - number of simulated paths
            q - Dividend rate
            
        Returns the price of a geometric asian call or put option
        '''
        sigma_hat = sigma*np.sqrt((2*N + 1)/(6*(N+1)))
        ro = 0.5*(r - 0.5*sigma**2 + sigma_hat**2)      
        d1 = (1/(sigma_hat*np.sqrt(T)))*(np.log(S/K) + (ro + 0.5*sigma_hat**2)*T)
        d2 = (1/(sigma_hat*np.sqrt(T)))*(np.log(S/K) + (ro - 0.5*sigma_hat**2)*T)       
        if CallPut == 'Call':
            return np.exp(-r*T)*(S*np.exp(ro*T)*si.norm.cdf(d1) - K*si.norm.cdf(d2))         
        elif CallPut == 'Put':
            return np.exp(-r*T)*(K*si.norm.cdf(-d2)- S*np.exp(ro*T)*si.norm.cdf(-d1) - K*si.norm.cdf(d2))

    X = np.zeros(M) # vector of simulated values
    Y = np.zeros(M)
    t = np.linspace(0,T,N+1)
    dt = T/N
    
    for i in range(M):     
        W = [0] + list(np.random.standard_normal(size=N))
        W = np.cumsum(W)*np.sqrt(dt)
        St = S*np.exp((r-0.5*sigma**2)*t - sigma*W)
        X[i] = np.sum(St)/(N+1)
        Y[i] = np.product(St)**(1/(N+1))
        
    X = X - K;      Y = Y - K
    X[X < 0] = 0;   Y[Y < 0] = 0
    X = X*np.exp(-r*T);     Y = Y*np.exp(-r*T)
    
    X_bar = np.mean(X)     
    Y_bar = np.mean(Y)    
    b_star = np.sum((X - X_bar)*(Y - Y_bar))/np.sum((X - X_bar)**2)    
    Eg = geom_asian_black_scholes(S, K, T, r, sigma, T*252, 'Call') - Y
    Pa = X + b_star*Eg    
    mu_hat = np.mean(Pa)    
    sigma_hat = np.sqrt(np.sum((Pa-mu_hat)**2)/(M-1))    
    CI = [mu_hat - 1.96*sigma_hat/np.sqrt(M), mu_hat + 1.96*sigma_hat/np.sqrt(M)]    
    return mu_hat, CI


def asian_barrier_MC(S,K,B,T,r,q,sigma,n,m):
    '''
    Monte Carlo simulation for Up-and-Out Asian Barrier Call option
    Assumes the closing price is checked at the end of every trading day.
    
    Args:
        S - Price of the underlying asset at time 0     q - dividend rate
        K - Strike price                                sigma - volatility
        B - Barrier value                               n - time steps for each path
        T - time to maturity                            m - number of paths in the simulation
        r - risk-free rate
    '''
    def GBM_sim(mu,sigma,T,S,N, M):
        '''Simulates paths of Geometric Brownian motion, checking or the up-and-out condition'''
        sims = np.zeros(M)
        dt = T/N
        for i in range(M):
            W = [0]+np.random.standard_normal(size=N)
            W = np.cumsum(W)*np.sqrt(dt)
            St = S*np.exp((mu-0.5*sigma**2)*T + sigma*W)
            BARRIER_CROSSED = St  > B
            if True in BARRIER_CROSSED:
                sims[i] = 0
            else:
                sims[i] = np.mean(St[::24]) # mean of just the closing prices           
        return sims
    
    paths = GBM_sim(r-q,sigma,T,S,n,m)
    paths = paths - K
    paths[paths < 0] = 0    
    return np.mean(paths)*np.exp(-r*T)





