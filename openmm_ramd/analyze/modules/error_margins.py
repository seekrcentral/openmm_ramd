"""
Code for handling error margins in openmm_ramd analysis.
"""

import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
from scipy.stats import expon

import openmm_ramd.analyze.modules.samples as samples

def Q_to_K(Q):
    """Given a rate matrix Q, compute probability transition matrix K."""
    K = np.zeros(Q.shape, dtype=np.longdouble)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[0]):
            if i == j:
                K[i,j] = 0.0
            else:
                if Q[j,j] == 0.0:
                    K[i,j] = 0.0
                else:
                    K[i,j] = -Q[i,j] / Q[j,j]
    
    return K

def Q_to_t(Q):
    """Given a rate matrix Q, compute avg incubation time vector t."""
    t = np.zeros((Q.shape[0], 1))
    for i in range(Q.shape[0]):
        if Q[i,i] == 0.0:
            t[i, 0] = 0.0
        else:
            t[i, 0] = -1.0 / Q[i,i]
            
    return t

def irreversible_stochastic_matrix_algorithm_sample(
        Q, N, R, nonzero_indices=None):
    """
    Copied, more or less, from seekr2
    
    Sample an additional matrix Q from the distribution using a Monte-Carlo
    method. Use a modified version of Algorithm 1 from Noe 2008.
    """
    MAX_ITER = 1e7
    if (not Q.any()) or (not N.any()) or (not R.any()):
        return Q
    # step 0 - Generate uniform random variables: i,j members of {1,...,m}
    m = Q.shape[0]
    if nonzero_indices == None:
        i = random.choice(range(m))
        j = random.choice(range(m))
        counter = 0
        while (Q[i,j] == 0.0) or (Q[i,i] == 0.0) or (N[i,j] <= 1) \
                or (R[i,0] == 0) or (i==j):
            i = random.choice(range(m))
            j = random.choice(range(m))
            counter += 1
            if counter > MAX_ITER:
                raise Exception("Maximum iterations exceeded.")
    else:
        i, j = random.choice(nonzero_indices)
        assert i!=j, "Cannot choose same indices i and j"
    
    # step 1 - Initialize the matrix (already done)
    Qnew = deepcopy(Q)
    # step 2.1 - delta, r between 0 and 1
    # delta must be sampled such that when it's subtracted from q_ij, the 
    #  result doesn't go below zero
    delta = Q[i,j] * (1.0 - expon.rvs())
    random_uniform = random.random()  
    # step 2.2 - copy Q to a new matrix - already done
    
    # step 2.3 - Nonreversible element shift
    # prior probability: P(Q)
    log_prior_probability_old = np.abs(-Qnew[i,j])
    log_prior_probability_new = np.abs(-Qnew[i,j] + delta)
    if (log_prior_probability_old == 0.0) or (log_prior_probability_new == 0.0):
        p_acc = 0.0
    else:
        N_eff = Qnew[i,j] * R[i,0]
        #log_p_Q_old = N[i,j] * np.log(np.abs(Qnew[i,j]))  - Qnew[i,j] * R[i,0]
        #log_p_Q_new = N[i,j] * np.log(np.abs(Qnew[i,j] - delta)) \
        #    - (Qnew[i,j] - delta) * R[i,0]
        log_p_Q_old = N_eff * np.log(np.abs(Qnew[i,j]))  - Qnew[i,j] * R[i,0]
        log_p_Q_new = N_eff * np.log(np.abs(Qnew[i,j] - delta)) \
            - (Qnew[i,j] - delta) * R[i,0]
        p_acc = log_p_Q_new - log_p_Q_old + log_prior_probability_new \
            - log_prior_probability_old
        
    if np.log(random_uniform) <= p_acc: 
        #log(r) can be directly compared to 
        # log-likelihood acceptance, p_acc
        Qnew[i,i] = Qnew[i,i] + delta
        Qnew[i,j] = Qnew[i,j] - delta
        
    return Qnew

def monte_carlo_milestoning_error(
        main_data_sample, num=1000, skip=None, stride=None, verbose=False):
    """
    More or less copied from seekr2's Elber error code.
    
    Calculates an error estimate by sampling a distribution of rate 
    matrices assumming a Poisson (gamma) distribution with 
    parameters Nij and Ri using Markov chain Monte Carlo.
        
    Enforces detailed Balance-- using a modified version of 
    Algorithm 4 from Noe 2008 for rate matrices.-- 
    Citation: Noe, F. "Probability Distributions of molecular observables
    computed from Markov models." J. Chem. Phys. 2008, 128, No. 244103.
    Distribution is:  p(Q|N) = p(Q)p(N|Q)/p(N) = 
        p(Q) PI(q_ij**N_ij * exp(-q_ij * Ri))
        
    Parameters
    ----------
    
    main_data_sample : MMVT_data_sample()
        The data sample contains all data extracted from simulations as
        well as model information.
    
    num : int, default 1000
        number of rate matrix (Q) samples to be generated
        
    skip : int, default None
        number of inital rate matrix samples to skip for "burn in". If
        None, it will be assigned 10 * N**2, where N is the size of the
        rate matrix Q
        
    stride : int, default None
        frequency at which rate matrix samples are recorded- larger
        frequency reduces correlation between samples. If None, it will
        be assigned N**2, where N is the size of the rate matrix Q.
        
    verbose : bool, default False
        allow additional verbosity/printing
    """
    MAX_SKIP = 1000
    MAX_STRIDE = 100
    Q_ij = main_data_sample.Q
    N_ij = main_data_sample.N_ij
    R_i = main_data_sample.R_i
    Q = deepcopy(Q_ij)
    m = Q.shape[0] #get size of count matrix
    N = np.zeros((m, m))
    R = np.zeros((m, 1))
    if skip is None:
        skip = 40 * Q.shape[0]
        if skip > MAX_SKIP:
            skip = MAX_SKIP
    if stride is None:
        stride = 4 * Q.shape[0]
        if stride > MAX_STRIDE:
            stride = MAX_STRIDE
        
    for i in range(m):
        for j in range(m):
            N[i,j] = N_ij[i,j]
        
        R[i,0] = R_i[i]
        
    data_sample_list = []
    if verbose: print("collecting", num, "MCMC samples from", 
                      num*(stride) + skip + 1, "total moves")
    new_data_sample = None
    for counter in range(num * stride + skip + 1):
        #if verbose: print("MCMC stepnum: ", counter)
        Qnew = irreversible_stochastic_matrix_algorithm_sample(Q.T, N.T, R)
        if counter > skip and counter % stride == 0:
            new_data_sample = samples.Data_sample()
            new_data_sample.N_ij = N_ij
            new_data_sample.R_i = R_i
            new_data_sample.Q = Qnew.T
            new_data_sample.K = Q_to_K(Qnew.T)
            new_data_sample.t = Q_to_t(Qnew.T)
            data_sample_list.append(new_data_sample)
            
        Q = Qnew.T
        Q[:,-2:] = Q_ij[:,-2:]
        

    assert new_data_sample is not None, "Nothing sampled in Monte Carlo "\
        "milestoning procedure. Please choose different arrangement of num, "\
        "skip, and stride."
    #if verbose: print("final MCMC matrix", Q)
    
    return data_sample_list
