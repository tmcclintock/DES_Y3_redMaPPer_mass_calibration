"""This file contains the (log)probability density functions (PDFs, or 
probability distribution functions) for the lensing profile(s) and other
associated quantities (like boost factors). It also has priors.
"""

import numpy as np
import lensing_models

def lnprior(parameter_dict, args):

    #Variable to hold the log-prior
    ln_pr = 0
    
    log10_M = parameter_dict["log10_M"] # log_10; Msun/h
    c = parameter_dict["c"] #concentration 200m

    #Trivial, hard priors
    if log10_M < 12 or log10_M > 16:
        return -np.inf
    if c < 0.1 or c > 20:
        return -np.inf

    if "has_multiplicative_bias" in args:
        Am = parameter_dict["Am"]
        
    if "has_RM_selection" in args:
        A = parameter_dict["A_matrix"]
        A_prior_means = args["A_matrix_prior_means"]
        A_prior_cov_inv = args["A_matrix_prior_cov_inv"]
        #Gaussian prior on these parameters
        ln_pr -= 0.5 * \
            (A - A_prior_means).T @ A_prior_cov_inv @ (A - A_prior_means)

    if "has_miscentering" in args:
        f_mis = parameter_dict["f_mis"]
        tau = parameter_dict["tau_mis"]

    if "has_boost_factors" in args:
        B_0 = parameter_dict["B_0"]
        R_scale_boost = parameter_dict["R_scale_boost"]
        
    return ln_pr

def lnlike(parameter_dict, args):

    #Variable to hold the log-likelihood
    ln_L = 0

    #Pull out the data and covariance
    DeltaSigma_data = args["DeltaSigma_data"]
    Cov = args["DeltaSigma_cov"]
    
    #Get the model
    DeltaSigma_model = lensing_models.get_lensing_profile(parameter_dict, args)

    X = (DeltaSigma_data - DeltaSigma_model)
    ln_L -= 0.5 * X @ np.linalg.solve(Cov, X)
    
    return ln_L

def lnposterior(params, args):
    #Turn the parameter array into a dictionary

    ##########################################
    #This is the part that is tuned by hand for each analysis
    ##########################################
    # Buzzard redMaPPer halo-run analysis below
    log10_M, c, a2, a3, a4 = params
    A = np.array([a2, a3, a4])
    parameter_dict = {"log10_M":log10_M, "M":10**log10_M, "c":c, "A_matrix":A}
    
    # Buzzard halo same-mass analysis below
    #log10_M, c = params
    #parameter_dict = {"log10_M":log10_M, "M":10**log10_M, "c":c}
    
    ##########################################

    ln_pr = lnprior(parameter_dict, args)
    if np.isinf(ln_pr): #Check for an invalid proposal
        return -1e99 #Don't bother with the the likelihood in this case
    
    ln_L = lnlike(parameter_dict, args)
    
    return ln_pr + ln_L
