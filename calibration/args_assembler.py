"""This file fascilitates the assembly of the `args` dictionary. This dictionary
contains things like the data vector(s), the covariance, any precomputed
parts of the model like the matter power spectrum, and independent variables
like radial distances.

Cuts to the data vectors are performed here, as are linear modifications
to the covaraince such as the Hartlap correction.
"""

import numpy as np
import pandas as pd
import scipy as sp
import cluster_toolkit as ct

def get_args_dictionary(analysis_name, z_index, lambda_index, model_choices):
    """Obtain a dictionary called `args` that contains relevant
    pre-computed quantities depending on the analysis being performed.
    """

    if "Buzzard" in analysis_name:
        if "redMaPPer" in analysis_name:
            args = get_Buzzard_args(z_index, lambda_index, RM=True)
        elif "samemass" in analysis_name:
            args = get_Buzzard_args(z_index, lambda_index, RM=False)
        else:
            raise Exception("Incorrect Buzzard analysis option.")

    #Add the prior information on the args as appropriate

    #Return the args dictionary
    return args

def get_Buzzard_args(z_index, lambda_index, RM):
    assert z_index in [0,1,2]
    assert lambda_index in [0,1,2,3]

    zstr = ["0.2_0.35", "0.35_0.5", "0.5_0.65"][z_index]
    lstr = ["20_30", "30_45", "45_60", "60_10000"][lambda_index]
    
    base = "../data/DeltaSigma_from_Buzzard/"
    if RM:
        inpath = base+"DeltaSigma_z_{0}_lam_{1}.dat".format(zstr, lstr)
    else:
        inpath = base+"DeltaSigma_same_mass_redshift_distribution_z_{0}_lam_{1}.dat".format(zstr, lstr)

    columns = ["R", "DeltaSigma", "DeltaSigma_err"]
    dat = pd.read_csv(inpath, sep = ' ', skiprows=0, usecols=[0,1,2])
    dat.columns = columns
    Cov = np.diagonal(dat["DeltaSigma_err"]**2)


    index = 4*z_index + lamda_index
    z = np.loadtxt("Buzzard_redMaPPer_redshift_information.dat")[4, index]
    richness = np.array([25, 37.5, 52.5, 70])[lambda_index]
    R_lambda = (richness/100.)**0.2 #Mpc/h comoving

    #Define the cosmology
    h = 0.7 #Hubble constant
    Omega_m = 0.3 #check this
    Sigma_crit_inverse = 1. #For now

    #Compute power spectra
    #Compute correlation functions and radii arrays
    #Create halo bias spline in ln(M)

    #PERFORM SCALE CUTS HERE

    #Assemble the args dict
    args = {"z":z, "richness":richness, "R_lambda":R_lambda, "h":h,
            "Omega_m":Omega_m, "Sigma_crit_inverse":Sigma_crit_inverse,
            "DeltaSigma_data": dat.DeltaSigma, "DeltaSigma_cov":Cov}
    return args
