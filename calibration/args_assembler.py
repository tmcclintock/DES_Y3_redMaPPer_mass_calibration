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
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import cluster_toolkit as ct
from classy import Class

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

    #For each modeling scenario add the extra information
    if "has_RM_selection" in model_choices:
        args["has_RM_selection"] = True
        args["powers"] = np.array([2, 3, 4])
        R = args["R"]
        args["lower_mask"] = np.where(R < 0.2) #200 kpc/h com. cut
        args["upper_mask"] = np.where(R > 30.) #30 Mpc/h com. cut
        args["lowest_index"] = np.argmax(R > 0.2)
        args["x"] = np.log(R/30.)
        args["X"] = np.ones((len(x), len(powers)))

        A_mean = np.load("./prior_information/A_means.npy")[z_index, lambda_index]
        A_cov = np.load("./prior_information/A_covs.npy")[z_index, lambda_index]
        args["A_matrix_prior_means"] = A_mean
        args["A_matrix_prior_cov_inv"] = np.linalg.inv(A_cov)

    if "has_multiplicative_bias" in model_choices:
        args["has_multiplicative_bias"] = True
        print("Can't do multiplicative bias yet. No priors!")

    if "has_miscentering" in model_choices:
        args["has_miscentering"] = True
        print("Can't do miscentering yet. No priors!")

    if "has_boost_factors" in model_choics:
        args["has_boost_factors"] = True
        print("Can't do boost factors yet. No priors or data!")
        
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
    Omega_m = 0.286 #check this
    Sigma_crit_inverse = 1. #For now

    #Radial bin edges in Mpc physical
    #Radial arrays
    r = np.logspace(-2, 3, num=1000) #Mpc/h comoving; 3d radii
    R = np.logspace(-2, 2.4, 1000, base=10) #Mpc/h comoving; 2d projected radii
    #k = np.logspace(-5, 3, num=4000) #1/Mpc comoving; wavenumbers
    k = np.logspace(-3, 2, num=2000) #1/Mpc comoving; wavenumbers
    M = np.logspace(12, 17, 1000) #Msun/h; halo masses
    
    args["R_edges"] = np.logspace(np.log10(0.0323), np.log10(30.), num=15+1) * h * (1+z)


    #Precompute theory quantities
    class_params = { # 'N_eff': 3.04
        'output': 'mPk',
        "h":h,
        "sigma8":0.82,
        "n_s":0.96,
        'Omega_b': 0.047,
        "Omega_cdm":Omega_m - 0.47,
        'P_k_max_1/Mpc':100.,
        'z_max_pk':1.0,
        'non linear':'halofit'}
    class_cosmo = Class()
    class_cosmo.set(class_params)
    print("Running CLASS for P(k)")
    class_cosmo.compute()
    P_nl = np.array([class_cosmo.pk(ki, z) for ki in k])*h**3
    P_lin = np.array([class_cosmo.pk_lin(ki, z) for ki in k])*h**3
    print("CLASS computation complete")
    xi_lin = ct.xi.xi_mm_at_r(r, k/h, P_lin)
    xi_nl  = ct.xi.xi_mm_at_r(r, k/h, P_nl)
    biases = ct.bias.bias_at_M(M, k, P_lin, Omega_m)
    bias_spline = IUS(np.log(M), biases)
    

    #PERFORM SCALE CUTS HERE

    #Assemble the args dict
    args = {"z":z, "richness":richness, "R_lambda":R_lambda, "h":h,
            "Omega_m":Omega_m, "Sigma_crit_inverse":Sigma_crit_inverse,
            "DeltaSigma_data": dat.DeltaSigma, "DeltaSigma_cov":Cov,
            "r":r, "R":R, "k":k/h, "P_lin":P_lin, "P_nl":P_nl,
            "xi_lin":xi_lin, "xi_nl":xi_nl, "bias_spline":bias_spline}
    return args
