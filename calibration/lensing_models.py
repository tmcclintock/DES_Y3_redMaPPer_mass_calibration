"""Model for the weak lensing (DeltaSigma) profile, including the selection effects.
"""
import numpy as np
import cluster_toolkit as ct

def get_lensing_profile(parameter_dict, args, return_all_parts=False):
    """A method to obtain the weak lensing profile (DeltaSigma).

    Args:
        parameter_dict (dict): contains the physical parameters of the modle
        args (dict): extra arguments, such as radial distances
        return_all_parts (boolean): flag indicating whether to return
            everything including Sigma(R) and non-binned profiles

    Returns:
        weak lensing profile, unless return_all_parts is True in which case
            it is a dictionary of profiles

    """
    ####
    #(1) Pull out arguments
    ####
    r = args["r"] #Mpc/h comoving; 3d distances
    R = args["R"] #Mpc/h comoving; 2d projected
    k = args["k"] #h/Mpc comoving; 3d wavenumbers
    P_lin = args["P_lin"] #(Mpc/h)^3 comoving; linear matter power spectrum
    P_nl = args["P_nl"] #(Mpc/h)^3 comoving; nonlinear matter power spectrum
    xi_lin = args["xi_lin"] #linear matter correlation function
    xi_nl = args["xi_nl"] #nonlinear matter correlation function
    R_lambda = args["R_lambda"] #Mpc/h comoving; richness radius
    richness = args["richness"] #redmapper richness
    z = args["z"] #redshift
    h = args["h"] #Hubble constant / 100 km/s/Mpc
    Omega_m = args["Omega_m"] #matter density fraction
    Sigma_crit_inverse = args["Sigma_crit_inverse"] #estimate of
    #the inverse of the critical surface mass density
    R_edges = args["R_edges"] #Mpc/h comoving; projected bin edges
    bias_spline = args["bias_spline"] #interpolator for halo bias

    #boolean variables for the analysis
    has_multiplicative_bias = args["has_multiplicative_bias"]
    has_reduced_shear = args["has_reduced_shear"]
    has_RM_selection = args["has_RM_selection"]
    has_miscentering = args["has_miscentering"]
    has_boost_factors = args["has_boost_factors"]

    output_dictionary = {}
    
    ####
    #(2) Pull out parameters
    ####
    M = parameter_dict["M"] #Msun/h; 200m
    c = parameter_dict["c"] #concentration; 200m
    
    if has_multiplicative_bias:
        Am = parameter_dict["Am"]
    if has_RM_selection:
        A = parameter_dict["A_matrix"] #linear parameters
        powers = args["powers"]
        upper_mask = args["upper_mask"]
        lower_mask = args["lower_mask"]
        lowest_index = args["lowest_index"]
        x = args["x"] #log(R) with a pivot at 30 Mpc/h
        X = np.ones((len(x), len(A))) #Matrix of powers of ln(R)
        for i in range(0, len(A)):
            X[:, i] = A[i] * x**powers[i]
    if has_miscentering:
        f_mis = parameter_dict["f_mis"]
        tau = parameter_dict["tau_mis"]
        R_mis = tau*R_lambda
    if has_boost_factors:
        B_0 = parameter_dict["B_0"]
        R_scale_boost = parameter_dict["R_scale_boost"]

    ####
    #(3) Compute the basic parts of the model
    ####
    xi_nfw = ct.xi.xi_nfw_at_R(r, M, c, Omega_m)
    bias = bias_spline(np.log(M))
    xi_2h = ct.xi.xi_2halo(bias, xi_nl) #Can choose a different xi_mm here
    xi_hm = ct.xi.xi_hm(xi_nfw, xi_2h) #3d halo matter correlation function

    output_dict["r_3d"] = r
    output_dict["xi_nfw"] = xi_nfw
    output_dict["xi_2h"] = xi_2h
    output_dict["xi_hm"] = xi_hm

    #h Msun/pc^2 comoving
    Sigma = ct.deltasigma.Sigma_at_R(R, r, xi_hm, M, c, Omega_m)

    output_dict["R_2d"] = R
    output_dict["Sigma"] = Sigma

    if has_RM_selection:
        F_model = 1 + X @ A
        F_model[upper_mask] = 1. #large scales are unaffected
        F_model[lower_mask] = F_model[lowest_index] #constant at small scales
        Sigma *= F_model
        output_dict["Sigma_RM_selection"] = Sigma

    if has_miscentering:
        print("Miscentering not implemented yet!")

    if has_multiplicative_bias:
        print("Multiplicative bias not implemented yet!")

    if has_reduced_shear:
        print("Reduced shear not implemented yet!")

    if has_boost_factors:
        print("Boost factors not implemented yet!")

    DeltaSigma = ct.deltasigma.DeltaSigma_at_R(R, R, Sigma, M, c, Omega_m)
    
    output_dict["DeltaSigma"] = DeltaSigma
    
    ####
    #(4) Average over radial bins
    ####
    ave_DeltaSigma = ct.averaging.average_profile_in_bins(R_edges, R, DeltaSigma)

    output_dict["R_edges"] = R_edges
    output_dict["ave_DeltaSigma"] = ave_DeltaSigma

    if return_all_parts:
        return output_dict

    return ave_DeltaSigma
