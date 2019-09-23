"""This file contains the (log)probability density functions (PDFs, or 
probability distribution functions) for the lensing profile(s) and other
associated quantities (like boost factors). It also has priors.
"""

import numpy as np

def lnprior(parameter_dict, args):
    print("No priors yet!")
    return 0

def lnlike(parameter_dict, args):
    print("No likelihood yet!")
    return 0

def lnposterior(params, args):
    #Turn the parameter array into a dictionary
    #This is the part that is tuned by hand

    print("No posterior yet!")
    return 0
