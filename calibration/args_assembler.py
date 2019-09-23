"""This file fascilitates the assembly of the `args` dictionary. This dictionary
contains things like the data vector(s), the covariance, any precomputed
parts of the model like the matter power spectrum, and independent variables
like radial distances.

Cuts to the data vectors are performed here, as are linear modifications
to the covaraince such as the Hartlap correction.
"""

import numpy as np
import scipy as sp
import cluster_toolkit as ct

print("Still working on this!")
