"""
units.py

Utilities for RAMD testing
"""

import numpy as np

def slant_time(a, b, D, beta, force_constant, use_abs=False):
    if use_abs:
        dist = abs(a-b)
    else:
        dist = a-b
    if np.isclose(force_constant, 0.0):
        time = dist**2/(2.0*D)
    else:
        time = ((np.exp(-beta*force_constant*dist) - 1.0)\
                /(beta*force_constant) + dist)/(beta*force_constant*D)
    return time