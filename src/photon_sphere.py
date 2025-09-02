"""
Numerical solver for photon sphere radii in Hayward regular black holes with power-law plasma.
Author: Ali Hasnain
GitHub: https://github.com/ali5789/Hayward-PhotonSpheres-
"""

import numpy as np
from scipy.optimize import root_scalar

def f(r, l, M=1):
    """
    Calculate the Hayward metric function f(r).

    Parameters
    ----------
    r : float
        Radial coordinate from the black hole center
    l : float
        Regularization parameter (quantum correction scale)
    M : float, optional
        Black hole mass, default=1 (natural units)

    Returns
    -------
    float
        Value of the metric function at r
    """
    return 1 - (2*M*r**2)/(r**3 + l**3)

def photon_condition(r, l, K, k, M=1):
    """
    Compute the photon sphere condition (derivative of effective potential).

    Parameters
    ----------
    r : float
        Test radius for photon sphere
    l : float
        Regularization parameter
    K : float
        Plasma strength parameter
    k : int
        Plasma density exponent (k=1 for isothermal, k=2 for disk-like)
    M : float, optional
        Black hole mass, default=1

    Returns
    -------
    float
        Value of the photon sphere condition equation
    """
    term1 = f(r, l, M) / r**2
    term2 = (1 - K / r**k)
    h = 1e-5  # Small step for numerical derivative
    derivative = (term1*term2)(r+h) - (term1*term2)(r-h)) / (2*h)
    return derivative

def solve_r_ph(l, K, k, r_min=2.0, r_max=4.0):
    """
    Solve for the photon sphere radius using Brent's method.

    Parameters
    ----------
    l : float
        Regularization parameter
    K : float
        Plasma strength
    k : int
        Plasma exponent
    r_min : float, optional
        Minimum search radius, default=2.0
    r_max : float, optional
        Maximum search radius, default=4.0

    Returns
    -------
    float
        Photon sphere radius in units of M

    Raises
    ------
    ValueError
        If no root is found in the given bracket
    """
    try:
        sol = root_scalar(
            lambda r: photon_condition(r, l, K, k),
            bracket=[r_min, r_max],
            method='brentq'
        )
        return sol.root
    except ValueError as e:
        raise ValueError(f"No photon sphere found for l={l}, K={K}, k={k}") from e

if __name__ == "__main__":
    """
    Main execution: Reproduces Table 1 results from the paper.
    """
    print("l\tK\tk\tr_ph (M)")
    print("-----------------------")
    
    # Parameter sets matching Table 1
    params = [
        (0.0, 0.0, 1),  # Schwarzschild case
        (0.4, 0.1, 1),
        (0.7, 0.3, 2),
        (1.0, 0.5, 2)
    ]
    
    for l, K, k in params:
        try:
            r_ph = solve_r_ph(l, K, k)
            print(f"{l}\t{K}\t{k}\t{r_ph:.6f}")
        except ValueError as e:
            print(f"{l}\t{K}\t{k}\tERROR: {str(e)}")
