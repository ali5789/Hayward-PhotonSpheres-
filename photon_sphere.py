# photon_sphere.py
import numpy as np
from scipy.optimize import root_scalar

def f(r, l, M=1):
    return 1 - (2*M*r**2)/(r**3 + l**3)

def photon_condition(r, l, K, k, M=1):
    term1 = f(r, l, M) / r**2
    term2 = (1 - K / r**k)
    # Derivative calculation (simplified form of Eq.13)
    h = 1e-5
    f_prime = (term1*term2)(r+h) - (term1*term2)(r-h)) / (2*h)
    return f_prime

# Compute r_ph for given (l, K, k)
def solve_r_ph(l, K, k, r_min=2.0, r_max=4.0):
    sol = root_scalar(lambda r: photon_condition(r, l, K, k), 
                      bracket=[r_min, r_max], 
                      method='brentq')
    return sol.root

# Example: Reproduce Table 1 values
if __name__ == "__main__":
    print("l\tK\tk\tr_ph")
    for l, K, k in [(0.0,0.0,1), (0.4,0.1,1), (0.7,0.3,2), (1.0,0.5,2)]:
        r_ph = solve_r_ph(l, K, k)
        print(f"{l}\t{K}\t{k}\t{r_ph:.6f}")
