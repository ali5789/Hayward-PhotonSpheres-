# Hayward-PhotonSpheres

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Numerical code for computing photon sphere radii in Hayward regular black holes immersed in power-law plasma media.

## Paper Reference

**Title:** Photon Sphere Structure in Hayward Regular Black Holes Immersed in Power-Law Plasma Media  
**Author:** Ali Hasnain  
**Affiliation:** Department of Physics, Govt. Postgraduate College, Jhang, Punjab, Pakistan  
**Email:** alii.hassnain601@gmail.com

##  Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/ali5789/Hayward-PhotonSpheres-.git
cd Hayward-PhotonSpheres-

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```python
from src.photon_sphere import solve_r_ph

# Schwarzschild case (l=0, K=0)
r_ph = solve_r_ph(l=0.0, K=0.0, k=1.0)
print(f"Photon sphere radius: {r_ph:.6f} M")  # Output: 3.000000 M

# Hayward with plasma (reproduces paper results)
r_ph = solve_r_ph(l=0.5, K=0.2, k=1.5)
print(f"Photon sphere radius: {r_ph:.6f} M")
```

### Run Validation Tests
```bash
python src/photon_sphere.py
```

Expected output:
```
==================================================
VALIDATION: Schwarzschild Limit
==================================================
Computed r_ph = 3.00000000 M
Expected      = 3.00000000 M
Error         = 1.23e-08
✅ PASSED
```

##  Reproducing Paper Results

### Table 1: Validation Against Perlick & Tsupko (2015)
```python
from src.photon_sphere import reproduce_table1
reproduce_table1()
```

### Figure 1: r_ph vs Regularization Parameter
```python
from src.photon_sphere import generate_figure1_data
import matplotlib.pyplot as plt

l_vals, r_ph_vals = generate_figure1_data(K=0.1, k=1.0, n_points=50)

plt.figure(figsize=(10, 6))
plt.plot(l_vals, r_ph_vals, 'b-', linewidth=2, label='Hayward + Plasma')
plt.axhline(y=3.0, color='r', linestyle='--', label='Schwarzschild')
plt.xlabel('Regularization parameter l/M', fontsize=14)
plt.ylabel('Photon sphere radius $r_{ph}$/M', fontsize=14)
plt.title('Figure 1: Effect of Quantum Regularization', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1.pdf', dpi=300)
plt.show()
```

##  Mathematica Verification

Cross-check results using independent Newton-Raphson implementation:
```bash
# Open in Mathematica
wolframscript src/verification.nb

# Or use Wolfram Engine
wolframscript -file src/verification.nb
```

##  Physical Parameters

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| Mass | M | 1 (natural units) | Black hole mass |
| Regularization | l | 0–1 M | Quantum correction scale |
| Plasma strength | K | 0–0.5 | Dimensionless plasma parameter |
| Density exponent | k | 1–2 | Power-law index (1=isothermal, 2=disk) |

##  Key Equations

**Hayward Metric:**
```
f(r) = 1 - 2Mr²/(r³ + l³)
```

**Photon Sphere Condition (Eq. 17):**
```
d/dr[f(r)/r² × (1 - K/r^k)] = 0
```

**Plasma Refractive Index:**
```
n²(r) = 1 - ωₚ²(r)/ω² ,  ωₚ² ∝ r^(-k)
```

##  Results Summary

- **Schwarzschild limit:** Validated to within 10⁻⁸ M
- **Maximum r_ph shift:** ~18% for K=0.5
- **Nonlinear coupling:** 60% enhancement at l≈0.8M, K≈0.3
- **EHT relevance:** 4–12% shadow diameter reduction

##  Testing
```bash
# Run all validation tests
python src/photon_sphere.py

# Expected output: All tests pass 
```

##  Dependencies

- Python ≥ 3.8
- NumPy ≥ 1.24
- SciPy ≥ 1.11
- Matplotlib ≥ 3.7 (optional, for plotting)

##  Contact

**Ali Hasnain**  
Email: alii.hassnain601@gmail.com  
Institution: Govt. Postgraduate College, Jhang, Punjab, Pakistan

##  License

MIT License - see [LICENSE](LICENSE) file

##  Acknowledgments

- Hayward (2006) for the regular black hole framework
- Perlick & Tsupko (2015) for plasma dispersion formalism
- Event Horizon Telescope Collaboration for observational context

##  Citation

If you use this code, please cite:
```bibtex
@article{Hasnain2024,
  title={Photon Sphere Structure in Hayward Regular Black Holes Immersed in Power-Law Plasma Media},
  author={Hasnain, Ali},
  journal={[Journal Name]},
  year={2024},
  note={Code: https://github.com/ali5789/Hayward-PhotonSpheres-}
}
```
