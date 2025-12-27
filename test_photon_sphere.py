"""
Unit tests for photon sphere calculations.
Run with: pytest test_photon_sphere.py
"""

import pytest
import numpy as np
from src.photon_sphere import (
    f, df_dr, photon_condition, solve_r_ph, 
    validate_schwarzschild, get_event_horizon
)

class TestMetricFunction:
    """Test Hayward metric function and derivatives."""
    
    def test_schwarzschild_limit(self):
        """l=0 should recover Schwarzschild."""
        r = 3.0
        assert np.isclose(f(r, l=0), 1 - 2/r, rtol=1e-10)
    
    def test_asymptotic_flatness(self):
        """f(r) → 1 as r → ∞."""
        assert np.isclose(f(1000, l=0.5), 1.0, atol=1e-6)
    
    def test_horizon_existence(self):
        """Event horizon should exist for physical l."""
        r_plus = get_event_horizon(l=0.5)
        assert 1.5 < r_plus < 2.5  # Physical range

class TestPhotonSphere:
    """Test photon sphere calculations."""
    
    def test_schwarzschild_photon_sphere(self):
        """Schwarzschild: r_ph = 3M exactly."""
        r_ph = solve_r_ph(l=0, K=0, k=1, r_min=2.0)
        assert np.isclose(r_ph, 3.0, atol=1e-6)
    
    def test_perlick_tsupko_k1(self):
        """Validate against Perlick & Tsupko (2015) for k=1."""
        r_ph = solve_r_ph(l=0, K=0.2, k=1.0, r_min=2.0)
        assert np.isclose(r_ph, 2.801, atol=0.01)
    
    def test_perlick_tsupko_k15(self):
        """Validate against Perlick & Tsupko (2015) for k=1.5."""
        r_ph = solve_r_ph(l=0, K=0.3, k=1.5, r_min=2.0)
        assert np.isclose(r_ph, 2.692, atol=0.01)
    
    def test_monotonic_decrease_with_l(self):
        """r_ph should decrease as l increases."""
        r1 = solve_r_ph(l=0.2, K=0.1, k=1)
        r2 = solve_r_ph(l=0.5, K=0.1, k=1)
        assert r2 < r1
    
    def test_monotonic_decrease_with_K(self):
        """r_ph should decrease as K increases."""
        r1 = solve_r_ph(l=0.5, K=0.1, k=1)
        r2 = solve_r_ph(l=0.5, K=0.3, k=1)
        assert r2 < r1
    
    def test_outside_horizon(self):
        """Photon sphere must be outside event horizon."""
        l = 0.7
        r_plus = get_event_horizon(l)
        r_ph = solve_r_ph(l=l, K=0.2, k=1)
        assert r_ph > r_plus

class TestEdgeCases:
    """Test boundary conditions and edge cases."""
    
    def test_large_K_no_photon_sphere(self):
        """Very large K may destroy photon sphere."""
        with pytest.raises(ValueError):
            solve_r_ph(l=0, K=1.0, k=1, r_max=10)  # Unphysical
    
    def test_large_l_limit(self):
        """l → M limit should still give valid solution."""
        r_ph = solve_r_ph(l=1.0, K=0.1, k=1)
        assert 2.0 < r_ph < 3.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
