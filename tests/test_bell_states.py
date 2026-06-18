"""
test_bell_states.py
=====================

Unit tests for quantum entanglement simulation modules.

Run with: pytest test_bell_states.py -v
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bell_states import (
    bell_state, density_matrix, partial_trace, werner_state,
    fidelity, von_neumann_entropy, validate_density_matrix, ket
)
from chsh_inequality import quantum_correlation, chsh_parameter, optimal_angles
from entanglement_monotones import concurrence, werner_concurrence_analytical


class TestBellStates:
    """Tests for Bell state construction and properties."""

    def test_bell_state_normalization(self):
        """All Bell states must be normalized."""
        for name in ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]:
            psi = bell_state(name)
            norm = np.linalg.norm(psi)
            assert np.isclose(norm, 1.0), f"{name} not normalized: {norm}"

    def test_bell_state_orthogonality(self):
        """Different Bell states must be orthogonal."""
        states = [bell_state(n) for n in ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]]
        for i in range(4):
            for j in range(i+1, 4):
                overlap = abs(states[i].conj().T @ states[j])
                assert np.isclose(overlap, 0.0), f"States {i},{j} not orthogonal: {overlap}"

    def test_density_matrix_properties(self):
        """Density matrices must be Hermitian, positive, trace-1."""
        psi = bell_state("phi_plus")
        rho = density_matrix(psi)
        assert validate_density_matrix(rho)

    def test_partial_trace_maximally_mixed(self):
        """Reduced state of Bell state must be maximally mixed."""
        psi = bell_state("phi_plus")
        rho = density_matrix(psi)
        rho_A = partial_trace(rho, (2, 2), subsystem=1)
        expected = np.eye(2) / 2
        assert np.allclose(rho_A, expected), f"Reduced state: {rho_A}"


class TestWernerStates:
    """Tests for Werner state properties."""

    def test_werner_valid_density_matrix(self):
        """Werner states must be valid density matrices."""
        for p in [0.0, 0.25, 1/3, 0.5, 1.0]:
            rho = werner_state(p)
            assert validate_density_matrix(rho)

    def test_werner_pure_state_limit(self):
        """p=1 should give pure Bell state."""
        rho_w = werner_state(1.0)
        rho_bell = density_matrix(bell_state("phi_plus"))
        assert np.allclose(rho_w, rho_bell)

    def test_werner_maximally_mixed_limit(self):
        """p=0 should give maximally mixed state."""
        rho_w = werner_state(0.0)
        expected = np.eye(4) / 4
        assert np.allclose(rho_w, expected)


class TestCHSH:
    """Tests for CHSH inequality calculations."""

    def test_quantum_correlation_range(self):
        """Quantum correlations must be in [-1, 1]."""
        for a in np.linspace(0, 2*np.pi, 10):
            for b in np.linspace(0, 2*np.pi, 10):
                E = quantum_correlation(a, b)
                assert -1.0 <= E <= 1.0

    def test_chsh_tsirelson_bound(self):
        """CHSH parameter must reach Tsirelson bound."""
        a1, a2, b1, b2 = optimal_angles()
        S = chsh_parameter(a1, a2, b1, b2)
        assert np.isclose(abs(S), 2*np.sqrt(2), rtol=1e-10)

    def test_chsh_antisymmetric(self):
        """CHSH parameter should be antisymmetric under angle shifts."""
        S1 = chsh_parameter(0, np.pi/4, np.pi/8, 3*np.pi/8)
        S2 = chsh_parameter(np.pi, np.pi/4 + np.pi, np.pi/8, 3*np.pi/8)
        assert np.isclose(S1, S2, rtol=1e-10)


class TestEntanglementMonotones:
    """Tests for entanglement measures."""

    def test_concurrence_pure_bell_state(self):
        """Concurrence of pure Bell state must be 1."""
        psi = bell_state("phi_plus")
        rho = density_matrix(psi)
        C = concurrence(rho)
        assert np.isclose(C, 1.0)

    def test_concurrence_separable_state(self):
        """Concurrence of separable state must be 0."""
        rho = np.diag([0.5, 0, 0, 0.5])
        C = concurrence(rho)
        assert np.isclose(C, 0.0)

    def test_werner_concurrence_analytical(self):
        """Numerical concurrence must match analytical formula."""
        for p in np.linspace(0, 1, 20):
            rho = werner_state(p)
            C_num = concurrence(rho)
            C_ana = werner_concurrence_analytical(p)
            assert np.isclose(C_num, C_ana, atol=1e-10)

    def test_werner_separability_threshold(self):
        """Concurrence must vanish at p = 1/3."""
        rho = werner_state(1/3)
        C = concurrence(rho)
        assert np.isclose(C, 0.0, atol=1e-10)


class TestVonNeumannEntropy:
    """Tests for entropy calculations."""

    def test_pure_state_zero_entropy(self):
        """Pure states must have zero entropy."""
        psi = bell_state("phi_plus")
        rho = density_matrix(psi)
        S = von_neumann_entropy(rho)
        assert np.isclose(S, 0.0, atol=1e-10)

    def test_maximally_mixed_entropy(self):
        """Maximally mixed state of dimension d must have entropy log2(d)."""
        rho = np.eye(4) / 4
        S = von_neumann_entropy(rho)
        assert np.isclose(S, 2.0)

    def test_entropy_subadditivity(self):
        """Entropy must satisfy subadditivity for product states."""
        rho_A = np.array([[0.7, 0], [0, 0.3]])
        rho_B = np.array([[0.6, 0], [0, 0.4]])
        rho_AB = np.kron(rho_A, rho_B)

        S_AB = von_neumann_entropy(rho_AB)
        S_A = von_neumann_entropy(rho_A)
        S_B = von_neumann_entropy(rho_B)

        assert S_AB <= S_A + S_B + 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
