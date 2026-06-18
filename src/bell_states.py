"""
bell_states.py
==============

Bell state preparation, density matrix construction, and subsystem analysis.

Implements the four maximally entangled Bell states and provides tools
for density matrix manipulation including partial trace and fidelity
calculations.

References:
-----------
[1] M.A. Nielsen & I.L. Chuang, Quantum Computation and Quantum Information
[2] R.F. Werner, Phys. Rev. A 40, 4277 (1989)
"""

import numpy as np
from typing import Tuple, Optional


def ket(label: str) -> np.ndarray:
    """
    Return a computational basis state vector.

    Parameters
    ----------
    label : str
        Binary string, e.g., '00', '01', '10', '11'.

    Returns
    -------
    np.ndarray
        Column vector of shape (4, 1) for two-qubit states.
    """
    n = len(label)
    idx = int(label, 2)
    state = np.zeros((2**n, 1), dtype=complex)
    state[idx] = 1.0
    return state


def bell_state(state_label: str = "phi_plus") -> np.ndarray:
    """
    Construct a Bell state vector.

    Parameters
    ----------
    state_label : str
        One of 'phi_plus', 'phi_minus', 'psi_plus', 'psi_minus'.

    Returns
    -------
    np.ndarray
        Normalized Bell state vector of shape (4, 1).

    Raises
    ------
    ValueError
        If state_label is not recognized.
    """
    ket00 = ket("00")
    ket01 = ket("01")
    ket10 = ket("10")
    ket11 = ket("11")

    states = {
        "phi_plus":  (ket00 + ket11) / np.sqrt(2),
        "phi_minus": (ket00 - ket11) / np.sqrt(2),
        "psi_plus":  (ket01 + ket10) / np.sqrt(2),
        "psi_minus": (ket01 - ket10) / np.sqrt(2),
    }

    if state_label not in states:
        raise ValueError(f"Unknown Bell state: {state_label}. "
                         f"Choose from {list(states.keys())}")

    return states[state_label]


def density_matrix(state_vector: np.ndarray) -> np.ndarray:
    """
    Construct density matrix from pure state vector.

    rho = |psi><psi|

    Parameters
    ----------
    state_vector : np.ndarray
        State vector of shape (d, 1) or (d,).

    Returns
    -------
    np.ndarray
        Density matrix of shape (d, d).
    """
    psi = state_vector.reshape(-1, 1)
    return psi @ psi.conj().T


def partial_trace(rho: np.ndarray, dims: Tuple[int, int],
                  subsystem: int = 1) -> np.ndarray:
    """
    Compute the partial trace over a subsystem.

    For a bipartite system with dimensions (d_A, d_B), traces out
    the specified subsystem to return the reduced density matrix.

    Parameters
    ----------
    rho : np.ndarray
        Full density matrix of shape (d_A * d_B, d_A * d_B).
    dims : Tuple[int, int]
        Dimensions (d_A, d_B) of the two subsystems.
    subsystem : int
        Subsystem to trace out: 0 for A, 1 for B (default).

    Returns
    -------
    np.ndarray
        Reduced density matrix of the remaining subsystem.
    """
    d_A, d_B = dims
    assert rho.shape == (d_A * d_B, d_A * d_B),         f"Density matrix shape {rho.shape} incompatible with dims {dims}"

    # Reshape to tensor form: rho_{(i,j),(k,l)} -> rho_{i,j,k,l}
    rho_tensor = rho.reshape(d_A, d_B, d_A, d_B)

    if subsystem == 1:
        # Trace out B: sum over j index
        rho_reduced = np.trace(rho_tensor, axis1=1, axis2=3)
    else:
        # Trace out A: sum over i index
        rho_reduced = np.trace(rho_tensor, axis1=0, axis2=2)

    return rho_reduced


def werner_state(p: float) -> np.ndarray:
    """
    Construct the Werner state density matrix.

    rho_W(p) = p * |Phi+><Phi+| + (1-p) * I/4

    Parameters
    ----------
    p : float
        Werner parameter in [0, 1].
        Separable for p <= 1/3, entangled for p > 1/3.

    Returns
    -------
    np.ndarray
        Werner state density matrix of shape (4, 4).
    """
    if not (0 <= p <= 1):
        raise ValueError(f"Werner parameter p must be in [0, 1], got {p}")

    phi_plus = bell_state("phi_plus")
    rho_pure = density_matrix(phi_plus)
    I_4 = np.eye(4) / 4
    return p * rho_pure + (1 - p) * I_4


def fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """
    Compute quantum fidelity between two density matrices.

    F(rho1, rho2) = Tr(sqrt(sqrt(rho1) * rho2 * sqrt(rho1)))^2

    For pure states, F = |<psi1|psi2>|^2.

    Parameters
    ----------
    rho1, rho2 : np.ndarray
        Density matrices.

    Returns
    -------
    float
        Fidelity in [0, 1].
    """
    # Compute sqrt(rho1)
    eigvals1, eigvecs1 = np.linalg.eigh(rho1)
    sqrt_rho1 = eigvecs1 @ np.diag(np.sqrt(np.maximum(eigvals1, 0))) @ eigvecs1.conj().T

    # Compute sqrt(sqrt_rho1 * rho2 * sqrt_rho1)
    M = sqrt_rho1 @ rho2 @ sqrt_rho1
    eigvals_M = np.linalg.eigvalsh(M)

    return float(np.sum(np.sqrt(np.maximum(eigvals_M, 0)))**2)


def von_neumann_entropy(rho: np.ndarray) -> float:
    """
    Compute von Neumann entropy S(rho) = -Tr(rho log2 rho).

    Parameters
    ----------
    rho : np.ndarray
        Density matrix (must be positive semi-definite).

    Returns
    -------
    float
        von Neumann entropy in bits.
    """
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.maximum(eigvals, 1e-15)  # Regularize
    return float(-np.sum(eigvals * np.log2(eigvals)))


def validate_density_matrix(rho: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Validate that a matrix is a proper density matrix.

    Checks: Hermiticity, positive semi-definiteness, unit trace.

    Parameters
    ----------
    rho : np.ndarray
        Candidate density matrix.
    tol : float
        Numerical tolerance.

    Returns
    -------
    bool
        True if rho is a valid density matrix.
    """
    # Hermiticity
    if not np.allclose(rho, rho.conj().T, atol=tol):
        print("FAIL: Not Hermitian")
        return False

    # Unit trace
    if not np.isclose(np.trace(rho), 1.0, atol=tol):
        print(f"FAIL: Trace = {np.trace(rho)} != 1")
        return False

    # Positive semi-definite
    eigvals = np.linalg.eigvalsh(rho)
    if np.any(eigvals < -tol):
        print(f"FAIL: Negative eigenvalues: {eigvals[eigvals < -tol]}")
        return False

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Bell States - Validation Tests")
    print("=" * 60)

    # Test all four Bell states
    for name in ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]:
        psi = bell_state(name)
        rho = density_matrix(psi)
        valid = validate_density_matrix(rho)
        rho_A = partial_trace(rho, (2, 2), subsystem=1)
        S = von_neumann_entropy(rho_A)
        print(f"\n{name}:")
        print(f"  Valid density matrix: {valid}")
        print(f"  Reduced entropy S(rho_A) = {S:.6f} (expected: 1.0)")
        print(f"  Fidelity with self: {fidelity(rho, rho):.6f}")

    # Test Werner states
    print("\n" + "=" * 60)
    print("Werner State Tests")
    print("=" * 60)
    for p in [0.0, 1/3, 0.5, 1.0]:
        rho_w = werner_state(p)
        valid = validate_density_matrix(rho_w)
        S = von_neumann_entropy(rho_w)
        print(f"\np = {p:.4f}: valid={valid}, S(rho) = {S:.6f}")
