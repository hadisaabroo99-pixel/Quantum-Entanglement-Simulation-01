"""
decoherence.py
==============

Simulation of decoherence in entangled two-qubit systems.

Models the decay of off-diagonal density matrix elements due to
environmental coupling, and computes the resulting entanglement
degradation via von Neumann entropy.

References:
-----------
[1] W.H. Zurek, Rev. Mod. Phys. 75, 715 (2003)
[2] M.A. Nielsen & I.L. Chuang, Quantum Computation and Quantum Information
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional


def decohering_density_matrix(t: float, gamma: float = 0.3) -> np.ndarray:
    """
    Construct the density matrix of a decohering Bell state.

    Model: off-diagonal elements decay exponentially:
    rho_{00,11}(t) = (1/2) * exp(-gamma * t)

    Parameters
    ----------
    t : float
        Time (arbitrary units).
    gamma : float
        Decoherence rate.

    Returns
    -------
    np.ndarray
        Density matrix of shape (4, 4).
    """
    decay = np.exp(-gamma * t)

    rho = np.array([
        [0.5,    0,    0, 0.5*decay],
        [0,      0,    0,         0],
        [0,      0,    0,         0],
        [0.5*decay, 0, 0,       0.5]
    ], dtype=complex)

    return rho


def reduced_density_matrix_A(t: float, gamma: float = 0.3) -> np.ndarray:
    """
    Reduced density matrix for subsystem A under decoherence.

    For the decohering Bell state, the reduced matrix is:
    rho_A = (1/2) * [[1, exp(-gamma*t)],
                     [exp(-gamma*t), 1]]

    Parameters
    ----------
    t : float
        Time.
    gamma : float
        Decoherence rate.

    Returns
    -------
    np.ndarray
        Reduced density matrix of shape (2, 2).
    """
    decay = np.exp(-gamma * t)
    return 0.5 * np.array([[1, decay], [decay, 1]], dtype=complex)


def von_neumann_entropy_rho(t: float, gamma: float = 0.3) -> float:
    """
    Compute von Neumann entropy of reduced density matrix at time t.

    S(t) = -lambda_+(t)*log2(lambda_+(t)) - lambda_-(t)*log2(lambda_-(t))
    where lambda_+/-(t) = (1 +/- exp(-gamma*t))/2

    Parameters
    ----------
    t : float
        Time.
    gamma : float
        Decoherence rate.

    Returns
    -------
    float
        Entropy in bits.
    """
    decay = np.exp(-gamma * t)
    lambda_plus = 0.5 * (1 + decay)
    lambda_minus = 0.5 * (1 - decay)

    # Regularize to avoid log(0)
    lambda_plus = max(lambda_plus, 1e-15)
    lambda_minus = max(lambda_minus, 1e-15)

    S = -(lambda_plus * np.log2(lambda_plus) +
          lambda_minus * np.log2(lambda_minus))

    return float(S)


def entanglement_entropy_trajectory(t_max: float = 10.0, gamma: float = 0.3,
                                    n_points: int = 500) -> tuple:
    """
    Compute entanglement entropy over a time interval.

    Parameters
    ----------
    t_max : float
        Maximum time.
    gamma : float
        Decoherence rate.
    n_points : int
        Number of time points.

    Returns
    -------
    tuple
        (time_array, entropy_array)
    """
    t = np.linspace(0, t_max, n_points)
    S = np.array([von_neumann_entropy_rho(ti, gamma) for ti in t])
    return t, S


def plot_decoherence_dynamics(t_max: float = 10.0, gamma: float = 0.3,
                               save_path: str = None):
    """
    Generate the decoherence dynamics figure.

    Shows von Neumann entropy evolution from pure state to maximally mixed.

    Parameters
    ----------
    t_max : float
        Maximum time.
    gamma : float
        Decoherence rate.
    save_path : str, optional
        Path to save the figure.
    """
    t, S = entanglement_entropy_trajectory(t_max, gamma)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(t, S, color='#9B59B6', linewidth=2.5, label='von Neumann Entropy')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7,
               label='Maximally Mixed (S=1)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7,
               label='Pure State (S=0)')
    ax.fill_between(t, 0, S, alpha=0.2, color='#9B59B6')

    # Mark characteristic times
    tau = 1 / gamma
    ax.axvline(x=tau, color='red', linestyle=':', alpha=0.5,
               label=f'Characteristic time τ=1/γ={tau:.2f}')

    ax.set_xlabel('Time (arb. units)', fontweight='bold', fontsize=12)
    ax.set_ylabel(r'Entanglement Entropy $S(\rho)$ (bits)',
                  fontweight='bold', fontsize=12)
    ax.set_title('Decoherence Dynamics of a Bell State',
                 fontweight='bold', fontsize=14)
    ax.legend(loc='center right', fontsize=10)
    ax.set_xlim(0, t_max)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")

    return fig, ax


def plot_density_matrix_evolution(times: list = [0, 1, 3, 10],
                                   gamma: float = 0.3,
                                   save_path: str = None):
    """
    Visualize density matrix at different times during decoherence.

    Parameters
    ----------
    times : list
        List of times to visualize.
    gamma : float
        Decoherence rate.
    save_path : str, optional
        Path to save the figure.
    """
    n_times = len(times)
    fig, axes = plt.subplots(1, n_times, figsize=(4*n_times, 4))

    if n_times == 1:
        axes = [axes]

    for ax, t in zip(axes, times):
        rho = decohering_density_matrix(t, gamma)

        im = ax.imshow(np.abs(rho), cmap='RdYlBu_r', vmin=0, vmax=0.6,
                       interpolation='nearest')

        labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

        for i in range(4):
            for j in range(4):
                text_color = "white" if np.abs(rho[i, j]) > 0.3 else "black"
                ax.text(j, i, f'{np.abs(rho[i, j]):.2f}',
                       ha="center", va="center", color=text_color,
                       fontweight='bold', fontsize=10)

        ax.set_title(f't = {t}', fontweight='bold')

    plt.suptitle(r'Decohering $|\Phi^+\rangle$ Density Matrix $|
ho_{ij}|$',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")

    return fig, axes


def coherence_time(gamma: float, threshold: float = 0.95) -> float:
    """
    Compute the time for entropy to reach a fraction of maximum.

    Parameters
    ----------
    gamma : float
        Decoherence rate.
    threshold : float
        Fraction of maximum entropy (default 0.95).

    Returns
    -------
    float
        Characteristic time.
    """
    # Solve: S(t) = threshold * S_max = threshold
    # lambda_+ = (1 + exp(-gamma*t))/2
    # S = -lambda_+ log2(lambda_+) - lambda_- log2(lambda_-)

    from scipy.optimize import fsolve

    def equation(t):
        return von_neumann_entropy_rho(t, gamma) - threshold

    t_sol = fsolve(equation, x0=5.0)[0]
    return float(t_sol)


if __name__ == "__main__":
    print("=" * 60)
    print("Decoherence Dynamics")
    print("=" * 60)

    gamma = 0.3
    print(f"\nDecoherence rate: γ = {gamma}")
    print(f"Characteristic time: τ = 1/γ = {1/gamma:.2f}")

    # Test at key times
    test_times = [0, 0.5, 1.0, 2.0, 5.0, 10.0]
    print("\nEntropy evolution:")
    print("-" * 40)
    print(f"{'Time':>8} {'S(ρ)':>12} {'Decay factor':>15}")
    print("-" * 40)
    for t in test_times:
        S = von_neumann_entropy_rho(t, gamma)
        decay = np.exp(-gamma * t)
        print(f"{t:8.2f} {S:12.6f} {decay:15.6f}")

    # Compute 95% coherence time
    try:
        t_95 = coherence_time(gamma, 0.95)
        print(f"\nTime to reach 95% of max entropy: t_95 = {t_95:.2f}")
    except ImportError:
        print("\nSciPy not available for coherence time calculation")

    print("\nGenerating figures...")
    plot_decoherence_dynamics(save_path="decoherence_dynamics.png")
    plot_density_matrix_evolution(save_path="density_matrix_evolution.png")
    plt.show()
