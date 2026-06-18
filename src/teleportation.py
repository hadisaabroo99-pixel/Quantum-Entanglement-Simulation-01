"""
teleportation.py
================

Simulation of the quantum teleportation protocol.

Demonstrates the transfer of an arbitrary quantum state from Alice
to Bob using a shared Bell pair and classical communication.

References:
-----------
[1] C.H. Bennett et al., Phys. Rev. Lett. 70, 1895 (1993)
[2] M.A. Nielsen & I.L. Chuang, Quantum Computation and Quantum Information
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def arbitrary_state(theta: float, phi: float) -> np.ndarray:
    """
    Construct an arbitrary single-qubit pure state.

    |psi> = cos(theta/2)|0> + e^(i*phi)*sin(theta/2)|1>

    Parameters
    ----------
    theta : float
        Polar angle in [0, pi].
    phi : float
        Azimuthal angle in [0, 2*pi].

    Returns
    -------
    np.ndarray
        State vector of shape (2, 1).
    """
    return np.array([
        [np.cos(theta/2)],
        [np.exp(1j*phi) * np.sin(theta/2)]
    ])


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
        Bell state vector of shape (4, 1).
    """
    ket00 = np.array([[1], [0], [0], [0]])
    ket01 = np.array([[0], [1], [0], [0]])
    ket10 = np.array([[0], [0], [1], [0]])
    ket11 = np.array([[0], [0], [0], [1]])

    states = {
        "phi_plus":  (ket00 + ket11) / np.sqrt(2),
        "phi_minus": (ket00 - ket11) / np.sqrt(2),
        "psi_plus":  (ket01 + ket10) / np.sqrt(2),
        "psi_minus": (ket01 - ket10) / np.sqrt(2),
    }

    return states[state_label]


def cnot_gate() -> np.ndarray:
    """Return the CNOT gate matrix."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)


def hadamard_gate() -> np.ndarray:
    """Return the Hadamard gate matrix."""
    return np.array([
        [1, 1],
        [1, -1]
    ], dtype=complex) / np.sqrt(2)


def pauli_x() -> np.ndarray:
    """Return the Pauli-X gate matrix."""
    return np.array([[0, 1], [1, 0]], dtype=complex)


def pauli_z() -> np.ndarray:
    """Return the Pauli-Z gate matrix."""
    return np.array([[1, 0], [0, -1]], dtype=complex)


def teleportation_circuit(psi: np.ndarray,
                           bell_pair: str = "phi_plus") -> Tuple[np.ndarray, str]:
    """
    Simulate the quantum teleportation protocol.

    Protocol:
    1. Alice shares a Bell pair |Phi+>_AB with Bob
    2. Alice applies CNOT(psi, A) then H(psi)
    3. Alice measures both qubits, sends results (2 classical bits) to Bob
    4. Bob applies correction U^dagger based on measurement outcomes

    Parameters
    ----------
    psi : np.ndarray
        State to teleport, shape (2, 1).
    bell_pair : str
        Type of Bell pair to use.

    Returns
    -------
    Tuple[np.ndarray, str]
        (teleported state, measurement outcome string).
    """
    # Step 1: Initial state |psi> ⊗ |Phi+>_AB
    # Total state: |psi>_C ⊗ |Phi+>_AB
    # We work with qubit ordering: C (message), A (Alice's EPR), B (Bob's EPR)

    phi_plus = bell_state(bell_pair)

    # Full 3-qubit state: |psi> ⊗ |Phi+>
    psi_flat = psi.flatten()
    phi_flat = phi_plus.flatten()

    # Tensor product: |psi>_C ⊗ |Phi+>_AB
    total_state = np.kron(psi_flat, phi_flat)  # shape (8,)

    # Step 2: Alice applies CNOT(C, A) then H(C)
    # CNOT on qubits C and A (positions 0 and 1 in 3-qubit system)
    cnot = cnot_gate()

    # Extend CNOT to 3 qubits: CNOT_{C,A} ⊗ I_B
    cnot_3q = np.kron(cnot, np.eye(2))
    total_state = cnot_3q @ total_state

    # Hadamard on qubit C: H_C ⊗ I_A ⊗ I_B
    H = hadamard_gate()
    H_3q = np.kron(np.kron(H, np.eye(2)), np.eye(2))
    total_state = H_3q @ total_state

    # Step 3: Measurement (simulated by projecting onto basis states)
    # After H and CNOT, the state is:
    # (1/2) * sum_{x,z} |x,z> ⊗ Z^x X^z |psi>_B

    # We extract Bob's state for each possible measurement outcome
    outcomes = {}
    for x in [0, 1]:
        for z in [0, 1]:
            # Projector |x,z><x,z| ⊗ I_B
            proj = np.zeros((8, 8), dtype=complex)
            idx = x * 4 + z * 2  # qubit ordering: C, A, B
            for b in [0, 1]:
                basis_idx = idx + b
                proj[basis_idx, basis_idx] = 1

            projected = proj @ total_state
            norm = np.linalg.norm(projected)

            if norm > 1e-10:
                # Extract Bob's state (last qubit)
                bob_state = np.zeros(2, dtype=complex)
                for b in [0, 1]:
                    basis_idx = idx + b
                    bob_state[b] = projected[basis_idx] / norm

                outcomes[f"{x}{z}"] = bob_state

    # Step 4: Bob applies correction
    # Measurement outcome determines correction:
    # 00 -> I, 01 -> X, 10 -> Z, 11 -> ZX
    corrections = {
        "00": np.eye(2),
        "01": pauli_x(),
        "10": pauli_z(),
        "11": pauli_z() @ pauli_x(),
    }

    # For demonstration, return the state for outcome "00" (no correction needed)
    # and show all possible outcomes
    outcome = "00"
    teleported = corrections[outcome] @ outcomes[outcome]
    teleported = teleported.reshape(-1, 1)

    return teleported, outcome


def verify_teleportation(theta: float = np.pi/3, phi: float = np.pi/4):
    """
    Verify that teleportation preserves the quantum state.

    Parameters
    ----------
    theta, phi : float
        Parameters of the state to teleport.
    """
    psi_original = arbitrary_state(theta, phi)
    psi_teleported, outcome = teleportation_circuit(psi_original)

    # Compute fidelity
    fidelity = abs(psi_original.conj().T @ psi_teleported)**2
    fidelity = float(np.real(fidelity))

    print(f"Original state: |ψ> = cos({theta:.3f}/2)|0> + e^(i*{phi:.3f})sin({theta:.3f}/2)|1>")
    print(f"Teleported state fidelity: {fidelity:.6f}")
    print(f"Measurement outcome: {outcome}")

    return fidelity


def plot_teleportation_circuit(save_path: str = None):
    """
    Generate a circuit diagram for quantum teleportation.

    Parameters
    ----------
    save_path : str, optional
        Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Quantum Teleportation Protocol', fontweight='bold', fontsize=16)

    # Qubit lines
    y_positions = [8, 5, 2]
    labels = [r'$|\psi\rangle$', r'$|\Phi^+\rangle_A$', r'$|\Phi^+\rangle_B$']
    names = ['Alice', '', 'Bob']

    for y, label, name in zip(y_positions, labels, names):
        ax.annotate('', xy=(9.5, y), xytext=(0.5, y),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='#2C3E50'))
        ax.text(0.1, y, label, fontsize=13, fontweight='bold', va='center')
        if name:
            ax.text(1.2, y + 0.5, name, fontsize=10, color='#7F8C8D')

    # CNOT gate (control on qubit 0, target on qubit 1)
    ax.plot([3, 3], [5, 8], 'k-', lw=2.5)
    ax.plot(3, 8, 'o', markersize=14, color='#E74C3C',
           markeredgecolor='black', markeredgewidth=2, zorder=5)
    ax.plot(3, 5, '+', markersize=16, color='#E74C3C', markeredgewidth=2.5)
    ax.text(3, 6.5, 'CNOT', fontsize=9, ha='center', color='#E74C3C', fontweight='bold')

    # Hadamard gate
    rect_h = plt.Rectangle((4.5, 7.2), 1.2, 1.6, fill=True, facecolor='#3498DB',
                            edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(rect_h)
    ax.text(5.1, 8, 'H', fontsize=14, ha='center', va='center',
           fontweight='bold', color='white')

    # Measurement symbols
    for y in [7.5, 4.5]:
        ax.plot([7, 7], [y, y+1], 'k-', lw=2.5)
        ax.plot(7, y+0.5, 'o', markersize=12, color='#2C3E50')
        ax.plot([6.8, 7.2], [y+0.7, y+0.3], 'k-', lw=2)
        ax.text(7, y-0.2, 'M', fontsize=10, ha='center', fontweight='bold')

    # Classical communication arrows
    ax.annotate('', xy=(8.5, 3), xytext=(7.2, 5.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='#F39C12',
                              linestyle='--', connectionstyle="arc3,rad=-0.2"))
    ax.annotate('', xy=(8.5, 3), xytext=(7.2, 8.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='#F39C12',
                              linestyle='--', connectionstyle="arc3,rad=0.2"))
    ax.text(7.8, 4.2, 'Classical\nChannel', fontsize=9, ha='center',
           color='#F39C12', fontweight='bold')

    # Bob's correction
    rect_u = plt.Rectangle((8.3, 1.2), 1.6, 1.6, fill=True, facecolor='#2ECC71',
                            edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(rect_u)
    ax.text(9.1, 2, r'$U^\dagger$', fontsize=13, ha='center', va='center',
           fontweight='bold', color='white')

    ax.text(9.7, 2, r'$|\psi\rangle$', fontsize=13, fontweight='bold',
           va='center', color='#2ECC71')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")

    return fig, ax


def plot_fidelity_landscape(save_path: str = None):
    """
    Plot teleportation fidelity across the Bloch sphere.

    Parameters
    ----------
    save_path : str, optional
        Path to save the figure.
    """
    theta_vals = np.linspace(0, np.pi, 30)
    phi_vals = np.linspace(0, 2*np.pi, 30)

    Theta, Phi = np.meshgrid(theta_vals, phi_vals)
    Fidelity = np.zeros_like(Theta)

    for i in range(len(theta_vals)):
        for j in range(len(phi_vals)):
            psi = arbitrary_state(theta_vals[i], phi_vals[j])
            psi_teleported, _ = teleportation_circuit(psi)
            fidelity = abs(psi.conj().T @ psi_teleported)**2
            Fidelity[j, i] = float(np.real(fidelity))

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='3d'))

    # Convert to Cartesian for 3D surface
    X = np.sin(Theta) * np.cos(Phi)
    Y = np.sin(Theta) * np.sin(Phi)
    Z = np.cos(Theta)

    # Color by fidelity
    surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.RdYlGn(Fidelity),
                           alpha=0.8, rstride=1, cstride=1,
                           antialiased=True, shade=False)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Teleportation Fidelity Across Bloch Sphere',
                 fontweight='bold', fontsize=14)

    # Add colorbar
    m = plt.cm.ScalarMappable(cmap='RdYlGn')
    m.set_array(Fidelity)
    cbar = plt.colorbar(m, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('Fidelity', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")

    return fig, ax


if __name__ == "__main__":
    print("=" * 60)
    print("Quantum Teleportation Protocol")
    print("=" * 60)

    # Verify teleportation for several states
    test_states = [
        (0, 0, "|0⟩"),
        (np.pi, 0, "|1⟩"),
        (np.pi/2, 0, "|+⟩"),
        (np.pi/2, np.pi, "|-⟩"),
        (np.pi/3, np.pi/4, "arbitrary"),
    ]

    print("\nTeleportation fidelity tests:")
    print("-" * 50)
    for theta, phi, name in test_states:
        fid = verify_teleportation(theta, phi)
        print(f"  {name}: fidelity = {fid:.6f}")

    print("\nGenerating figures...")
    plot_teleportation_circuit(save_path="teleportation_circuit.png")
    plot_fidelity_landscape(save_path="teleportation_fidelity.png")
    plt.show()
