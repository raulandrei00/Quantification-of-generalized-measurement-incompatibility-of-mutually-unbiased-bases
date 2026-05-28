"""
Additional test measurement sets for incompatibility_sdp.py.

We feed each "sharp" set (eta_input = 1) through the SDPs and read off the
white-noise robustness eta*, then cross-check with the feasibility SDP at
eta* * (1 ± epsilon).
"""

import numpy as np
from scipy.linalg import expm

from incompatibility_sdp import (
    feasibility_eq10,
    robustness_eq12,
    noisy_pauli_measurements,
)


# ----------------------------------------------------------------------
# Test set 1:  trine on the X–Z plane (qubit, 2 outcomes each)
# ----------------------------------------------------------------------
def trine_measurements(eta: float) -> np.ndarray:
    """
    Three coplanar projective qubit measurements with Bloch vectors at
    angles 0, 2pi/3, 4pi/3 in the X-Z plane.  The three vectors sum to
    zero (linearly dependent), unlike the mutually orthogonal Paulis.
    """
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    M = np.zeros((3, 2, 2, 2), dtype=complex)
    for x in range(3):
        theta = 2 * np.pi * x / 3
        n_sigma = np.cos(theta) * X + np.sin(theta) * Z
        Pi0 = (I + n_sigma) / 2
        Pi1 = (I - n_sigma) / 2
        M[x, 0] = eta * Pi0 + (1 - eta) * I / 2
        M[x, 1] = eta * Pi1 + (1 - eta) * I / 2
    return M


# ----------------------------------------------------------------------
# Test set 2:  random global unitary applied to noisy Paulis
# ----------------------------------------------------------------------
def random_rotated_paulis(eta: float, seed: int = 0) -> np.ndarray:
    """
    Pauli measurements conjugated by a random Haar-distributed unitary.
    The genuine-triplewise threshold is unitarily invariant, so it must
    be exactly (sqrt(2)+1)/3 regardless of the rotation.
    """
    rng = np.random.default_rng(seed)
    # Haar unitary via QR of complex Gaussian (standard trick).
    Z = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    Q, R = np.linalg.qr(Z)
    U = Q @ np.diag(np.diag(R) / np.abs(np.diag(R)))

    M = noisy_pauli_measurements(eta)
    out = np.zeros_like(M)
    for x in range(3):
        for a in range(M.shape[1]):
            out[x, a] = U @ M[x, a] @ U.conj().T
    return out


# ----------------------------------------------------------------------
# Test set 3:  three MUBs in dimension 3 (qutrit)
# ----------------------------------------------------------------------
def qutrit_mubs(eta: float) -> np.ndarray:
    """
    Three mutually unbiased bases in dimension d = 3 as 3-outcome
    projective measurements (computational + two "tilted Fouriers"
    from the Heisenberg–Weyl construction).
    """
    d = 3
    omega = np.exp(2j * np.pi / d)
    I = np.eye(d, dtype=complex)

    bases = []
    # Computational basis.
    bases.append(np.eye(d, dtype=complex))
    # Two non-trivial MUBs:  |e^k_l> = (1/sqrt(d)) sum_j w^{k j^2 + l j} |j>.
    for k in (0, 1):
        B = np.zeros((d, d), dtype=complex)
        for l in range(d):
            for j in range(d):
                B[j, l] = omega ** (k * j * j + l * j) / np.sqrt(d)
        bases.append(B)

    M = np.zeros((3, d, d, d), dtype=complex)
    for x, B in enumerate(bases):
        for a in range(d):
            v = B[:, a]
            Pi = np.outer(v, v.conj())
            M[x, a] = eta * Pi + (1 - eta) * I / d
    return M


# ----------------------------------------------------------------------
# Run the new tests
# ----------------------------------------------------------------------
def _feas(M):
    """Feasibility (eq. 10): try CLARABEL, fall back to SCS at higher precision."""
    try:
        return feasibility_eq10(M, solver="CLARABEL"), "CLARABEL"
    except Exception:
        # eps=1e-7 instead of SCS's default 1e-4 -- the noisy MUB problem
        # needs the tighter tolerance to resolve infeasibility cleanly.
        return feasibility_eq10(M, solver="SCS", eps=1e-7), "SCS"


def _robust(M):
    """Robustness (eq. 12): try CLARABEL, fall back to SCS at higher precision."""
    try:
        val, _ = robustness_eq12(M, solver="CLARABEL")
        return val, "CLARABEL"
    except Exception:
        val, _ = robustness_eq12(M, solver="SCS", eps=1e-7)
        return val, "SCS"


def _row(label, M_sharp_fn, expected=None):
    M = M_sharp_fn(1.0)
    is_compat, solver_used = _feas(M)
    eta_star, _ = _robust(M)
    # Cross-check feasibility at +/- 5% of eta*  (5% > SCS's numerical slack).
    below, _ = _feas(M_sharp_fn(eta_star * 0.95))
    above, _ = _feas(M_sharp_fn(min(eta_star * 1.05, 1.0)))

    line = (f"{label:<42} sharp incompat.: {'YES' if not is_compat else 'no '}   "
            f"eta* = {eta_star:.6f}   [{solver_used}]")
    if expected is not None:
        line += f"   (expected {expected:.6f})"
    print(line)
    print(f"   cross-check feasibility:  "
          f"eta = 0.95 eta*  ->  {'feas' if below else 'infeas'};   "
          f"eta = 1.05 eta*  ->  {'feas' if above else 'infeas'}")


def main():
    print("=" * 78)
    print(" Additional incompatible-measurement test cases")
    print("=" * 78)

    pauli_threshold = (np.sqrt(2) + 1) / 3

    _row("Sharp Paulis (X, Y, Z), d=2", noisy_pauli_measurements, pauli_threshold)
    _row("Trine 120 deg in X-Z plane, d=2", trine_measurements)
    _row("Random rotated Paulis, d=2 (seed=0)",
         lambda e: random_rotated_paulis(e, seed=0), pauli_threshold)
    _row("Random rotated Paulis, d=2 (seed=7)",
         lambda e: random_rotated_paulis(e, seed=7), pauli_threshold)
    _row("Three MUBs in d=3 (qutrit)", qutrit_mubs)


if __name__ == "__main__":
    main()
