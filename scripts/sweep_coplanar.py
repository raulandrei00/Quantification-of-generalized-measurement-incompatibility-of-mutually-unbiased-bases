"""
Sweep over the planar-trio angle theta:  three coplanar qubit measurements
in the X-Z plane at angles 0, theta, 2*theta. For each theta, compute the
white-noise genuine-triplewise-incompatibility threshold eta*(theta) via
equation (12) and plot the curve, marking the symmetric trine at
theta = 2 pi / 3 where we conjecture eta* = (1 + sqrt(3)) / 3.
"""

import numpy as np
import matplotlib.pyplot as plt

from scripts.incompatibility_sdp import robustness_eq12


# ----------------------------------------------------------------------
# 1.  Family of measurement triples
# ----------------------------------------------------------------------
def coplanar_triple(theta: float, eta: float = 1.0) -> np.ndarray:
    """Three coplanar qubit measurements in X-Z plane at angles 0, theta, 2 theta."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    M = np.zeros((3, 2, 2, 2), dtype=complex)
    for x, ang in enumerate((0.0, theta, 2.0 * theta)):
        n_sigma = np.cos(ang) * X + np.sin(ang) * Z
        Pi0 = (I + n_sigma) / 2
        Pi1 = (I - n_sigma) / 2
        M[x, 0] = eta * Pi0 + (1 - eta) * I / 2
        M[x, 1] = eta * Pi1 + (1 - eta) * I / 2
    return M


# ----------------------------------------------------------------------
# 2.  Sweep
# ----------------------------------------------------------------------
def _solve(M):
    try:
        v, _ = robustness_eq12(M, solver="CLARABEL")
    except Exception:
        v, _ = robustness_eq12(M, solver="SCS", eps=1e-8)
    return v


def sweep(n_points: int = 81):
    # Avoid theta = 0 (all three measurements collapse) and theta = pi
    # (third measurement aligns with the first up to relabel).
    thetas = np.linspace(0.02 * np.pi, 0.98 * np.pi, n_points)
    etas = np.array([_solve(coplanar_triple(t)) for t in thetas])
    return thetas, etas


# ----------------------------------------------------------------------
# 3.  Plot
# ----------------------------------------------------------------------
def main():
    thetas, etas = sweep(n_points=81)

    trine_theta = 2 * np.pi / 3
    trine_eta   = (1 + np.sqrt(3)) / 3

    # Re-solve precisely AT the trine for comparison
    eta_at_trine = _solve(coplanar_triple(trine_theta))

    # Also re-solve at the +X / +Z / -X degeneracy point theta = pi/2
    eta_at_half  = _solve(coplanar_triple(np.pi / 2))

    print(f"eta*(2 pi / 3)   = {eta_at_trine:.10f}")
    print(f"(1 + sqrt 3)/3   = {trine_eta:.10f}")
    print(f"difference       = {abs(eta_at_trine - trine_eta):.2e}")
    print()
    print(f"eta*(pi / 2)     = {eta_at_half:.10f}   (degenerate +X/+Z/-X case)")
    print(f"(1 + sqrt 2)/3   = {(1 + np.sqrt(2)) / 3:.10f}   (for comparison)")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thetas / np.pi, etas, marker="o", markersize=3, lw=1.2,
            label=r"$\eta^\star(\theta)$ from SDP")
    ax.axvline(2/3, color="C3", ls="--", lw=1,
               label=r"$\theta = 2\pi/3$ (symmetric trine)")
    ax.axhline(trine_eta, color="C3", ls=":", lw=1,
               label=r"$(1+\sqrt{3})/3 \approx 0.9107$")
    ax.axvline(1/2, color="C2", ls="--", lw=0.8, alpha=0.6,
               label=r"$\theta = \pi/2$ (degeneracy)")

    ax.set_xlabel(r"$\theta / \pi$")
    ax.set_ylabel(r"$\eta^\star$")
    ax.set_title("Genuine-triplewise-incompatibility threshold for three "
                 "coplanar qubit\nmeasurements at angles $0,\\ \\theta,\\ 2\\theta$")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(min(etas) - 0.02, 1.02)

    out = "coplanar_trio_sweep.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nfigure saved to: {out}")

    return thetas, etas


if __name__ == "__main__":
    main()
