"""
Python translations of the MATLAB functions JMPOVMs.m and MUBMeasurements.m.

JMPOVMs:
    Skrzypczyk and Cavalcanti, March 2016.
    Decide whether a set of POVMs is jointly measurable and return the
    parent POVM Glam when it is.

MUBMeasurements:
    Tendick, Kliesch, Kampermann, Bruss --- "Distance-based resource
    quantification for sets of quantum measurements", arXiv:2205.08546,
    code by Lucas Tendick.
    Generate the d+1 mutually-unbiased bases in prime dimension d using
    QETLAB's GenPauli (X = generalized shift, Z = generalized clock).

MATLAB-layout convention is preserved:
    M has shape  (d, d, oa, ma)
    with         M[:, :, a, x]  ==  M_{a|x}.
(This is DIFFERENT from the (m, oa, d, d) convention used elsewhere in
this project.)  Outcomes  a in 0..oa-1,  settings x in 0..ma-1.

JMPOVMs in MATLAB can also act as a "partially specified problem", i.e.
be dropped inside a CVX block as `JMPOVMs(Max) == 1` to enforce joint
measurability of a CVX variable Max.  In cvxpy you build constraints
yourself --- the helper `jm_constraints` below returns that constraint
list explicitly for use inside a larger cvxpy problem.
"""

from __future__ import annotations

import itertools
import cvxpy as cp
import numpy as np


# ======================================================================
#  GenPauli  -- QETLAB's generalized Pauli  X^j Z^k  in dimension d
# ======================================================================
def gen_pauli(j: int, k: int, d: int) -> np.ndarray:
    """
    QETLAB-style generalized Pauli  U = X^j Z^k  in dimension d, where
        X|n>  =  |(n+1) mod d>     (shift)
        Z|n>  =  omega^n |n>,  omega = exp(2 pi i / d)   (clock).

    Matrix form:  (X^j Z^k)[m, n] = omega^{n k} * delta_{m, (n+j) mod d}.
    """
    omega = np.exp(2j * np.pi / d)
    U = np.zeros((d, d), dtype=complex)
    for n in range(d):
        U[(n + j) % d, n] = omega ** (n * k)
    return U


# ======================================================================
#  MUBMeasurements
# ======================================================================
def mub_measurements(p: int, m: int) -> np.ndarray:
    """
    Generate the first ``m`` MUB measurements in prime dimension ``p``.

    The MUBs are the eigenbases of  X, Z, X Z, X Z^2, ..., X Z^{d-1}
    where X = gen_pauli(1,0,d) and Z = gen_pauli(0,1,d).  In prime
    dimension d these d+1 bases are pairwise unbiased (Wootters-Fields).

    Parameters
    ----------
    p : int
        Prime dimension.  Number of outcomes is also p.
    m : int
        Number of measurement settings, 1 <= m <= p+1.

    Returns
    -------
    M : ndarray, complex, shape (p, p, p, m)
        Rank-1 projective POVMs.  M[:, :, a, x] is the projector onto
        eigenvector ``a`` of the x-th MUB-generating operator.
    """
    if not (1 <= m <= p + 1):
        raise ValueError(f"m must satisfy 1 <= m <= p+1; got m={m}, p={p}")

    d = p
    o = d

    X = gen_pauli(1, 0, d)
    Z = gen_pauli(0, 1, d)
    # Operators whose eigenbases are the d+1 MUBs.
    ops = [X, Z] + [X @ np.linalg.matrix_power(Z, n) for n in range(1, d)]

    M = np.zeros((d, d, o, m), dtype=complex)
    for x in range(m):
        _, evecs = np.linalg.eig(ops[x])
        for a in range(o):
            v = evecs[:, a]
            v = v / np.linalg.norm(v)         # safety renormalisation
            M[:, :, a, x] = np.outer(v, v.conj())
    return M


# ======================================================================
#  Helpers
# ======================================================================
def _gen_single_party_array(oa: int, ma: int) -> np.ndarray:
    """
    MATLAB's ``genSinglePartyArray(oa, ma)``: an  (oa, ma, oa**ma)  array
    S where  S[a, x, k] = 1  iff the k-th deterministic strategy assigns
    outcome ``a`` to setting ``x``, and 0 otherwise.

    Strategies are enumerated as k <-> (a_0, a_1, ..., a_{ma-1}) in
    lexicographic order via itertools.product.
    """
    n_det = oa ** ma
    S = np.zeros((oa, ma, n_det), dtype=float)
    for k, lam in enumerate(itertools.product(range(oa), repeat=ma)):
        for x in range(ma):
            S[lam[x], x, k] = 1.0
    return S


def valid_povms(M: np.ndarray, tol: float = 1e-7) -> bool:
    """Return True iff each M_{a|x} is Hermitian PSD and  sum_a M_{a|x} = I."""
    d, d2, oa, ma = M.shape
    if d != d2:
        return False
    I = np.eye(d)
    for x in range(ma):
        s = np.zeros((d, d), dtype=complex)
        for a in range(oa):
            E = M[:, :, a, x]
            if not np.allclose(E, E.conj().T, atol=tol):
                return False
            if np.linalg.eigvalsh((E + E.conj().T) / 2).min() < -tol:
                return False
            s = s + E
        if not np.allclose(s, I, atol=tol):
            return False
    return True


# ======================================================================
#  JMPOVMs --- standalone solver
# ======================================================================
def jm_povms(M: np.ndarray, solver: str = "CLARABEL",
             verbose: bool = False, **solver_kwargs):
    """
    Decide whether the POVMs ``M`` are jointly measurable; return the
    parent POVM Glam if so.

    Parameters
    ----------
    M : ndarray, shape (d, d, oa, ma)
        POVM elements with  M[:, :, a, x] = M_{a|x}.
    solver, verbose, solver_kwargs : forwarded to cvxpy ``prob.solve``.

    Returns
    -------
    are_JM_POVMs : int  (1 if jointly measurable, 0 otherwise)
    Glam         : ndarray of shape (d, d, oa**ma) if JM, else None.
                   Glam[:, :, k] is the parent-POVM element of the k-th
                   deterministic strategy (lex-ordered, same convention
                   as _gen_single_party_array).
    """
    if not valid_povms(M):
        raise ValueError("POVMs are not valid")

    d, _, oa, ma = M.shape
    n_det = oa ** ma
    S = _gen_single_party_array(oa, ma)
    Z = np.zeros((d, d))

    Glam = [cp.Variable((d, d), hermitian=True) for _ in range(n_det)]
    cons = [G >> Z for G in Glam]                              # G_lam >= 0
    cons.append(sum(Glam) == np.eye(d))                        # sum_lam G_lam = I
    for x in range(ma):                                        # marginals
        for a in range(oa):
            ks = [k for k in range(n_det) if S[a, x, k] > 0.5]
            cons.append(sum(Glam[k] for k in ks) == M[:, :, a, x])

    prob = cp.Problem(cp.Minimize(0), cons)
    try:
        prob.solve(solver=solver, verbose=verbose, **solver_kwargs)
    except cp.error.SolverError:
        # Large MUB cases (3^4 PSD vars) sometimes break CLARABEL; SCS handles
        # them at tight tolerance.
        prob.solve(solver="SCS", eps=1e-9, max_iters=200000, verbose=verbose)

    if prob.status in ("optimal", "optimal_inaccurate"):
        Glam_val = np.stack([G.value for G in Glam], axis=2)
        return 1, Glam_val
    return 0, None


# ======================================================================
#  jm_constraints --- "inside CVX" usage
# ======================================================================
def jm_constraints(M_cvx, oa: int, ma: int, d: int):
    """
    Return the list of cvxpy constraints expressing "the POVMs M are
    jointly measurable", plus the parent-POVM variables.  Drop the
    constraints into a larger cvxpy problem to enforce JM as a side
    condition on a cvxpy-variable measurement set.

    ``M_cvx`` may be a numpy array of shape (d, d, oa, ma) OR a 2-D
    Python list M_cvx[a][x] of cvxpy expressions of shape (d, d).

    Returns
    -------
    constraints : list of cvxpy constraints
    Glam        : list of cvxpy variables (the parent POVM, length oa**ma)
    """
    n_det = oa ** ma
    S = _gen_single_party_array(oa, ma)
    Z = np.zeros((d, d))
    Glam = [cp.Variable((d, d), hermitian=True) for _ in range(n_det)]

    def Max(a, x):
        if isinstance(M_cvx, np.ndarray):
            return M_cvx[:, :, a, x]
        return M_cvx[a][x]

    cons = [G >> Z for G in Glam]
    cons.append(sum(Glam) == np.eye(d))
    for x in range(ma):
        for a in range(oa):
            ks = [k for k in range(n_det) if S[a, x, k] > 0.5]
            cons.append(sum(Glam[k] for k in ks) == Max(a, x))
    return cons, Glam


# ======================================================================
#  Demo / sanity check
# ======================================================================
def _noisy(M: np.ndarray, eta: float) -> np.ndarray:
    """Add white noise:  M_{a|x} -> eta M_{a|x} + (1-eta) tr(M_{a|x}) I/d."""
    d, _, oa, ma = M.shape
    out = np.zeros_like(M)
    for x in range(ma):
        for a in range(oa):
            E = M[:, :, a, x]
            out[:, :, a, x] = eta * E + (1 - eta) * np.trace(E) * np.eye(d) / d
    return out


# ======================================================================
#  Robustness wrapper:  max eta such that white-noise mixture is JM
# ======================================================================
def jm_povms_robustness(M: np.ndarray, solver: str = "CLARABEL",
                        verbose: bool = False, **solver_kwargs):
    """
    Maximum eta such that  eta * M_{a|x} + (1-eta) * tr(M_{a|x}) * I/d
    is jointly measurable.  Returns (eta_star, status).
    """
    d, _, oa, ma = M.shape
    n_det = oa ** ma
    S = _gen_single_party_array(oa, ma)
    Z = np.zeros((d, d))
    I = np.eye(d)

    Glam = [cp.Variable((d, d), hermitian=True) for _ in range(n_det)]
    eta = cp.Variable()

    cons = [G >> Z for G in Glam]
    cons.append(sum(Glam) == I)
    for x in range(ma):
        for a in range(oa):
            ks = [k for k in range(n_det) if S[a, x, k] > 0.5]
            noisy = eta * M[:, :, a, x] + (1 - eta) * np.trace(M[:, :, a, x]) * I / d
            cons.append(sum(Glam[k] for k in ks) == noisy)

    prob = cp.Problem(cp.Maximize(eta), cons)
    try:
        prob.solve(solver=solver, verbose=verbose, **solver_kwargs)
    except cp.error.SolverError:
        prob.solve(solver="SCS", eps=1e-9, max_iters=200000, verbose=verbose)
    return float(eta.value), prob.status


# ======================================================================
#  Demo / sanity check
# ======================================================================
def _noisy(M: np.ndarray, eta: float) -> np.ndarray:
    """White-noise mixture for the demo only."""
    d, _, oa, ma = M.shape
    out = np.zeros_like(M)
    for x in range(ma):
        for a in range(oa):
            E = M[:, :, a, x]
            out[:, :, a, x] = eta * E + (1 - eta) * np.trace(E) * np.eye(d) / d
    return out


if __name__ == "__main__":
    print("=" * 68)
    print("  Translation sanity check: MUBMeasurements + JMPOVMs")
    print("=" * 68)

    # Known JM thresholds for noisy MUBs.
    cases = [
        ("2 MUBs in d=2", 2, 2, 1 / np.sqrt(2),       "1/sqrt(2)"),
        ("3 MUBs in d=2", 2, 3, 1 / np.sqrt(3),       "1/sqrt(3)"),
        ("2 MUBs in d=3", 3, 2, (1 + np.sqrt(3))/4,   "(1+sqrt(3))/4"),
        ("3 MUBs in d=3", 3, 3, None,                  "--"),
        ("4 MUBs in d=3", 3, 4, (1 + 3*np.sqrt(5))/16, "(1+3*sqrt(5))/16"),
        ("2 MUBs in d=5", 5, 2, (3 + np.sqrt(5))/8,    "(3+sqrt(5))/8"),
    ]
    print(f"\n{'config':<18} {'eta* (SDP)':<14} {'closed form':<18} {'expected':<12}")
    print("-" * 64)
    for label, p, m, expected, exp_name in cases:
        M = mub_measurements(p, m)
        eta_star, _ = jm_povms_robustness(M)
        exp_str = f"{expected:.6f}" if expected is not None else ""
        print(f"   {label:<15} {eta_star:.6f}      {exp_name:<18} {exp_str}")
