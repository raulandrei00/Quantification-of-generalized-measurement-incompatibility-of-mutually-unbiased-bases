"""
SDP implementation of equations (10), (11), and (12) from Appendix C of

    Quintino, Budroni, Woodhead, Cabello, Cavalcanti,
    "Device-independent tests of structures of measurement incompatibility"
    arXiv:1902.05841

These SDPs decide / quantify whether a set of three POVMs is **genuinely
triplewise incompatible**, i.e. whether it CANNOT be written as a convex
combination of three sets that are pairwise compatible on the three possible
pairs (1,2), (1,3), (2,3) — Definition 1 in the paper.

Variables (shared across the three SDPs)
----------------------------------------
For each pair (s, t) of measurements in {(0,1), (0,2), (1,2)} (0-indexed),
we introduce a joint POVM
        E^{st} = { E^{st}_{a_s, a_t} : a_s, a_t = 0, ..., n_a - 1 },
all elements PSD.  By construction the marginals
        J^{st}_{a|s} = sum_{a_t} E^{st}_{a, a_t},
        J^{st}_{a|t} = sum_{a_s} E^{st}_{a_s, a}
make the measurements x = s and x = t jointly measurable inside this branch.

For the *third* measurement (the one not in {s, t}) we introduce a free
positive operator family  J^{st}_{a|x_third}  (variables only in eq. (10)/(11);
eliminated in (12) by absorption into a PSD inequality on the noisy M).

The probability p_{st} of the convex combination satisfies
        sum_a J^{st}_{a|x} = p_{st} * I   for every x.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np


# ----------------------------------------------------------------------
# 0.  Utility: build three noisy Pauli measurements (the running example)
# ----------------------------------------------------------------------

def noisy_pauli_measurements(eta: float) -> np.ndarray:
    """
    Three noisy Pauli measurements with white-noise parameter ``eta``:

        M^eta_{a|x} = eta * Pi_{a|x} + (1 - eta) * I/2

    where x = 0, 1, 2  ->  X, Y, Z  and  Pi_{a|x}  are their eigenprojectors.

    Returns an array of shape (3, 2, 2, 2): ``M[x, a]`` is the operator
    M_{a|x} on Hilbert-space dimension d = 2.
    """
    I = np.eye(2, dtype=complex)
    paulis = [
        np.array([[0, 1],  [1, 0]],  dtype=complex),   # X
        np.array([[0, -1j], [1j, 0]], dtype=complex),  # Y
        np.array([[1, 0],  [0, -1]], dtype=complex),   # Z
    ]
    M = np.zeros((3, 2, 2, 2), dtype=complex)
    for x in range(3):
        Pi0 = (I + paulis[x]) / 2
        Pi1 = (I - paulis[x]) / 2
        M[x, 0] = eta * Pi0 + (1 - eta) * I / 2
        M[x, 1] = eta * Pi1 + (1 - eta) * I / 2
    return M


# ----------------------------------------------------------------------
# Internal helper: build the (s, t) pair list and "third" index map
# ----------------------------------------------------------------------

_PAIRS = [(0, 1), (0, 2), (1, 2)]


def _third(pair: tuple[int, int]) -> int:
    """Return the index in {0, 1, 2} that is not in ``pair``."""
    return [k for k in (0, 1, 2) if k not in pair][0]


def _marginal(E_pair, pair_key, x_target, a_target, n_a):
    """Sum-marginal of ``E_pair`` onto ``x_target`` at outcome ``a_target``."""
    i, j = pair_key
    if x_target == i:
        return sum(E_pair[a_target][a_other] for a_other in range(n_a))
    if x_target == j:
        return sum(E_pair[a_other][a_target] for a_other in range(n_a))
    raise ValueError(f"x_target={x_target} is not in pair {pair_key}")


# ----------------------------------------------------------------------
# 1.  Equation (10) — feasibility SDP
# ----------------------------------------------------------------------

def feasibility_eq10(M: np.ndarray, solver: str = "CLARABEL",
                     verbose: bool = False, **solver_kwargs) -> bool:
    """
    Decide whether three POVMs ``M`` are *not* genuinely triplewise
    incompatible (i.e. whether the SDP in eq. (10) is feasible).

    Parameters
    ----------
    M : ndarray, shape (3, n_a, d, d)
        ``M[x, a]`` is the operator M_{a|x}.
    solver, verbose : passed to cvxpy.

    Returns
    -------
    bool
        True  -> SDP feasible        -> NOT genuinely triplewise incompatible.
        False -> SDP infeasible      -> GENUINELY triplewise incompatible.
    """
    _, n_a, d, _ = M.shape
    I = np.eye(d)
    O = np.zeros((d, d))   # use a matrix, not scalar 0, on the RHS of >>

    # Joint POVMs E^{st}: PSD d x d variables, indexed E[(s,t)][a_s][a_t].
    E = {p: [[cp.Variable((d, d), hermitian=True) for _ in range(n_a)]
             for _ in range(n_a)] for p in _PAIRS}
    # Free positive operators for the "third" measurement of each branch.
    J_third = {p: [cp.Variable((d, d), hermitian=True) for _ in range(n_a)]
               for p in _PAIRS}
    # Branch probabilities p_{st} >= 0.
    p = {pair: cp.Variable(nonneg=True) for pair in _PAIRS}

    cons = []

    # 1a. PSD on all variables.
    for pair in _PAIRS:
        for a_s in range(n_a):
            for a_t in range(n_a):
                cons.append(E[pair][a_s][a_t] >> O)
        for a in range(n_a):
            cons.append(J_third[pair][a] >> O)

    # 1b. Normalisation.
    for pair in _PAIRS:
        total_E = sum(E[pair][a_s][a_t]
                      for a_s in range(n_a) for a_t in range(n_a))
        cons.append(total_E == p[pair] * I)
        cons.append(sum(J_third[pair][a] for a in range(n_a)) == p[pair] * I)

    # 1c. M_{a|x} = J^{01}_{a|x} + J^{02}_{a|x} + J^{12}_{a|x}.
    for x in range(3):
        for a in range(n_a):
            terms = []
            for pair in _PAIRS:
                if x in pair:
                    terms.append(_marginal(E[pair], pair, x, a, n_a))
                else:
                    terms.append(J_third[pair][a])
            cons.append(sum(terms) == M[x, a])

    prob = cp.Problem(cp.Minimize(0), cons)
    prob.solve(solver=solver, verbose=verbose, **solver_kwargs)
    return prob.status in ("optimal", "optimal_inaccurate")


# ----------------------------------------------------------------------
# 2.  Equation (11) — white-noise robustness (un-simplified)
# ----------------------------------------------------------------------

def robustness_eq11(M: np.ndarray, solver: str = "CLARABEL",
                    verbose: bool = False, **solver_kwargs) -> tuple[float, str]:
    """
    Maximum eta such that the white-noise mixture

        eta * M_{a|x}  +  (1 - eta) * tr(M_{a|x}) * I/d

    is *not* genuinely triplewise incompatible.  This is equation (11) in
    the paper, with the J^3_{a|x} feasibility constraint of eq. (10) inlined.

    Returns
    -------
    eta_value : float       -- the optimal eta found
    status    : str         -- cvxpy solver status
    """
    _, n_a, d, _ = M.shape
    I = np.eye(d)
    O = np.zeros((d, d))

    E = {p: [[cp.Variable((d, d), hermitian=True) for _ in range(n_a)]
             for _ in range(n_a)] for p in _PAIRS}
    J_third = {p: [cp.Variable((d, d), hermitian=True) for _ in range(n_a)]
               for p in _PAIRS}
    p = {pair: cp.Variable(nonneg=True) for pair in _PAIRS}
    eta = cp.Variable()

    cons = []

    for pair in _PAIRS:
        for a_s in range(n_a):
            for a_t in range(n_a):
                cons.append(E[pair][a_s][a_t] >> O)
        for a in range(n_a):
            cons.append(J_third[pair][a] >> O)

    for pair in _PAIRS:
        total_E = sum(E[pair][a_s][a_t]
                      for a_s in range(n_a) for a_t in range(n_a))
        cons.append(total_E == p[pair] * I)
        cons.append(sum(J_third[pair][a] for a in range(n_a)) == p[pair] * I)

    # sum_of_J == eta * M + (1-eta) * tr(M) * I / d
    for x in range(3):
        for a in range(n_a):
            terms = []
            for pair in _PAIRS:
                if x in pair:
                    terms.append(_marginal(E[pair], pair, x, a, n_a))
                else:
                    terms.append(J_third[pair][a])
            noisy = eta * M[x, a] + (1 - eta) * np.trace(M[x, a]) * I / d
            cons.append(sum(terms) == noisy)

    prob = cp.Problem(cp.Maximize(eta), cons)
    prob.solve(solver=solver, verbose=verbose, **solver_kwargs)
    return float(eta.value), prob.status


# ----------------------------------------------------------------------
# 3.  Equation (12) — simplified white-noise robustness
# ----------------------------------------------------------------------

def robustness_eq12(M: np.ndarray, solver: str = "CLARABEL",
                    verbose: bool = False, **solver_kwargs) -> tuple[float, str]:
    """
    Same problem as (11) but with the ``J_third`` variables eliminated.

    The trick (described between (11) and (12) in the paper):

        noisy_M_{a|x} - J^{sx}_{a|x} - J^{tx}_{a|x}  =  J^{st}_{a|x}  >= 0,

    so we drop the equality and replace it with a PSD inequality.
    Likewise  sum_lambda E^{st}_lambda = (I/d) * sum_lambda tr E^{st}_lambda
    plays the role of  p_{st} I,  and  (1/d) sum_{pairs,lambda} tr E = 1
    enforces p_{12} + p_{13} + p_{23} = 1.
    """
    _, n_a, d, _ = M.shape
    I = np.eye(d)
    O = np.zeros((d, d))

    E = {pair: [[cp.Variable((d, d), hermitian=True) for _ in range(n_a)]
                for _ in range(n_a)] for pair in _PAIRS}
    eta = cp.Variable()

    cons = []

    # E^{st}_lambda >= 0.
    for pair in _PAIRS:
        for a_s in range(n_a):
            for a_t in range(n_a):
                cons.append(E[pair][a_s][a_t] >> O)

    # sum_lambda E^{st}_lambda  ==  (I/d) * sum_lambda tr E^{st}_lambda.
    total_trace = 0
    for pair in _PAIRS:
        sum_E  = sum(E[pair][a_s][a_t]
                     for a_s in range(n_a) for a_t in range(n_a))
        sum_tr = sum(cp.real(cp.trace(E[pair][a_s][a_t]))
                     for a_s in range(n_a) for a_t in range(n_a))
        cons.append(sum_E == (I / d) * sum_tr)
        total_trace = total_trace + sum_tr

    # Branch-probabilities sum to one:  (1/d) * total trace == 1.
    cons.append(total_trace / d == 1)

    # PSD inequality: noisy_M_{a|x}  >>  J^{sx}_{a|x} + J^{tx}_{a|x}.
    # The three (s, t, x) triples cover the three pairs containing x.
    for (s, t, x) in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
        pair_sx = tuple(sorted((s, x)))
        pair_tx = tuple(sorted((t, x)))
        for a in range(n_a):
            J_sx = _marginal(E[pair_sx], pair_sx, x, a, n_a)
            J_tx = _marginal(E[pair_tx], pair_tx, x, a, n_a)
            noisy = eta * M[x, a] + (1 - eta) * np.trace(M[x, a]) * I / d
            cons.append(noisy >> J_sx + J_tx)

    prob = cp.Problem(cp.Maximize(eta), cons)
    prob.solve(solver=solver, verbose=verbose, **solver_kwargs)
    return float(eta.value), prob.status


# ----------------------------------------------------------------------
# 4.  Demo / sanity checks
# ----------------------------------------------------------------------

def _demo():
    print("=" * 70)
    print("SDP tests of genuine triplewise incompatibility")
    print("arXiv:1902.05841, Appendix C, equations (10) / (11) / (12)")
    print("=" * 70)

    # Theoretical thresholds for noisy Paulis  M^eta_{a|x} = eta * Pi + (1-eta) I/2
    eta_3wise   = 1 / np.sqrt(3)            # triplewise compatible up to here
    eta_pair    = 1 / np.sqrt(2)            # pairwise compatible up to here
    eta_genuine = (np.sqrt(2) + 1) / 3      # genuine-triplewise threshold

    print("\nNoisy-Pauli thresholds:")
    print(f"  triplewise compat.:        eta <= 1/sqrt(3)        ~= {eta_3wise:.4f}")
    print(f"  pairwise compat.:          eta <= 1/sqrt(2)        ~= {eta_pair:.4f}")
    print(f"  not genuinely 3-wise inc.: eta <= (sqrt(2)+1)/3    ~= {eta_genuine:.4f}")

    # ---- Feasibility, eq. (10) ----
    print("\n--- Feasibility check via equation (10) ---")
    for e in [0.50, 0.70, 0.80, eta_genuine, 0.81, 0.85, 1.00]:
        ok = feasibility_eq10(noisy_pauli_measurements(e))
        tag = "feasible    -> NOT genuinely 3-wise incompatible" if ok \
              else "infeasible  -> GENUINELY 3-wise incompatible"
        print(f"  eta = {e:.4f}  :  {tag}")

    # ---- Robustness, eq. (12) ----
    print("\n--- White-noise robustness via equation (12) ---")
    print("  (eta*  scales the measurement; on noisy Paulis the threshold is")
    print("   eta * eta_input <= (sqrt(2)+1)/3,  so we expect eta* = "
          "(sqrt(2)+1)/(3 * eta_input).)")
    for e in [0.5, 0.7, 0.8, eta_genuine, 0.85, 1.0]:
        val, _ = robustness_eq12(noisy_pauli_measurements(e))
        print(f"  eta_input = {e:.4f}  :  eta* = {val:.4f}   "
              f"(expected ~ {eta_genuine / e:.4f})")

    # ---- Cross-check (11) vs (12) on sharp Paulis ----
    print("\n--- Cross-check (11) vs (12) on sharp Paulis (eta_input = 1) ---")
    Msharp = noisy_pauli_measurements(1.0)
    v11, _ = robustness_eq11(Msharp)
    v12, _ = robustness_eq12(Msharp)
    print(f"  equation (11):  eta* = {v11:.6f}")
    print(f"  equation (12):  eta* = {v12:.6f}")
    print(f"  expected      :  eta* = {eta_genuine:.6f}   "
          f"= (sqrt(2)+1)/3")


if __name__ == "__main__":
    M = noisy_pauli_measurements(1.0)  
    print ("(11) robustness: ", robustness_eq11(M))
    print ("(12) robustness: ", robustness_eq12(M))
