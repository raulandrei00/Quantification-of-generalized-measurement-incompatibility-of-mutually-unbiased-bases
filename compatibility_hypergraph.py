"""
General hypergraph-parameterised incompatibility SDP.

For a compatibility hypergraph C = [C_1, ..., C_N], we ask:
  "Can M_{a|x} be written as a convex combination of N branches, where
   branch i has the measurements indexed by C_i jointly measurable, and
   the measurements outside C_i are arbitrary POVMs?"

Special cases:
  C = [(0, 1, ..., m-1)]                          -> full m-wise joint measurability
                                                     (matches solve_joint_measurement_general)
  C = [(0,1), (0,2), (1,2)]                       -> "not genuinely triplewise incompatible"
                                                     (matches eq (10) of arXiv:1902.05841)
  C = [all size-k subsets of {0,...,m-1}]          -> "not genuinely k+1-wise incompatible"
"""

from __future__ import annotations

import itertools
import cvxpy as cp
import numpy as np


# ----------------------------------------------------------------------
# Core SDP
# ----------------------------------------------------------------------

def _tree_sum(terms):
    """Balanced binary-tree reduction of a list of CVXPY expressions.

    Equivalent to ``sum(terms)`` but produces an expression of depth
    O(log N) rather than O(N). CVXPY canonicalises the tree by walking
    it; a depth-N chain triggers the ``too many subexpressions`` warning
    and scales super-linearly with N for large hypergraphs (e.g.
    d^k > a few hundred). The balanced tree keeps each constraint
    shallow.
    """
    if not terms:
        return 0
    while len(terms) > 1:
        terms = [terms[i] + terms[i + 1] if i + 1 < len(terms) else terms[i]
                 for i in range(0, len(terms), 2)]
    return terms[0]


def _marginal_E(E_edge, edge, x_target, a_target):
    """Sum-marginal of joint POVM E^C indexed by tuples lambda in outcomes^|C|."""
    pos = edge.index(x_target)
    return _tree_sum([E_edge[lam] for lam in E_edge if lam[pos] == a_target])


def _build_variables(M, hypergraph):
    """E[edge_idx][lam], J_outside[edge_idx][x][a], p[edge_idx] for every branch."""
    m, n_a, d, _ = M.shape
    E, J_outside, p = {}, {}, {}
    for i, edge in enumerate(hypergraph):
        lambdas = list(itertools.product(range(n_a), repeat=len(edge)))
        E[i] = {lam: cp.Variable((d, d), hermitian=True) for lam in lambdas}
        J_outside[i] = {x: [cp.Variable((d, d), hermitian=True) for _ in range(n_a)]
                        for x in range(m) if x not in edge}
        p[i] = cp.Variable(nonneg=True)
    return E, J_outside, p


def _base_constraints(M, hypergraph, E, J_outside, p):
    """Positivity, branch normalisation. Returns the list of constraints."""
    m, n_a, d, _ = M.shape
    I = np.eye(d)
    Z = np.zeros((d, d))
    cons = []
    # Positivity.
    for i in range(len(hypergraph)):
        for E_lam in E[i].values():
            cons.append(E_lam >> Z)
        for J_list in J_outside[i].values():
            for J in J_list:
                cons.append(J >> Z)
    # Branch normalisation: every "POVM-like" object in branch i sums to p[i] * I.
    # Use balanced tree-sums so the constraint expression stays O(log N) deep
    # rather than O(N) -- this is what the CVXPY "too many subexpressions"
    # warning was complaining about.
    for i, edge in enumerate(hypergraph):
        cons.append(_tree_sum(list(E[i].values())) == p[i] * I)
        for J_list in J_outside[i].values():
            cons.append(_tree_sum(J_list) == p[i] * I)
    return cons


def _glue_terms(x, a, hypergraph, E, J_outside):
    """For (x, a), build the list of contributions from each branch to M[x, a]."""
    terms = []
    for i, edge in enumerate(hypergraph):
        if x in edge:
            terms.append(_marginal_E(E[i], edge, x, a))
        else:
            terms.append(J_outside[i][x][a])
    return terms


def feasibility_hypergraph(M, hypergraph, *,
                           solver="CLARABEL", **solver_kwargs) -> bool:
    """
    True iff M admits a decomposition along the given compatibility hypergraph.

    With hypergraph = [(0,...,m-1)]  this is full m-wise joint measurability
    (your solve_joint_measurement_general).

    With hypergraph = [(0,1), (0,2), (1,2)] and m = 3, this is feasibility of
    eq. (10) -- "not genuinely triplewise incompatible".
    """
    m, n_a, d, _ = M.shape
    hypergraph = [tuple(sorted(edge)) for edge in hypergraph]
    E, J_outside, p = _build_variables(M, hypergraph)
    cons = _base_constraints(M, hypergraph, E, J_outside, p)
    # Glue: sum over branches must equal M.
    for x in range(m):
        for a in range(n_a):
            cons.append(_tree_sum(_glue_terms(x, a, hypergraph, E, J_outside)) == M[x, a])
    prob = cp.Problem(cp.Minimize(0), cons)
    prob.solve(solver=solver, **solver_kwargs)
    return prob.status in ("optimal", "optimal_inaccurate")


def robustness_hypergraph(M, hypergraph, *,
                          solver="CLARABEL", **solver_kwargs) -> tuple[float, str]:
    """
    Max eta such that the noisy version of M admits a decomposition.

    Returns (eta_star, solver_status).
    """
    m, n_a, d, _ = M.shape
    I = np.eye(d)
    hypergraph = [tuple(sorted(edge)) for edge in hypergraph] # (1,2)(2,3)(3,4)(4,1),(4,2)(1,3)
    E, J_outside, p = _build_variables(M, hypergraph)
    cons = _base_constraints(M, hypergraph, E, J_outside, p)
    eta = cp.Variable()
    # Same gluing as feasibility, but with eta-noisy RHS.
    for x in range(m):
        for a in range(n_a):
            noisy = eta * M[x, a] + (1 - eta) * np.trace(M[x, a]) * I / d
            cons.append(_tree_sum(_glue_terms(x, a, hypergraph, E, J_outside)) == noisy)
    prob = cp.Problem(cp.Maximize(eta), cons)
    prob.solve(solver=solver, **solver_kwargs)
    return float(eta.value), prob.status


# ----------------------------------------------------------------------
# Demo: walk up the lattice of compatibility hypergraphs
# ----------------------------------------------------------------------

from incompatibility_sdp import noisy_pauli_measurements
def _demo():

    print("=" * 74)
    print(" Climbing the lattice of compatibility hypergraphs for noisy Paulis")
    print("=" * 74)

    M = noisy_pauli_measurements(1.0)  # sharp X, Y, Z

    cases = [
        ("Singletons  C = [(0,), (1,), (2,)]   (trivially decomposable)",
         [(0,), (1,), (2,)], None),
        ("Two pairs   C = [(0,1), (0,2)]      (asymmetric: edge (1,2) missing)",
         [(0, 1), (0, 2)], None),
        ("All pairs   C = [(0,1),(0,2),(1,2)] (eq (10): not genuinely 3-wise inc.)",
         [(0, 1), (0, 2), (1, 2)], (1 + np.sqrt(2)) / 3),
        ("Full edge   C = [(0,1,2)]           (full joint measurability)",
         [(0, 1, 2)], 1 / np.sqrt(3)),
    ]

    print()
    for label, hg, expected in cases:
        try:
            eta_star, st = robustness_hypergraph(M, hg)
        except Exception:
            eta_star, st = robustness_hypergraph(M, hg, solver="SCS", eps=1e-8)
        line = f"  {label:<70} eta* = {eta_star:.6f}"
        if expected is not None:
            line += f"   (expected {expected:.6f})"
        print(line)

    print()
    print("Note: eta* is monotone non-increasing as we ADD edges to the hypergraph.")
    print("Adding the big edge (0,1,2) tightens the constraints the most;")
    print("breaking it into pairs is more permissive (higher eta*).")

from mub_joint_measurability import mubs_d2, mubs_d3, mubs_d4, noisy_mubs_d3

def hypergraph_compatible(k):
    """Compatibility hypergraph for the given number of POVMs."""
    hypergraph = [tuple(range(k))]
    return hypergraph

def hyperg_pair(k):
    """Compatibility hypergraph with all pairs of POVMs."""
    hypergraph = [tuple(edge) for edge in itertools.combinations(range(k), 2)]
    return hypergraph

if __name__ == "__main__":
    M = noisy_mubs_d3(1.0)            # 4 sharp MUBs, shape (4, 3, 3, 3)
    print("Robustness for compatibility on 4 MUBs (d=3):")
    try:
        eta_star, st = robustness_hypergraph(M, hypergraph_compatible(4))
    except Exception:
        eta_star, st = robustness_hypergraph(M, hypergraph_compatible(4), solver="SCS", eps=1e-8)
    print(f"  eta* = {eta_star:.6f}   [{st}]")

    print("Robustness for pairwise hypergraph on 4 MUBs (d=3):")
    try:        eta_star, st = robustness_hypergraph(M, hyperg_pair)
    except Exception:
        eta_star, st = robustness_hypergraph(M, hyperg_pair, solver="SCS", eps=1e-8)
    print(f"  eta* = {eta_star:.6f}   [{st}]")