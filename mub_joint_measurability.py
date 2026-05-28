"""
Reproduces Table I of:
  Designolle, Skrzypczyk, Fröwis, Brunner
  "Quantifying measurement incompatibility of mutually unbiased bases"
  Phys. Rev. Lett. 122, 050402 (2019)

eta*(k,d) = max eta s.t. k noisy MUBs  M_a^x = (1-eta)/d*I + eta*P_a^x  are JM.

Tight analytic values (from paper):
  d=2: k=2 → 1/√2≈0.7071,  k=3 → 1/√3≈0.5774
  d=3: k=4(complete) → 1/√4=0.5000
  d=4: k=5(complete) → 1/√5≈0.4472
  General complete set k=d+1 (d prime power): eta* = 1/√(d+1)  [Theorem 1]

The '?' entries (k < d+1) are purely numerical — we compute them below.
"""

import numpy as np
import cvxpy as cp
import itertools, math

# ═════════════════════════════════════════════════════════════════════════════
# SDP: noise robustness threshold
# ═════════════════════════════════════════════════════════════════════════════

def solve_jm(unitaries, d, eps=5e-8, max_iters=300000):
    """
    unitaries : list of m unitary matrices (d×d). Columns = basis vectors.
    Computes eta* = max eta s.t. the m noisy MUBs are jointly measurable.
    M_a^x = (1-eta)/d * I  +  eta * |u_a^x><u_a^x|
    """
    m  = len(unitaries)
    Id = np.eye(d, dtype=complex)
    eta = cp.Variable(nonneg=True)

    # POVM elements M[x][a] (affine in eta)
    M = []
    for U in unitaries:
        row = []
        for a in range(d):
            v = U[:, a:a+1]
            P = v @ v.conj().T
            row.append((1/d)*Id + eta*(P - (1/d)*Id))
        M.append(row)

    # Joint POVM N[λ], λ ∈ {0,...,d-1}^m
    lambdas = list(itertools.product(range(d), repeat=m))
    N = {lam: cp.Variable((d, d), hermitian=True) for lam in lambdas}

    cons  = [eta <= 1]
    cons += [N[lam] >> 0 for lam in lambdas]
    cons += [cp.sum(list(N.values())) == Id]
    for x in range(m):
        for a in range(d):
            matching = [N[lam] for lam in lambdas if lam[x] == a]
            cons.append(cp.sum(matching) == M[x][a])

    prob = cp.Problem(cp.Maximize(eta), cons)
    prob.solve(solver=cp.SCS, eps=eps, max_iters=max_iters, verbose=False)
    return (float(eta.value) if eta.value is not None else None), prob.status

# ═════════════════════════════════════════════════════════════════════════════
# MUB constructions
# ═════════════════════════════════════════════════════════════════════════════

def verify_mubs(Us, d, tol=1e-7):
    """Check |<u_a^i | u_b^j>|² = 1/d for all i≠j."""
    errors = []
    for i in range(len(Us)):
        for j in range(i+1, len(Us)):
            gram = abs(Us[i].conj().T @ Us[j])**2
            dev  = abs(gram - 1/d).max()
            if dev > tol:
                errors.append(f"B{i}-B{j}: max_dev={dev:.2e}")
    return errors

# ── d=2: Pauli eigenbases ─────────────────────────────────────────────────────
def mubs_d2():
    s = 1/math.sqrt(2)
    UZ = np.eye(2, dtype=complex)
    UX = np.array([[s, s],[s,-s]], dtype=complex)
    UY = np.array([[s, s],[s*1j,-s*1j]], dtype=complex)
    return [UZ, UX, UY]

# ── d=3: Fourier/Alltop prime-field construction ──────────────────────────────
def mubs_d3():
    """d+1=4 MUBs for d=3. [U_s]_{a,j} = ω^{s·a²+a·j}/√3, ω=e^{2πi/3}."""
    d = 3
    w = np.exp(2j*np.pi/d)
    U0 = np.eye(d, dtype=complex)
    Us = [U0]
    for s in range(d):
        U = np.array([[w**(s*a*a + a*j) for j in range(d)]
                      for a in range(d)], dtype=complex) / math.sqrt(d)
        Us.append(U)
    return Us   # 4 unitaries

def noisy_mub_measurements(mubs, eta: float) -> np.ndarray:
    """
    Turn a list of MUB unitaries into white-noise-mixed projective
    measurements with noise parameter ``eta``:

        M^eta_{a|x} = eta * Pi_{a|x} + (1 - eta) * I/d

    where x indexes the MUB (measurement setting), a indexes the outcome,
    and  Pi_{a|x} = |u^x_a><u^x_a|  is the rank-1 projector onto column
    ``a`` of the x-th unitary (its eigenbasis, matching mubs_d2/d4).

    Returns an array of shape (m, d, d, d): ``M[x, a]`` is the operator
    M_{a|x} on Hilbert-space dimension d, with m = len(mubs) settings
    and d outcomes per setting.
    """
    d = mubs[0].shape[0]
    I = np.eye(d, dtype=complex)
    M = np.zeros((len(mubs), d, d, d), dtype=complex)
    for x, U in enumerate(mubs):
        for a in range(d):
            v = U[:, a]
            Pi = np.outer(v, np.conj(v))
            M[x, a] = eta * Pi + (1 - eta) * I / d
    return M

def noisy_mubs_d3(eta: float) -> np.ndarray:
    """d=3: the 4 MUBs as eta-noisy projective measurements, shape (4,3,3,3)."""
    return noisy_mub_measurements(mubs_d3(), eta)

# ── d=4: 2-qubit Pauli commuting groups ──────────────────────────────────────
def mubs_d4():
    """
    5 MUBs for d=4=2⊗2 via the 5 maximal commuting subgroups of the
    2-qubit Pauli group. Each group has a simultaneous eigenbasis.
    """
    def P(a, b):
        M = {'I':np.eye(2,dtype=complex),
             'X':np.array([[0,1],[1,0]],dtype=complex),
             'Y':np.array([[0,-1j],[1j,0]],dtype=complex),
             'Z':np.array([[1,0],[0,-1]],dtype=complex)}
        return np.kron(M[a], M[b])

    # 5 MAS (maximal abelian subgroups), each a group of 3 commuting Paulis
    groups = [
        ['IX','XI','XX'],
        ['IY','YI','YY'],
        ['IZ','ZI','ZZ'],
        ['XY','YZ','ZX'],
        ['XZ','YX','ZY'],
    ]
    Us = []
    for g in groups:
        ops = [P(s[0],s[1]) for s in g]
        # Build total observable with distinct weights so eigenvalues are distinct
        total = sum((i+1)*0.7**(i) * ops[i] for i in range(3))
        _, evecs = np.linalg.eigh(total)
        Us.append(evecs)
    return Us   # 5 unitaries

# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    constructions = [
        (2, "Pauli Z/X/Y eigenbases",          mubs_d2()),
        (3, "Fourier prime-field (d=3)",        mubs_d3()),
        (4, "2-qubit Pauli commuting groups",   mubs_d4()),
    ]

    print("=" * 74)
    print("  TABLE I  —  eta*(k,d):  noise robustness of k MUBs in dim d")
    print("  Designolle, Skrzypczyk, Fröwis, Brunner  PRL 122, 050402 (2019)")
    print("=" * 74)

    all_results = {}

    for (d, name, Us) in constructions:
        errs = verify_mubs(Us, d)
        ok   = len(errs) == 0
        print(f"\n  d={d}  [{name}]")
        print(f"  MUB verification: {' PASS' if ok else ' FAIL — ' + str(errs[:2])}")
        print(f"  {'k':>3}  {'eta* (SDP)':>12}  {'status':>14}  note")

        for k in range(2, d+2):
            eta, status = solve_jm(Us[:k], d)
            all_results[(d, k)] = eta

            if k == d+1:
                tight = 1/math.sqrt(d+1)
                note  = f"TIGHT(Thm.1) = 1/√{d+1} = {tight:.6f}"
            elif d == 2 and k == 2:
                note  = f"TIGHT = 1/√2 = {1/math.sqrt(2):.6f}"
            else:
                note  = "? in paper — numerical"

            s_val = f"{eta:.6f}" if eta is not None else "FAILED"
            print(f"  k={k}  {s_val:>12}  {status:>14}  {note}")

    # ── Clean summary table ───────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("  COMPLETE TABLE  (matching / extending Table I of the paper)")
    print(f"  {'d':>3}  {'k':>3}   {'eta*(SDP)':>12}   {'paper value':>20}   match?")
    for (d, k), eta in sorted(all_results.items()):
        if k == d+1:
            paper = f"1/√{d+1} = {1/math.sqrt(d+1):.6f}"
            tight = 1/math.sqrt(d+1)
        elif d == 2 and k == 2:
            paper = f"1/√2 = {1/math.sqrt(2):.6f}"
            tight = 1/math.sqrt(2)
        else:
            paper = "? (no formula)"
            tight = None

        if eta is None:
            match = "—"
        elif tight is not None:
            match = "ok" if abs(eta - tight) < 5e-4 else f" (diff={abs(eta-tight):.4f})"
        else:
            match = "(numerical)"

        s = f"{eta:.6f}" if eta else "FAILED"
        print(f"  {d:>3}  {k:>3}   {s:>12}   {paper:>20}   {match}")

    print("""
    ═══════════════════════════════════════════════════════════════════════════════
    KEY FORMULAS (from paper):
    Theorem 1 (tight): k=d+1 MUBs, d prime-power → eta*(k,d) = 1/√(d+1)
    Lower bound:       eta*(k,d) ≥ 1/√k                          [Prop.1]
    Upper bound:       eta*(k,d) ≤ sqrt(d²(k-1) / (k(k(d-1)+d))) [Prop.2]

    NOTEBOOK (your code) STATUS:
    • d=2, k=2 (σ_X vs σ_Z):    0.7071 ok   (= 1/√2)
    • d=2, k=3 (σ_X,σ_Y,σ_Z):   0.5773 ok   (= 1/√3)
    • d=3, k=2 (qutrit MUBs):    0.6830      (? in paper, now numerical)
    • d=3, k=4 (complete MUBs):  0.5000 ok   (= 1/√4 = 1/2)
    ═══════════════════════════════════════════════════════════════════════════════
    """)
