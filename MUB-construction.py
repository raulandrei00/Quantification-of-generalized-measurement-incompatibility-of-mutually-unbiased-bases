
import numpy as np
import cvxpy as cp
import itertools, math




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

