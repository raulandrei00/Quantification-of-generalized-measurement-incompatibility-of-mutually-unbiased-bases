import cvxpy as cp
import numpy as np

# ── MUB projectors for qubit (d=2) ──────────────────────────────────────────
# Z basis
P0 = np.array([[1, 0], [0, 0]], dtype=complex)   # |0><0|
P1 = np.array([[0, 0], [0, 1]], dtype=complex)   # |1><1|

# X basis
Pp = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)   # |+><+|
Pm = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=complex) # |-><-|

# Noisy MUBs: M_noisy = (1-t)*M + t*(I/d)
# I/d for d=2
I2 = np.eye(2) / 2

d = 2
t = cp.Variable(nonneg=True)  # noise parameter to minimize

# Noisy POVM elements (affine in t)
M = [
    (1 - t) * P0 + t * I2,
    (1 - t) * P1 + t * I2,
    (1 - t) * Pp + t * I2,
    (1 - t) * Pm + t * I2,
]

# ── Joint POVM variable ──────────────────────────────────────────────────────
# 4 outcomes (2x2 outcomes for 2 measurements), each a 2x2 PSD matrix
# lambda indexes joint outcomes (a,b) in {0,1}x{0,1}
G = [cp.Variable((d, d), hermitian=True) for _ in range(4)]

# ── Constraints ──────────────────────────────────────────────────────────────
constraints = []

# 1. Each G_lambda is PSD
for Gl in G:
    constraints.append(Gl >> 0)

# 2. Normalization: sum of all G_lambda = I
constraints.append(sum(G) == np.eye(d))

# 3. Marginalization: noisy MUBs are marginals of G
# Z basis: M[0] = G[0,0] + G[0,1], M[1] = G[1,0] + G[1,1]
constraints.append(M[0] == G[0] + G[1])  # outcome 0 in Z, marginalize over X
constraints.append(M[1] == G[2] + G[3])  # outcome 1 in Z, marginalize over X

# X basis: M[2] = G[0,0] + G[1,0], M[3] = G[0,1] + G[1,1]
constraints.append(M[2] == G[0] + G[2])  # outcome + in X, marginalize over Z
constraints.append(M[3] == G[1] + G[3])  # outcome - in X, marginalize over Z

# 4. t in [0,1]
constraints.append(t <= 1)

# ── Solve ────────────────────────────────────────────────────────────────────
objective = cp.Minimize(t)
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.SCS)

print(f"Minimum noise t*: {t.value:.6f}")
print(f"Analytical value: {1 - 1/np.sqrt(2):.6f}")  # known closed form ~0.2929
print(f"Match: {np.isclose(t.value, 1 - 1/np.sqrt(2), atol=1e-3)}")
for i, Gl in enumerate(G):
    print(f"G[{i}]:\n{np.round(Gl.value, 4)}")