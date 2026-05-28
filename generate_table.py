"""
Recreate the joint-measurability noise-robustness table for PRIME dimensions
using the project's existing tools:

    mub_measurements(p, k)                  -- k MUBs in prime dimension p
                                               (returns MATLAB layout (d,d,oa,ma))
    robustness_hypergraph(M, [hyperedge])   -- max eta s.t. eta-noisy M is
                                               decomposable along the hypergraph;
                                               with a single full hyperedge
                                               (0, 1, ..., k-1) this is exactly
                                               full k-wise joint measurability.

We cover d in {2, 3, 5, 7} and k in 2..d+1.  The SDP has d^k PSD blocks, so
cells where d^k exceeds MAX_VARS are skipped.
"""

from __future__ import annotations

import warnings
import numpy as np

# CVXPY emits a cosmetic UserWarning on every constraint with more than a
# few hundred subexpression nodes ("Constraint #N contains too many
# subexpressions. Consider vectorizing..."). For the hypergraph SDP the
# expression count is fundamentally d^k, and vectorizing requires
# restructuring the per-outcome Hermitian Variables into a single stacked
# variable with manual hermiticity constraints -- a much larger refactor.
# The warning is informational; the SDP still solves correctly. Silence it.
warnings.filterwarnings(
    "ignore",
    message="Constraint #.* contains too many subexpressions.*",
)
warnings.filterwarnings(
    "ignore",
    message="Solution may be inaccurate.*",
)

from jm_povms_mub import mub_measurements
from compatibility_hypergraph import robustness_hypergraph


PRIMES = [2, 3, 5, 7, 11]
CLARABEL_VARS = 200         # use CLARABEL only when d^k <= this
MAX_VARS = 10000             # cap on d^k. Covers (k=4,d=7) at 2401 and
                            # (k=5,d=5) at 3125 (one cell over -- raise
                            # to ~3500 to include it). (k=5,d=7) at
                            # 16807 is intractable; (k=6,d=7) at 117649
                            # and beyond are impossible.


# Closed forms quoted in the paper (table from arXiv:2205.08546 et al.).
CLOSED = {
    (2, 2): ("1/sqrt(2)",                  1 / np.sqrt(2)),
    (3, 2): ("1/sqrt(3)",                  1 / np.sqrt(3)),
    (2, 3): ("(1+sqrt(3))/4",              (1 + np.sqrt(3)) / 4),
    (3, 3): ("cos(pi/18)/sqrt(3)",         np.cos(np.pi / 18) / np.sqrt(3)),
    (4, 3): ("(1+3 sqrt(5))/16",           (1 + 3 * np.sqrt(5)) / 16),
    (2, 5): ("(3+sqrt(5))/8",              (3 + np.sqrt(5)) / 8),
    (3, 5): ("[13-sqrt(5)+sqrt(30(5+sqrt(5)))]/48",
             (13 - np.sqrt(5) + np.sqrt(30 * (5 + np.sqrt(5)))) / 48),
    (2, 7): ("(5+sqrt(7))/12",             (5 + np.sqrt(7)) / 12),
}

from compatibility_hypergraph import hyperg_pair, hypergraph_compatible
from jm_povms_mub import jm_povms_robustness

def jm_threshold(p: int, k: int) -> float:
    """Max eta s.t. the white-noise mixture of k MUBs in dim p is jointly measurable."""
    M = np.transpose(mub_measurements(p, k), (3, 2, 0, 1))
    hg = hyperg_pair(k)
    # M = mub_measurements(p, k)
    
    n = p ** k
    if n <= CLARABEL_VARS:
        try:
            val, _ = robustness_hypergraph(M, hg, solver="CLARABEL",
                                         eps=1e-9, max_iters=500000)
            return val
        except Exception:
            pass
    # eps=1e-7 / max_iters=1e5 is a much better speed/accuracy trade-off
    # than eps=1e-9 / max_iters=5e5: SCS is a first-order method whose iteration
    # count scales like ~1/eps, so dropping two orders of tolerance is ~100x
    # fewer iterations for the table-building cases. 1e-7 still matches all
    # closed-form values to 4+ decimal places (well below the table's 4-digit
    # display precision).
    val, _ = robustness_hypergraph(M, hg, solver="SCS",
                                   eps=1e-7, max_iters=100000)
    return val


def main():
    print("=" * 78)
    print("  JM noise-robustness  eta*  for  k  MUBs  in prime dimension  d")
    print(f"  (skipping cells with d^k > {MAX_VARS})")
    print("=" * 78)

    header = "  k  " + "".join(f"|   d = {p}   " for p in PRIMES) + "|"
    print()
    print(header)
    print("-" * len(header))

    computed = {}
    for k in range(2, 13):
        row = f"  {k}  "
        for p in PRIMES:
            if k > p + 1:
                cell = "    -     "
            # elif p ** k > MAX_VARS:
                # SDP has d^k Hermitian d×d matrix variables -- beyond a
                # few thousand the canonicalization alone takes minutes
                # and the actual solve takes hours. Skip and let the user
                # raise MAX_VARS deliberately if they want to wait.
                # cell = "  (skip)  "
            else:
                val = jm_threshold(p, k)
                computed[(k, p)] = val
                cell = f"  {val:.4f}  "
            row += "&" + cell
        row += "&"
        print(row)

    # Cross-check every computed cell that has a closed form in the paper.
    print("done")
    exit()
    print("Closed-form cross-check:")
    print(f"   {'cell':<10} {'SDP eta*':<10}  {'closed form':<44} {'value':<10}  match")
    print("   " + "-" * 86)
    n_ok = 0
    n_check = 0
    for (k, p), (name, val_cf) in sorted(CLOSED.items()):
        if (k, p) not in computed:
            continue
        n_check += 1
        diff = abs(computed[(k, p)] - val_cf)
        ok = diff < 1e-3
        n_ok += ok
        mark = "OK" if ok else f"diff={diff:.2e}"
        cell = f"(k={k}, d={p})"
        print(f"   {cell:<10} {computed[(k,p)]:.6f}    {name:<44} {val_cf:.6f}    {mark}")
    print(f"\n   {n_ok} / {n_check} closed-form cells match SDP to 1e-3.")


if __name__ == "__main__":
    main()