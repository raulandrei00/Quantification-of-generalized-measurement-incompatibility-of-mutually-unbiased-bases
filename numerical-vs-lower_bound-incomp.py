import numpy as np
from jm_povms_mub import mub_measurements
from compatibility_hypergraph import robustness_hypergraph, hyperg_pair
from generate_table import jm_threshold


def eta_2_mub (d: int) -> float:
    
    return (2 + np.sqrt(d)) / (2 * (np.sqrt(d) + 1))

def lb_symmetric_mixture (d: int, k: int) -> float:
    """Lower bound on eta* for joint measurability of k MUBs in dim d."""
    ''' 1 / (k choose 2) * [(k-1 choose 1) * eta_2_mub(d) + (k choose 2 - k-1 choose 1) * 1]'''
    return ((k-1) * eta_2_mub(d) + (k * (k - 1) // 2 - (k-1)) * 1) / (k * (k - 1) // 2)


if __name__ == "__main__":
    print("Lower bound for noise-robustness of joint measurability of k MUBs in prime dimension d:")
    print("  (2 + sqrt(d)) / (2 * (sqrt(d) + 1))")
    print()
    for d in [2, 3, 5, 7]:
        for k in range(2, d+2):
            print(f"  d={d}, k={k}:   eta* >= {lb_symmetric_mixture(d, k):.6f}")
            print(f"    (numerical eta* from SDP: {jm_threshold(d, k):.6f})")
