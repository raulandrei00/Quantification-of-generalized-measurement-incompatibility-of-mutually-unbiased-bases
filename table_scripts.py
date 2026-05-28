
import jm_povms_mub, compatibility_hypergraph
import mub_joint_measurability
import numpy as np


def generate_compatibility_table():
    for d in [2, 3, 5, 7]:
        print(f"Dimension {d}:")
        povms = jm_povms_mub.mub_measurements(d, d+1)
        print(povms.shape)
        M = mub_joint_measurability.noisy_mub_measurements(povms, eta=1.0)  # sharp MUBs

        hypergraph = compatibility_hypergraph.hypergraph_compatible(len(povms))
        print(f"  Robustness for compatibility: ", end="")
        try:
            eta_star, st = compatibility_hypergraph.robustness_hypergraph(M, hypergraph)
        except Exception:
            eta_star, st = compatibility_hypergraph.robustness_hypergraph(M, hypergraph, solver="SCS", eps=1e-8)
        print(f"  eta* = {eta_star:.6f}   [{st}]")

if __name__ == "__main__":
    generate_compatibility_table()