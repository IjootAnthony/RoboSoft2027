"""
Generate equilibrium shapes of the three-segment arm.
Each shape: random intrinsic curvature per segment (direction uniform,
magnitude uniform up to a 90 degree bend per segment), gravity, and with
probability 0.5 a random tip force of up to 0.3 N (contact or payload).
Usage: python make_dataset.py SEED COUNT OUT.npz [GJ/EI] [BLEND_M]
"""
import sys

import numpy as np

from rod import Arm, solve_continuation

seed, count, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
arm = Arm(gj_ei=float(sys.argv[4]) if len(sys.argv) > 4 else 2.0 / 3.0,
          blend=float(sys.argv[5]) if len(sys.argv) > 5 else 0.0)
rng = np.random.default_rng(seed)
kmax = (np.pi / 2) / (arm.L / arm.nseg)
US, PS, RS, UST, FT = [], [], [], [], []
for i in range(count):
    mag = rng.uniform(0, kmax, arm.nseg)
    ang = rng.uniform(0, 2 * np.pi, arm.nseg)
    ust = np.column_stack([mag * np.cos(ang), mag * np.sin(ang)])
    F = np.zeros(3)
    if rng.random() < 0.5:
        v = rng.standard_normal(3)
        F = rng.uniform(0, 0.3) * v / np.linalg.norm(v)
    try:
        _, P, U, R = solve_continuation(arm, ust, F)
    except RuntimeError:
        continue
    US.append(U); PS.append(P); RS.append(R); UST.append(ust); FT.append(F)
    if i % 20 == 0:
        print(seed, i, flush=True)
np.savez_compressed(out, U=np.array(US), P=np.array(PS), R=np.array(RS),
                    ustar=np.array(UST), F=np.array(FT), L=arm.L, nseg=arm.nseg, gj_ei=arm.gj_ei, blend=arm.blend)
print("saved", len(US))
