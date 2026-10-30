"""Gauge length and load-case breakdown (3-axis, per segment, N=12, ideal),
and an example reconstruction for Fig. 2. Usage: python extra.py > results/extra.txt"""
import numpy as np
from recon import errors, measure, placements, profile, shape
d = [np.load(f) for f in ("data/shapes_a.npz", "data/shapes_b.npz")]
U = np.concatenate([x["U"] for x in d]); P = np.concatenate([x["P"] for x in d]); F = np.concatenate([x["F"] for x in d])
L, NSEG, NS = 0.25, 3, U.shape[1]
loaded = np.linalg.norm(F, axis=1) > 0

def run(N, g, twist=True, method="seglinear", kind="segment"):
    c = placements(L, NSEG, N, kind)
    y = measure(U, L, c, g, 0, 0, 20.0, None, twist=twist)
    p = shape(profile(y, c, L, NSEG, NS, method), L)
    return errors(p, P, L)[0], p

print("Gauge length (3-axis, per segment, ideal): median / 95th pct tip error, % of L")
for N in (6, 12):
    row = []
    for g in (0.002, 0.005, 0.010, 0.020, 0.0278):
        if g * N / 3 > L / 3 + 1e-9:
            row.append("   n/a     "); continue
        t, _ = run(N, g)
        row.append("%5.2f/%5.2f" % (np.median(t) * 100, np.percentile(t, 95) * 100))
    print(f"  N={N:<3}" + " ".join(row))
print("  g = 2, 5, 10, 20, 27.8 mm")
print("\nLoad cases, N=12 per segment, ideal: median / 95th pct (% of L)")
for tw in (False, True):
    t, _ = run(12, 0.010, twist=tw)
    print(f"  {'3-axis' if tw else '2-axis'}  no tip load {np.median(t[~loaded])*100:.2f}/{np.percentile(t[~loaded],95)*100:.2f}"
          f"   tip load {np.median(t[loaded])*100:.2f}/{np.percentile(t[loaded],95)*100:.2f}")
t2, p2 = run(12, 0.010, twist=False)
t3, p3 = run(12, 0.010, twist=True)
i = int(np.argsort(t2)[int(0.9 * len(t2))])     # a 90th-percentile case
print(f"\nExample shape {i}: bending-only tip error {t2[i]*100:.2f} % ({t2[i]*L*1e3:.1f} mm), "
      f"three-axis {t3[i]*100:.3f} % ({t3[i]*L*1e3:.2f} mm), tip load {np.linalg.norm(F[i]):.2f} N")
np.savez("results/example.npz", P=P[i], p2=p2[i], p3=p3[i])
