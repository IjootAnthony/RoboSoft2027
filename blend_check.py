"""Interpolation across vs within segments when the actuation changes
smoothly over 20 mm at each segment boundary (200 shapes), and with sharp
steps (400 shapes). 3-axis ideal sensors at the same positions.
Usage: python blend_check.py > results/blend_check.txt"""
import numpy as np
from recon import errors, measure, placements, profile, shape
sets = {"sharp steps": ("data/shapes_a.npz", "data/shapes_b.npz"),
        "20 mm transitions": ("data/shapes_blend_a.npz", "data/shapes_blend_b.npz")}
print("median / 95th pct tip error (% of L); same sensor positions, two interpolations")
for name, files in sets.items():
    d = [np.load(f) for f in files]
    U = np.concatenate([x["U"] for x in d]); P = np.concatenate([x["P"] for x in d])
    L, nseg = float(d[0]["L"]), int(d[0]["nseg"])
    row = []
    for N in (6, 9, 12, 24):
        c = placements(L, nseg, N, "uniform")
        y = measure(U, L, c, 0.010, 0, 0, 20.0, None, twist=True)
        r = []
        for m in ("linear", "seglinear"):
            t, _ = errors(shape(profile(y, c, L, nseg, U.shape[1], m), L), P, L)
            r.append("%.2f/%.2f" % (np.median(t) * 100, np.percentile(t, 95) * 100))
        row.append(f"N={N}: across {r[0]}  within {r[1]}")
    print(f"  {name:<18} ({len(U)} shapes)  " + "   ".join(row))
