"""Twist error floor versus torsional stiffness: bending-only (2-axis) and
three-axis sensing, N = 12, segment-aware linear reconstruction, ideal
sensors. Usage: python torsion.py > results/torsion.txt"""
import numpy as np
from recon import errors, measure, placements, profile, shape
rows = [("data/shapes_gj02_a.npz",), ("data/shapes_a.npz", "data/shapes_b.npz"), ("data/shapes_gj2_a.npz",)]
print("GJ/EI  shapes  max twist (1/m)  2-axis median/95th   3-axis median/95th   (tip error, % of L)")
for files in rows:
    d = [np.load(f) for f in files]
    U = np.concatenate([x["U"] for x in d]); P = np.concatenate([x["P"] for x in d])
    L, nseg = float(d[0]["L"]), int(d[0]["nseg"])
    gj = float(d[0]["gj_ei"]) if "gj_ei" in d[0] else 2 / 3
    c = placements(L, nseg, 12, "segment")
    res = []
    for tw in (False, True):
        y = measure(U, L, c, 0.010, 0, 0, 20.0, None, twist=tw)
        t, _ = errors(shape(profile(y, c, L, nseg, U.shape[1], "seglinear"), L), P, L)
        res.append("%5.2f / %5.2f" % (np.median(t) * 100, np.percentile(t, 95) * 100))
    print(f"{gj:5.2f}  {len(U):6d}  {np.abs(U[:, :, 2]).max():8.2f}        {res[0]}        {res[1]}")
