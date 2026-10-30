"""
Sweeps of shape-reconstruction error for the 400 simulated arm shapes.
Gauge length 10 mm, full scale +-20 1/m (covers the largest curvature in the
data set, 19.2 1/m).

Usage: python sweep.py > results/sweep.txt   (also writes results/sweep.npz)
"""
import numpy as np

from recon import errors, measure, placements, profile, shape

d = [np.load(f) for f in ("data/shapes_a.npz", "data/shapes_b.npz")]
U = np.concatenate([x["U"] for x in d])
P = np.concatenate([x["P"] for x in d])
F = np.concatenate([x["F"] for x in d])
L, NSEG = float(d[0]["L"]), int(d[0]["nseg"])
NS = U.shape[1]
G, KFS = 0.010, 20.0
NS_LIST = [3, 6, 9, 12, 15, 18, 21, 24]
out = {}


def run(N, method, twist, sigma=0.0, B=0, seed=0):
    kind = "uniform" if method == "linear" else "segment"
    c = placements(L, NSEG, N, kind)
    y = measure(U, L, c, G, sigma, B, KFS, np.random.default_rng(seed), twist=twist)
    t, m = errors(shape(profile(y, c, L, NSEG, NS, method), L), P, L)
    return t, m


def stats(t):
    return np.median(t) * 100, np.percentile(t, 95) * 100


print(f"{len(U)} shapes ({int((np.linalg.norm(F, axis=1) > 0).sum())} with a tip load), "
      f"L = {L} m, {NSEG} segments, gauge {G*1e3:.0f} mm, full scale {KFS} 1/m")
print("\n(A) ideal sensors: tip error, % of L, median / 95th percentile")
for twist in (False, True):
    for method in ("pcc", "linear", "seglinear"):
        row = []
        for N in NS_LIST:
            if method == "pcc" and N > 3 and False:
                pass
            t, m = run(N, method, twist)
            out[f"A_{method}_{int(twist)}_{N}"] = t
            row.append("%5.2f/%5.2f" % stats(t))
        print(f"  {'3-axis' if twist else '2-axis'} {method:<9} " + " ".join(row), flush=True)
print("  N = " + " ".join(f"{n:>11}" for n in NS_LIST))

print("\n(B) quantization, 3-axis, seglinear: median / 95th pct tip error (% of L)")
BITS = [6, 7, 8, 9, 10, 11, 12, 14]
for N in (6, 12, 24):
    row = []
    for B in BITS:
        t, _ = run(N, "seglinear", True, B=B)
        out[f"B_{N}_{B}"] = t
        row.append("%5.2f/%5.2f" % stats(t))
    print(f"  N={N:<3} " + " ".join(row), flush=True)
print("  B = " + " ".join(f"{b:>11}" for b in BITS))

print("\n(C) white noise (rms, 1/m), 3-axis, seglinear, unquantized")
for N in (6, 12, 24):
    row = []
    for sg in (0.02, 0.05, 0.1, 0.2):
        ts = np.concatenate([run(N, "seglinear", True, sigma=sg, seed=k)[0] for k in range(3)])
        out[f"C_{N}_{sg}"] = ts
        row.append("%5.2f/%5.2f" % stats(ts))
    print(f"  N={N:<3} " + " ".join(row), flush=True)
print("  sigma = 0.02, 0.05, 0.1, 0.2 1/m")
np.savez_compressed("results/sweep.npz", **out)
