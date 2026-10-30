"""
Error model versus simulation, and the bit budget.

Model: each of the N sensors carries an independent error of rms sigma_eff
per bending component, with sigma_eff^2 = sigma^2 + q^2/12 (white noise plus
quantization step q = 2 kfs / 2^B). A curvature error held over the sensor
spacing L/N at arc length s tilts the rest of the arm, moving the tip by
about (L - s) (L/N) times the error. Summing over sensors and two bending
axes:
    e_rms^2 = e_floor^2 + (2/3) sigma_eff^2 L^4 / N      (tip, absolute)
so the sensing term relative to L is sigma_eff * L * sqrt(2 / (3 N)).

Bit budget: the fewest bits whose 95th-percentile tip error is within 20 %
of the ideal-sensor value.
Usage: python analysis.py > results/analysis.txt
"""
import numpy as np

from recon import errors, measure, placements, profile, shape

d = [np.load(f) for f in ("data/shapes_a.npz", "data/shapes_b.npz")]
U = np.concatenate([x["U"] for x in d])
P = np.concatenate([x["P"] for x in d])
L, NSEG = float(d[0]["L"]), int(d[0]["nseg"])
NS = U.shape[1]
G, KFS = 0.010, 20.0


def run(N, sigma=0.0, B=0, seed=0):
    c = placements(L, NSEG, N, "segment")
    y = measure(U, L, c, G, sigma, B, KFS, np.random.default_rng(seed), twist=True)
    t, _ = errors(shape(profile(y, c, L, NSEG, NS, "seglinear"), L), P, L)
    return t


def rms(t):
    return np.sqrt(np.mean(t ** 2))


print("Model check (rms tip error, % of L): simulation vs model")
for N in (6, 12, 24):
    floor = rms(run(N))
    cells = []
    for B in (6, 8, 10):
        q = 2 * KFS / 2 ** B
        model = np.sqrt(floor ** 2 + (2 / 3) * (q ** 2 / 12) * L ** 2 / N)
        cells.append(f"B={B}: {rms(run(N, B=B))*100:.2f} vs {model*100:.2f}")
    for sg in (0.05, 0.2):
        model = np.sqrt(floor ** 2 + (2 / 3) * sg ** 2 * L ** 2 / N)
        sim = rms(np.concatenate([run(N, sigma=sg, seed=k) for k in range(3)]))
        cells.append(f"sigma={sg}: {sim*100:.2f} vs {model*100:.2f}")
    print(f"  N={N:<3} floor {floor*100:.3f}   " + "   ".join(cells))

print("\nBit budget: fewest bits with 95th-percentile tip error within 20 % of ideal")
print("  N   ideal p95 (% L)   bits   p95 at those bits")
for N in (3, 6, 9, 12, 18, 24):
    ideal = np.percentile(run(N), 95)
    for B in range(5, 16):
        p = np.percentile(run(N, B=B), 95)
        if p <= 1.2 * ideal:
            print(f"  {N:<3} {ideal*100:10.3f}        {B:4d}   {p*100:.3f}")
            break
