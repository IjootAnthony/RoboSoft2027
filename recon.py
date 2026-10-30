"""
Curvature sensing and shape reconstruction for the soft arm.

Sensor j, centred at s_j with gauge length g, reports the two bending
curvatures (u1, u2) averaged over its gauge, plus white noise of rms sigma,
quantized to B bits over +-kfs. Twist is not measured.

Reconstruction integrates R' = R hat(u_hat), p' = R e3 from the clamped base
with an estimated curvature profile u_hat(s) (twist taken as zero):
  pcc       one constant curvature per actuated segment (mean of its sensors)
  linear    linear interpolation between sensor centres over the whole arm,
            held constant beyond the first and last sensor
  seglinear linear interpolation inside each actuated segment only, with
            sensors spread uniformly inside each segment (needs N a multiple
            of the number of segments)
"""
import numpy as np

from rod import R0


def placements(L, nseg, N, kind):
    if kind == "uniform":
        return (np.arange(N) + 0.5) * L / N
    per = N // nseg
    ls = L / nseg
    return np.concatenate([k * ls + (np.arange(per) + 0.5) * ls / per for k in range(nseg)])


def measure(U, L, centres, g, sigma, B, kfs, rng, twist=False):
    """U (M, ns, 3) truth curvature on the midpoint grid. Returns (M, N, 2),
    or (M, N, 3) with twist=True (three-axis sensors)."""
    ns = U.shape[1]
    s = (np.arange(ns) + 0.5) * L / ns
    nc = 3 if twist else 2
    y = np.empty((U.shape[0], len(centres), nc))
    h = L / ns
    edges = np.arange(ns + 1) * h
    for j, c in enumerate(centres):
        # exact average of the piecewise-constant curvature over the gauge:
        # each grid cell weighted by its overlap with [c - g/2, c + g/2]
        lo, hi = max(c - g / 2, 0.0), min(c + g / 2, L)
        w = np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0, None)
        y[:, j, :] = np.einsum("k,mkc->mc", w / w.sum(), U[:, :, :nc])
    if sigma:
        y = y + sigma * rng.standard_normal(y.shape)
    if B:
        q = 2 * kfs / 2 ** B
        y = np.clip(np.round(y / q) * q, -kfs, kfs - q)
    return y


def profile(y, centres, L, nseg, ns, method):
    """Estimated (u1, u2) on the midpoint grid: (M, ns, 2)."""
    s = (np.arange(ns) + 0.5) * L / ns
    M = y.shape[0]
    nc = y.shape[2]
    out = np.empty((M, ns, nc))
    if method == "linear":
        for c in range(nc):
            for i in range(M):
                out[i, :, c] = np.interp(s, centres, y[i, :, c])
        return out
    seg_of_s = np.minimum((s / L * nseg).astype(int), nseg - 1)
    seg_of_c = np.minimum((centres / L * nseg).astype(int), nseg - 1)
    for k in range(nseg):
        ms, mc = seg_of_s == k, seg_of_c == k
        if method == "pcc":
            out[:, ms, :] = y[:, mc, :].mean(axis=1)[:, None, :]
        else:   # seglinear
            for c in range(nc):
                for i in range(M):
                    out[i, ms, c] = np.interp(s[ms], centres[mc], y[i, mc, c])
    return out


def rodrigues(w):
    """Batched exp(hat(w)) for rotation vectors w (M, 3)."""
    th = np.linalg.norm(w, axis=1)[:, None, None]
    th = np.where(th < 1e-12, 1e-12, th)
    k = w / th[:, :, 0]
    Kx = np.zeros((len(w), 3, 3))
    Kx[:, 0, 1], Kx[:, 0, 2] = -k[:, 2], k[:, 1]
    Kx[:, 1, 0], Kx[:, 1, 2] = k[:, 2], -k[:, 0]
    Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 1], k[:, 0]
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def shape(u, L):
    """Integrate positions from curvature u (M, ns, 2 or 3) on the midpoint
    grid; each step rotates by the exact rotation of its constant curvature
    and advances along the mean of the start and end tangents."""
    M, ns = u.shape[:2]
    h = L / ns
    if u.shape[2] == 2:
        u = np.concatenate([u, np.zeros((M, ns, 1))], axis=2)
    R = np.broadcast_to(R0, (M, 3, 3)).copy()
    p = np.zeros((M, ns + 1, 3))
    for k in range(ns):
        Rn = R @ rodrigues(h * u[:, k, :])
        # chord of a constant-curvature arc, approximated to second order
        p[:, k + 1] = p[:, k] + h * 0.5 * (R[:, :, 2] + Rn[:, :, 2])
        R = Rn
    return p


def errors(p_hat, P, L):
    """Tip error and mean position error along the arm, as fractions of L."""
    tip = np.linalg.norm(p_hat[:, -1] - P[:, -1], axis=1) / L
    mean = np.linalg.norm(p_hat - P, axis=2).mean(axis=1) / L
    return tip, mean
