"""
Static Kirchhoff rod model of a multi-segment soft continuum arm.

Inextensible, unshearable rod of length L, clamped at the base (tangent
horizontal, along +x), loaded by its own weight and an optional tip force.
Each actuated segment (pneumatic chambers or tendons) imposes a piecewise
constant intrinsic bending curvature u*(s) in the body frame. Constitutive
law: R^T m = K (u - u*), K = diag(EI, EI, GJ).

Equations (spatial force n, moment m):
    p' = R e3,  R' = R hat(u),  n' = -f,  m' = -p' x n,
    u = u* + K^-1 R^T m.
Boundary conditions: p(0)=0, R(0)=R0; n(L) = F_tip, m(L) = 0.
Solved by shooting on (n(0), m(0)) with Newton's method; all Jacobian
columns are integrated together (batched RK4).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Arm:
    L: float = 0.25          # m
    r: float = 0.015         # m, outer radius (solid section)
    E: float = 3.0e6         # Pa, effective modulus (reinforced elastomer)
    rho: float = 1080.0      # kg/m^3
    nseg: int = 3            # actuated segments of equal length
    nsteps: int = 300        # integration steps along the arm
    g: float = 9.81
    gj_ei: float = 2.0 / 3.0  # torsional / bending stiffness (solid incompressible)
    blend: float = 0.0        # m, width of a linear transition of u* across segment boundaries

    @property
    def I(self):
        return np.pi * self.r ** 4 / 4

    @property
    def K(self):
        EI = self.E * self.I
        return np.array([EI, EI, self.gj_ei * EI])

    @property
    def w(self):
        return self.rho * np.pi * self.r ** 2 * self.g   # N/m


R0 = np.column_stack([[0, 0, 1.0], [0, -1.0, 0], [1.0, 0, 0]])   # d3 = +x


def hat(u):
    """Batched skew matrix: u (..., 3) -> (..., 3, 3)."""
    z = np.zeros(u.shape[:-1])
    return np.stack([np.stack([z, -u[..., 2], u[..., 1]], -1),
                     np.stack([u[..., 2], z, -u[..., 0]], -1),
                     np.stack([-u[..., 1], u[..., 0], z], -1)], -2)


def ustar_profile(arm, ustar_seg, s):
    """Body-frame intrinsic curvature at arc lengths s; ustar_seg (nseg, 2)."""
    idx = np.minimum((s / arm.L * arm.nseg).astype(int), arm.nseg - 1)
    out = np.zeros((len(s), 3))
    out[:, :2] = ustar_seg[idx]
    if arm.blend > 0:
        ls = arm.L / arm.nseg
        for k in range(1, arm.nseg):
            b = k * ls
            w = np.abs(s - b) < arm.blend / 2
            if np.any(w):
                a = ((s[w] - b) / arm.blend + 0.5)[:, None]   # 0 -> 1 across the band
                out[w, :2] = (1 - a) * ustar_seg[k - 1] + a * ustar_seg[k]
    return out


def integrate(arm, ustar_seg, n0, m0, store=False):
    """RK4 from base to tip for a batch of initial (n0, m0) (B, 3) each.
    Returns tip n, m (B, 3) and, if store, the profiles of the first item."""
    B = n0.shape[0]
    h = arm.L / arm.nsteps
    Kinv = 1.0 / arm.K
    f = np.array([0.0, 0.0, -arm.w])
    e3 = np.array([0.0, 0.0, 1.0])

    def rhs(s, p, R, n, m):
        us = ustar_profile(arm, ustar_seg, np.array([s]))[0]
        u = us + Kinv * np.einsum("bji,bj->bi", R, m)
        dp = R @ e3
        dR = R @ hat(u)
        dn = np.broadcast_to(-f, n.shape)
        dm = -np.cross(dp, n)
        return dp, dR, dn, dm, u

    p = np.zeros((B, 3))
    R = np.broadcast_to(R0, (B, 3, 3)).copy()
    n, m = n0.copy(), m0.copy()
    if store:
        P, U, RR = [p[0].copy()], [], [R[0].copy()]
    for k in range(arm.nsteps):
        s = k * h
        # evaluate u* at the midpoint of the step for the whole step, so a
        # segment boundary on a step edge is resolved exactly
        sm = s + h / 2
        k1 = rhs(sm, p, R, n, m)
        k2 = rhs(sm, p + h / 2 * k1[0], R + h / 2 * k1[1], n + h / 2 * k1[2], m + h / 2 * k1[3])
        k3 = rhs(sm, p + h / 2 * k2[0], R + h / 2 * k2[1], n + h / 2 * k2[2], m + h / 2 * k2[3])
        k4 = rhs(sm, p + h * k3[0], R + h * k3[1], n + h * k3[2], m + h * k3[3])
        p = p + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        R = R + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        n = n + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        m = m + h / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        # re-orthonormalize (polar decomposition via SVD)
        Uu, _, Vt = np.linalg.svd(R)
        R = Uu @ Vt
        if store:
            P.append(p[0].copy()); RR.append(R[0].copy())
            us = ustar_profile(arm, ustar_seg, np.array([sm]))[0]
            U.append(us + Kinv * (R[0].T @ m[0]))
    if store:
        return n, m, np.array(P), np.array(U), np.array(RR)
    return n, m


def solve(arm, ustar_seg, F_tip, x0=None, tol=1e-10, maxit=30):
    """Newton shooting on y = (n0, m0). Returns y and the profiles."""
    y = np.zeros(6) if x0 is None else x0.copy()
    if x0 is None:
        # rigid-body guess: base reaction to weight and tip load
        y[:3] = F_tip + np.array([0, 0, arm.w * arm.L])
    target = np.concatenate([F_tip, np.zeros(3)])
    scale = np.array([1, 1, 1, 1 / arm.L, 1 / arm.L, 1 / arm.L])
    for it in range(maxit):
        eps = 1e-7 * (1 + np.abs(y))
        Y = np.vstack([y] + [y + eps[j] * np.eye(6)[j] for j in range(6)])
        nL, mL = integrate(arm, ustar_seg, Y[:, :3], Y[:, 3:])
        res = (np.hstack([nL, mL]) - target) * scale
        J = (res[1:] - res[0]).T / eps
        r0 = res[0]
        if np.max(np.abs(r0)) < tol:
            break
        dy = np.linalg.solve(J, -r0)
        # damped step
        lam = 1.0
        y = y + lam * dy
    else:
        raise RuntimeError(f"no convergence, residual {np.max(np.abs(r0)):.2e}")
    _, _, P, U, RR = integrate(arm, ustar_seg, y[None, :3], y[None, 3:], store=True)
    return y, P, U, RR


def solve_continuation(arm, ustar_seg, F_tip, steps=3):
    """Ramp actuation and tip load from zero for robust convergence."""
    y = None
    for a in np.linspace(1.0 / steps, 1.0, steps):
        y, P, U, RR = solve(arm, a * ustar_seg, a * F_tip, x0=y)
    return y, P, U, RR
