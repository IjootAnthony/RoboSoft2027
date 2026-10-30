"""Figures for the paper. Usage: python make_figs.py (needs results/sweep.npz)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recon import errors, measure, placements, profile, shape

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})
C = {"pcc": "#2a78d6", "linear": "#eb6834", "seglinear": "#1baf7a"}
LBL = {"pcc": "one curvature per segment", "linear": "across segments",
       "seglinear": "within segments"}
S = np.load("results/sweep.npz")
NS_LIST = [3, 6, 9, 12, 15, 18, 21, 24]

# --- shapes ------------------------------------------------------------------
d = np.load("data/shapes_a.npz")
P = d["P"]
L = float(d["L"])
fig = plt.figure(figsize=(3.4, 2.0))
ax = fig.add_subplot(projection="3d")
c = placements(L, 3, 12, "segment")
idx = np.round(c / L * (P.shape[1] - 1)).astype(int)
for i in range(8):
    ax.plot(P[i, :, 0] * 100, P[i, :, 1] * 100, P[i, :, 2] * 100, color="#555555", lw=1.0)
    ax.scatter(P[i, idx, 0] * 100, P[i, idx, 1] * 100, P[i, idx, 2] * 100, s=1.2, color=C["seglinear"])
    b = [P.shape[1] // 3, 2 * P.shape[1] // 3]
    ax.scatter(P[i, b, 0] * 100, P[i, b, 1] * 100, P[i, b, 2] * 100, s=6, marker="x", color="#e87ba4", lw=0.6)
ax.scatter([0], [0], [0], s=12, color="black")
ax.set_xlabel("x (cm)", labelpad=-6); ax.set_ylabel("y (cm)", labelpad=-6); ax.set_zlabel("z (cm)", labelpad=-6)
ax.tick_params(pad=-3, labelsize=6)
ax.view_init(elev=22, azim=-62)
fig.subplots_adjust(left=0, right=1, bottom=0.02, top=1)
fig.savefig("paper/fig_shapes.pdf")

# --- error versus sensor count -----------------------------------------------
fig, ax = plt.subplots(figsize=(3.4, 2.1))
for m in ("pcc", "linear", "seglinear"):
    for tw, ls, mk in ((0, "--", "o"), (1, "-", "s")):
        y = [np.median(S[f"A_{m}_{tw}_{n}"]) * 100 for n in NS_LIST]
        ax.plot(NS_LIST, y, ls, marker=mk, ms=2.5, lw=1.0, color=C[m],
                label=(LBL[m] if tw == 1 else None))
ax.set_yscale("log")
ax.set_xlabel("sensors along the arm, N")
ax.set_ylabel("median tip error (% of L)")
ax.set_xticks(NS_LIST)
ax.plot([], [], "k--", lw=1.0, label="bending only (2-axis)")
ax.plot([], [], "k-", lw=1.0, label="with twist (3-axis)")
ax.legend(fontsize=6.5, loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.6)
ax.set_ylim(0.04, 60)
ax.grid(True, which="major", lw=0.3, color="#dddddd")
fig.tight_layout(pad=0.3)
fig.savefig("paper/fig_sensors.pdf")

# --- error versus bits -------------------------------------------------------
BITS = [6, 7, 8, 9, 10, 11, 12, 14]
KFS = 20.0
fig, ax = plt.subplots(figsize=(3.4, 2.0))
cols = {6: "#2a78d6", 12: "#eb6834", 24: "#1baf7a"}
for N in (6, 12, 24):
    rms = [np.sqrt(np.mean(S[f"B_{N}_{b}"] ** 2)) * 100 for b in BITS]
    floor = np.sqrt(np.mean(S[f"A_seglinear_1_{N}"] ** 2))
    bb = np.linspace(6, 14, 50)
    q = 2 * KFS / 2 ** bb
    model = np.sqrt(floor ** 2 + (2 / 3) * q ** 2 / 12 * L ** 2 / N) * 100
    ax.plot(BITS, rms, "o", ms=3, color=cols[N], label=f"N = {N}")
    ax.plot(bb, model, "-", lw=0.9, color=cols[N])
ax.set_yscale("log")
ax.set_xlabel("ADC bits, B (full scale $\\pm$20 m$^{-1}$)")
ax.set_ylabel("rms tip error (% of L)")
ax.plot([], [], "k-", lw=0.9, label="model, Eq. (4)")
ax.legend(fontsize=6.5, loc="upper right")
ax.grid(True, which="major", lw=0.3, color="#dddddd")
fig.tight_layout(pad=0.3)
fig.savefig("paper/fig_bits.pdf")
print("figures written")

# --- example reconstruction ---------------------------------------------------
e = np.load("results/example.npz")
fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.55))
for ax, (a, b, la, lb) in zip(axes, ((0, 1, "x (cm)", "y (cm)"), (0, 2, "x (cm)", "z (cm)"))):
    ax.plot(e["P"][:, a] * 100, e["P"][:, b] * 100, color="black", lw=1.6, label="true")
    ax.plot(e["p2"][:, a] * 100, e["p2"][:, b] * 100, "--", color="#eb6834", lw=1.0, label="bending only")
    ax.plot(e["p3"][:, a] * 100, e["p3"][:, b] * 100, ":", color="#1baf7a", lw=1.3, label="with twist")
    ax.set_xlabel(la, labelpad=1); ax.set_ylabel(lb, labelpad=1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, lw=0.3, color="#dddddd")
axes[0].legend(fontsize=6, loc="best", handlelength=1.8)
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig("paper/fig_example.pdf")
print("example written")
