"""Cross-check of the twist profile against PyElastica (240 elements).
Compares the twist curvature u3 along the arm and the total twist angle.
Usage: python validate_twist.py > results/validate_twist.txt"""
import numpy as np
import elastica as ea
from rod import Arm, solve_continuation
arm = Arm()
CASES = [
    None,
    (np.array([[8.0, 3.0], [-5.0, 6.0], [10.0, -4.0]]), np.zeros(3)),     # 3-D actuation
    (np.array([[6.0, 0.0], [6.0, 0.0], [-4.0, 5.0]]), np.array([0.0, 0.15, -0.1])),  # + tip load
]
class Sim(ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping):
    pass
def run_elastica(us, F, n_elem=240, T=3.0, dt=5e-6):
    sim = Sim()
    rod = ea.CosseratRod.straight_rod(n_elem, np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]),
                                      arm.L, arm.r, arm.rho, youngs_modulus=arm.E, shear_modulus=arm.E / 3)
    sim.append(rod)
    sim.constrain(rod).using(ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,))
    sim.add_forcing_to(rod).using(ea.GravityForces, acc_gravity=np.array([0, 0, -arm.g]))
    if np.any(F):
        sim.add_forcing_to(rod).using(ea.EndpointForces, start_force=np.zeros(3), end_force=F, ramp_up_time=0.5)
    sim.dampen(rod).using(ea.AnalyticalLinearDamper, damping_constant=2.0, time_step=dt)
    sim.finalize()
    s_v = np.arange(1, n_elem) * arm.L / n_elem
    idx = np.minimum((s_v / arm.L * arm.nseg).astype(int), arm.nseg - 1)
    rod.rest_kappa[0, :] = us[idx, 0]; rod.rest_kappa[1, :] = us[idx, 1]
    ea.integrate(ea.PositionVerlet(), sim, T, int(T / dt), progress_bar=False)
    return s_v, rod.kappa[2].copy()
for us, F in CASES[1:]:
    _, P, U, _ = solve_continuation(arm, us, F)
    s = (np.arange(U.shape[0]) + 0.5) * arm.L / U.shape[0]
    sv, k3 = run_elastica(us, F)
    u3 = np.interp(sv, s, U[:, 2])
    tw_s = np.trapezoid(U[:, 2], s); tw_e = np.trapezoid(k3, sv)
    print(f"max |u3| shooting {np.abs(U[:,2]).max():.3f} 1/m, PyElastica {np.abs(k3).max():.3f} 1/m; "
          f"rms difference {np.sqrt(np.mean((u3-k3)**2)):.3f} 1/m; total twist {np.degrees(tw_s):.2f} vs {np.degrees(tw_e):.2f} deg", flush=True)
