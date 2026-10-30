"""
Cross-check of the static shooting solver against PyElastica (dynamic
Cosserat rod with shear and extension, relaxed to rest with damping).
Usage: python validate_elastica.py > results/validate_elastica.txt
"""
import time

import numpy as np
import elastica as ea

from rod import Arm, solve_continuation

arm = Arm()
CASES = [
    (np.array([[0.0, 0.0]] * 3), np.zeros(3)),                              # gravity only
    (np.array([[8.0, 3.0], [-5.0, 6.0], [10.0, -4.0]]), np.zeros(3)),     # 3-D actuation
    (np.array([[6.0, 0.0], [6.0, 0.0], [-4.0, 5.0]]), np.array([0.0, 0.15, -0.1])),  # + tip load
]


class Sim(ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping):
    pass


def elastica_tip(ustar_seg, F_tip, n_elem=240, T=3.0, dt=5e-6):
    sim = Sim()
    G = arm.E / 3.0
    rod = ea.CosseratRod.straight_rod(
        n_elem, np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]),
        arm.L, arm.r, arm.rho, youngs_modulus=arm.E, shear_modulus=G)
    sim.append(rod)
    sim.constrain(rod).using(ea.OneEndFixedBC, constrained_position_idx=(0,),
                             constrained_director_idx=(0,))
    sim.add_forcing_to(rod).using(ea.GravityForces, acc_gravity=np.array([0, 0, -arm.g]))
    if np.any(F_tip):
        sim.add_forcing_to(rod).using(ea.EndpointForces, start_force=np.zeros(3),
                                      end_force=F_tip, ramp_up_time=0.5)
    sim.dampen(rod).using(ea.AnalyticalLinearDamper, damping_constant=2.0, time_step=dt)
    sim.finalize()
    # intrinsic curvature per segment, at the n_elem-1 internal nodes (voronoi)
    s_v = (np.arange(1, n_elem) ) * arm.L / n_elem
    idx = np.minimum((s_v / arm.L * arm.nseg).astype(int), arm.nseg - 1)
    rod.rest_kappa[0, :] = ustar_seg[idx, 0]
    rod.rest_kappa[1, :] = ustar_seg[idx, 1]
    ea.integrate(ea.PositionVerlet(), sim, T, int(T / dt), progress_bar=False)
    return rod.position_collection[:, -1].copy(), float(np.abs(rod.velocity_collection).max())


for us, F in CASES:
    _, P, _, _ = solve_continuation(arm, us, F)
    t = time.time()
    tip_e, vmax = elastica_tip(us, F)
    d = np.linalg.norm(tip_e - P[-1])
    print(f"shooting tip {np.round(P[-1]*1e3,2)} mm   PyElastica tip {np.round(tip_e*1e3,2)} mm   "
          f"difference {d*1e3:.2f} mm ({d/arm.L*100:.2f} % of L), residual speed {vmax:.1e} m/s, {time.time()-t:.0f} s",
          flush=True)
