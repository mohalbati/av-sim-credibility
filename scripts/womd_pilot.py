"""Pilot on REAL data: compare rule-based simulated drivers with the logged drivers in WOMD.

For each scene (WOMD v1.1 validation, read from Google Cloud or a local copy):
  real = the logged drive (what humans did)
  sim  = closed-loop rollout from t = 1.1 s to 9.0 s: every non-SDC vehicle is driven by Waymax's IDMRoutePolicy
         (keeps its logged path, chooses its own speed); the SDC, pedestrians and cyclists replay their logs
Then for every non-SDC vehicle: minimum TTC (real vs sim), its scenario family, and Part-A metrics per family.

Run (WSL, venv active, from the repo folder):
    python scripts/womd_pilot.py --scenes 20          # quick test
    python scripts/womd_pilot.py --scenes 500         # pilot
    python scripts/womd_pilot.py --scenes 500 --local ~/womd     # use local copies of the shard files
Simulated-driver models (--agent), all controlling the non-SDC vehicles:
    idm             Waymax IDMRoutePolicy, default settings (desired speed 30 m/s, 2 m min gap, lookahead 10 points)
    idm_cautious    IDMRoutePolicy with assumed "cautious" settings: desired speed 15 m/s, 4 m min gap,
                    lookahead 20 points + 20 m extra; chosen by hand, not fitted (report as such)
    constant_speed  keeps the speed and heading from t = 1.1 s; ignores the road and other agents (deliberately poor)
    log             replays the log (sanity check: every metric should show perfect agreement)
Outputs: results/womd_agents_<agent>.csv (per-agent values; kept out of git), results/womd_partA_<agent>.csv / .json.
"""
import sys, argparse, dataclasses, pathlib, time, csv, json
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from avcred import validation, scenarios, credibility, womd

import jax
import jax.numpy as jnp
from waymax import config as _config, dataloader, dynamics, env as _env, agents
from waymax.utils import geometry

ap = argparse.ArgumentParser()
ap.add_argument("--scenes", type=int, default=20)
ap.add_argument("--start-shard", type=int, default=0)
ap.add_argument("--local", type=str, default=None, help="folder holding validation_tfexample.tfrecord-XXXXX-of-00150")
ap.add_argument("--tau", type=float, default=1.5, help="critical TTC (s) for the tail ratio")
ap.add_argument("--agent", choices=["idm", "idm_cautious", "constant_speed", "log"], default="idm")
args = ap.parse_args()
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
N_OBJ, T0 = 32, 11                                     # simulation starts after 1.1 s of history


def shard_cfg(k):
    cfg = _config.WOD_1_1_0_VALIDATION
    name = f"validation_tfexample.tfrecord-{k:05d}-of-00150"
    path = (str(pathlib.Path(args.local).expanduser() / name) if args.local
            else cfg.path.replace("gs:///", "gs://").replace("validation_tfexample.tfrecord@150", name))
    kw = {"path": path, "max_num_objects": N_OBJ}
    fields = {f.name for f in dataclasses.fields(cfg)}
    if "repeat" in fields: kw["repeat"] = None              # read each file once
    if "shuffle_seed" in fields: kw["shuffle_seed"] = None  # keep file order (reproducible)
    return dataclasses.replace(cfg, **kw)


def is_idm(state):
    md = state.object_metadata
    return (md.object_types == womd.TYPE_VEH) & ~md.is_sdc


dyn = dynamics.StateDynamics()
env = _env.BaseEnvironment(dynamics_model=dyn, config=dataclasses.replace(
    _config.EnvironmentConfig(), max_num_objects=N_OBJ, controlled_object=_config.ObjectType.VALID))
if args.agent == "idm":
    sim_actor = agents.IDMRoutePolicy(is_controlled_func=is_idm)
elif args.agent == "idm_cautious":
    sim_actor = agents.IDMRoutePolicy(is_controlled_func=is_idm, desired_vel=15.0, min_spacing=4.0,
                                      max_lookahead=20, additional_lookahead_distance=20.0)
elif args.agent == "constant_speed":
    sim_actor = agents.create_constant_speed_actor(dynamics_model=dyn, is_controlled_func=is_idm)
else:
    sim_actor = agents.create_expert_actor(dynamics_model=dyn, is_controlled_func=is_idm)
actors = [sim_actor,
          agents.create_expert_actor(dynamics_model=dyn, is_controlled_func=lambda s: ~is_idm(s))]
reset, step = env.reset, jax.jit(env.step)   # as in Waymax's multi_actors_demo
selects = [jax.jit(a.select_action) for a in actors]


def rollout(scenario):
    st = reset(scenario)
    for _ in range(int(st.remaining_timesteps)):
        st = step(st, agents.merge_actions([f({}, st, None, None) for f in selects]))
    return st


@jax.jit
def _overlaps(traj5, valid):
    """Waymax rotated-box overlap (the check behind WOSAC's collision feature); traj5 (N,T,5), valid (N,T)."""
    ov = geometry.compute_pairwise_overlaps(jnp.swapaxes(traj5, 0, 1))          # (T, N, N)
    v = jnp.swapaxes(valid, 0, 1)
    return ov & v[:, :, None] & v[:, None, :]


def overlaps(traj, valid):
    return np.asarray(_overlaps(traj.stack_fields(["x", "y", "length", "width", "yaw"])[:, T0:], jnp.asarray(valid)))


def arrays(traj):
    g = lambda a: np.asarray(a)[:, T0:]
    return g(traj.x), g(traj.y), g(traj.yaw), g(traj.vel_x), g(traj.vel_y), g(traj.length), g(traj.width), g(traj.valid)


rows, t_start, n_scene, shard = [], time.time(), 0, args.start_shard
fields = ["shard", "scene", "agent", "family", "real_ttc", "sim_ttc", "real_collision", "sim_collision", "real_col_where", "sim_col_where", "sim_col_partner", "lead_type_real", "lateral_shift_m", "turning", "vru_within_m"]
fcsv = open(OUT / f"womd_agents_{args.agent}.csv", "w", newline=""); wr = csv.DictWriter(fcsv, fieldnames=fields); wr.writeheader()
while n_scene < args.scenes:
    print(f"shard {shard}", flush=True); k_in_shard = 0
    for scenario in dataloader.simulator_state_generator(config=shard_cfg(shard)):
        if n_scene >= args.scenes:
            break
        final = rollout(scenario)
        types = np.asarray(scenario.object_metadata.object_types); sdc = np.asarray(scenario.object_metadata.is_sdc)
        xr, yr, hr, vxr, vyr, Lr, Wr, okr = arrays(final.log_trajectory)
        xs, ys, hs, vxs, vys, _, _, oks = arrays(final.sim_trajectory)
        L = np.array([np.median(Lr[i][okr[i]]) if okr[i].any() else 0.0 for i in range(N_OBJ)])
        W = np.array([np.median(Wr[i][okr[i]]) if okr[i].any() else 0.0 for i in range(N_OBJ)])
        ok = okr & oks
        ttc_r, lead_r, seen_r = womd.min_ttc_per_follower(xr, yr, hr, vxr, vyr, L, W, ok, types)
        ttc_s, _, _ = womd.min_ttc_per_follower(xs, ys, hs, vxs, vys, L, W, ok, types)
        ov_r, ov_s = overlaps(final.log_trajectory, ok), overlaps(final.sim_trajectory, ok)
        col_r, col_s = ov_r.any(axis=(0, 2)), ov_s.any(axis=(0, 2))
        reactive = (types == womd.TYPE_VEH) & ~sdc                                    # the IDM-controlled objects
        where_r, _ = womd.first_collision(ov_r, xr, yr, hr, np.zeros(N_OBJ, bool))
        where_s, partner_s = womd.first_collision(ov_s, xs, ys, hs, reactive)
        moving = np.nanmax(np.where(ok, np.hypot(vxr, vyr), np.nan), axis=1) > 2.0     # excludes parked vehicles
        followers = np.where((types == womd.TYPE_VEH) & ~sdc & ok[:, 0] & (ok.mean(1) >= 0.5) & moving)[0]
        for i in followers:
            f = womd.agent_features(i, xr, yr, hr, ok, types)
            fam = scenarios.classify(scenarios.SceneFeatures(
                vru_within_m=f["vru_within_m"], in_intersection=f["turning"],
                lateral_shift_m=f["lateral_shift_m"], lead_vehicle_present=bool(seen_r[i])))
            r = {"shard": shard, "scene": k_in_shard, "agent": int(i), "family": fam,
                 "real_ttc": float(ttc_r[i]), "sim_ttc": float(ttc_s[i]),
                 "real_collision": bool(col_r[i]), "sim_collision": bool(col_s[i]), "real_col_where": where_r[i],
                 "sim_col_where": where_s[i], "sim_col_partner": partner_s[i], "lead_type_real": int(lead_r[i]),
                 "lateral_shift_m": f["lateral_shift_m"], "turning": bool(f["turning"]), "vru_within_m": f["vru_within_m"]}
            wr.writerow(r); rows.append(r)
        n_scene += 1; k_in_shard += 1
        if n_scene % 10 == 0 or n_scene == 1:
            el = time.time() - t_start
            print(f"  {n_scene}/{args.scenes} scenes, {len(rows)} agents, {el:.0f} s ({el / n_scene:.1f} s/scene)", flush=True)
        fcsv.flush()
    shard += 1
fcsv.close()

# ---- Part A per family and pooled ----
COU = credibility.ContextOfUse(f"rate of min TTC < {args.tau} s among human-driven vehicles", "medium", "high")
summary = []
groups = {fam: [r for r in rows if r["family"] == fam] for fam in scenarios.FAMILIES}
groups["POOLED"] = rows
for fam, rs in groups.items():
    if not rs:
        continue
    real = np.array([r["real_ttc"] for r in rs]); sim = np.array([r["sim_ttc"] for r in rs])
    cr = np.array([r["real_collision"] for r in rs]); cs = np.array([r["sim_collision"] for r in rs])
    both = np.isfinite(real) & np.isfinite(sim) & (real > 0) & (sim > 0)
    s = {"family": fam, "agents": len(rs), "pairs_both_finite": int(both.sum()),
         "P_real_lt_tau": float(np.mean(real < args.tau)), "P_sim_lt_tau": float(np.mean(sim < args.tau)),
         "real_collision_rate": float(cr.mean()), "sim_collision_rate": float(cs.mean()),
         "real_rate_hit_ahead": float(np.mean([r["real_col_where"] == "ahead" for r in rs])),
         "sim_rate_hit_ahead": float(np.mean([r["sim_col_where"] == "ahead" for r in rs])),
         "sim_rate_struck_behind_by_log": float(np.mean([(r["sim_col_where"] == "behind") & (r["sim_col_partner"] == "log_replay") for r in rs])),
         "sim_rate_struck_behind_by_sim": float(np.mean([(r["sim_col_where"] == "behind") & (r["sim_col_partner"] == "reactive") for r in rs])),
         "area_capped10": validation.area_metric(np.minimum(sim, 10), np.minimum(real, 10))}
    if np.sum(real < args.tau) >= 5:                    # a tail ratio needs some real events
        tr = validation.tail_ratio(sim, real, args.tau, n_boot=500); s["tail_R"] = tr["R"]; s["tail_ci"] = list(tr["ci"])
    else:
        tr = {"R": float("nan")}; s["tail_R"] = float("nan")
    if both.sum() >= 3:
        b = validation.bias_uncertainty(sim[both], real[both], n_boot=500)
        s.update(delta=b["delta"], delta_ci=list(b["delta_ci"]), omega=b["omega"])
        if fam != "POOLED" and np.isfinite(tr["R"]):
            s.update(credibility.assess(COU, b["delta"], b["omega"], tr["R"]))
    summary.append(s)

keys = list(dict.fromkeys(k for s in summary for k in s))
with open(OUT / f"womd_partA_{args.agent}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summary)
json.dump({"scenes": n_scene, "tau": args.tau, "agent": args.agent, "setup": "sim agent drives non-SDC vehicles; SDC/VRUs log replay",
           "partA": summary}, open(OUT / f"womd_partA_{args.agent}.json", "w"), indent=2, default=float)
print(f"\nPart A on WOMD (sim agent = {args.agent}, real = log):")
for s in summary:
    print(f"  {s['family']:<16} agents {s['agents']:<5} pairs {s['pairs_both_finite']:<5} "
          f"delta {s.get('delta', float('nan')):.3f} omega {s.get('omega', float('nan')):.3f} "
          f"R {s['tail_R']:.2f}  collisions real/sim {100 * s['real_collision_rate']:.1f}%/{100 * s['sim_collision_rate']:.1f}%  "
          f"{s.get('decision', '')}")
print("\nSimulated collisions by cause (% of vehicles; real 'ran into' shown for comparison):")
print(f"  {'family':<16} {'ran into ahead':>15} {'(real)':>7} {'struck by log':>14} {'struck by sim':>14}")
for s in summary:
    print(f"  {s['family']:<16} {100 * s['sim_rate_hit_ahead']:>14.1f}% {100 * s['real_rate_hit_ahead']:>6.1f}% "
          f"{100 * s['sim_rate_struck_behind_by_log']:>13.1f}% {100 * s['sim_rate_struck_behind_by_sim']:>13.1f}%")
