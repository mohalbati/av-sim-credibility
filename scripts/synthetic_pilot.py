"""End-to-end pilot on SYNTHETIC data (no Waymo data needed). It checks the pipeline, not the world.

Four scenario families get known simulation errors; the script recovers them (Part A), shows that a pooled
average hides the bad families, scores credibility, and turns it into a safety estimate (Part B).
Run: python scripts/synthetic_pilot.py   -> results/pilot_*.csv, results/pilot_*.json, results/pilot_figure.png
"""
import sys, json, pathlib, csv
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from avcred import validation, credibility, evidence

OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(2026)
TAU = 1.5                                   # critical minimum TTC (s)
# family: (scenes, real median min-TTC, real log-SD, TRUE sim bias delta, TRUE sim scatter omega)
FAMILIES = {
    "car_following":   (3000, 4.0, 0.45, 1.05, 0.15),
    "lane_change":     (1500, 3.0, 0.55, 1.30, 0.35),
    "intersection":    (1200, 3.5, 0.60, 1.00, 0.50),
    "vru_interaction": (600,  2.8, 0.50, 1.60, 0.60),
}
COU = credibility.ContextOfUse("conflict rate (min TTC < 1.5 s) for release gating", "medium", "high")

rows, all_sim, all_real = [], [], []
for fam, (n, med, sd, d_true, w_true) in FAMILIES.items():
    real = med * rng.lognormal(0, sd, n)
    sim = real * np.exp(rng.normal(np.log(d_true), w_true, n))   # paired: same scene, simulated
    m = validation.validate_family(sim, real, TAU, n_boot=500)
    a = credibility.assess(COU, m["bias_delta"], m["bias_omega"], m["tail_R"])
    rows.append({"family": fam, "scenes": n, "true_delta": d_true, "true_omega": w_true,
                 "delta": round(m["bias_delta"], 3), "omega": round(m["bias_omega"], 3),
                 "area_s": round(m["area"], 3), "tail_R": round(m["tail_R"], 3),
                 "tail_R_ci": [round(float(x), 3) for x in m["tail_ci"]], "required": a["required"],
                 "achieved": a["achieved"], "decision": a["decision"]})
    all_sim.append(sim); all_real.append(real)

pooled = validation.validate_family(np.concatenate(all_sim), np.concatenate(all_real), TAU, n_boot=300)
pooled_row = {"family": "POOLED (all scenes)", "delta": round(pooled["bias_delta"], 3), "omega": round(pooled["bias_omega"], 3),
              "area_s": round(pooled["area"], 3), "tail_R": round(pooled["tail_R"], 3)}

# Part B: synthetic road and simulation exposure per family (events = scenes with min TTC < TAU)
partb = []
for r, sim, real in zip(rows, all_sim, all_real):
    M_road, M_sim = 50.0, 1000.0                       # exposure units (e.g. thousand encounters)
    k_road = rng.poisson(np.mean(real < TAU) * M_road * 2)   # road events, true rate = P(real<TAU) * 2 per unit
    k_sim = np.mean(sim < TAU) * M_sim * 2
    k_corr = evidence.corrected_sim_events(k_sim, r["tail_R"])
    rules = {"level": evidence.a_from_level(r["achieved"]), "omega": evidence.a_from_omega(r["omega"]),
             "empirical": evidence.a_empirical(k_road, M_road, k_corr, M_sim)}
    road_only = evidence.summarize(evidence.posterior(k_road, M_road))
    res = {"family": r["family"], "k_road": int(k_road), "road_only": road_only}
    for name, a in rules.items():
        res[f"a_{name}"] = round(a, 3)
        res[f"credit_{name}"] = round(evidence.simulation_credit(a, M_sim), 1)
        res[f"post_{name}"] = evidence.summarize(evidence.posterior(k_road, M_road, k_corr, M_sim, a))
    partb.append(res)

with open(OUT / "pilot_partA.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump({"tau": TAU, "context_of_use": COU.__dict__, "partA": rows, "pooled": pooled_row, "partB": partb},
          open(OUT / "pilot_results.json", "w"), indent=2)

# figure: per-family delta with CI, and the pooled value
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
names = [r["family"] for r in rows] + ["pooled"]
deltas = [r["delta"] for r in rows] + [pooled_row["delta"]]
tails = [r["tail_R"] for r in rows] + [pooled_row["tail_R"]]
cols = ["#2F6DB5" if r["decision"] == "credit" else "#E0A33A" if "correction" in r["decision"] else "#B4561C" for r in rows] + ["#8a8f98"]
ax[0].barh(names, deltas, color=cols); ax[0].axvline(1, color="k", lw=0.8); ax[0].invert_yaxis()
ax[0].set_xlabel("Bias delta (sim / real min TTC)"); ax[0].set_title("Per-family bias; pooled looks fine", loc="left", fontsize=10)
ax[1].barh(names, tails, color=cols); ax[1].axvline(1, color="k", lw=0.8); ax[1].invert_yaxis()
ax[1].set_xlabel("Tail ratio R = P_sim(TTC<1.5) / P_real(TTC<1.5)"); ax[1].set_title("Tail representation", loc="left", fontsize=10)
for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
fig.text(0.01, 0.01, "SYNTHETIC data. Blue: credit; amber: credit with correction; orange: do not credit; grey: pooled", fontsize=8)
plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig(OUT / "pilot_figure.png", dpi=160)

print("Part A (synthetic):")
for r in rows + [pooled_row]:
    print(f"  {r['family']:<20} delta {r['delta']:<6} omega {r['omega']:<6} area {r['area_s']:<6} R {r['tail_R']:<6} {r.get('decision', '')}")
print("Part B (synthetic): a and credit per rule")
for p in partb:
    print(f"  {p['family']:<16} road-only {p['road_only']['lo']:.3f}-{p['road_only']['hi']:.3f} | " +
          " | ".join(f"{k}: a={p['a_'+k]} credit={p['credit_'+k]} -> {p['post_'+k]['lo']:.3f}-{p['post_'+k]['hi']:.3f}" for k in ("level", "omega", "empirical")))
