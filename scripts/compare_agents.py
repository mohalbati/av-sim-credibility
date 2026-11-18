"""Compare simulated-driver models after running womd_pilot.py once per --agent.

Reads results/womd_partA_<agent>.json and writes results/agent_comparison.csv and results/agent_comparison.png.
Run: python scripts/compare_agents.py
"""
import json, csv, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
ORDER = ["log", "idm_cautious", "idm", "constant_speed"]
FAMS = ["car_following", "lane_change", "intersection", "vru_interaction", "POOLED"]
runs = {}
for a in ORDER:
    f = RES / f"womd_partA_{a}.json"
    if f.exists():
        runs[a] = {s["family"]: s for s in json.load(open(f))["partA"]}
if not runs:
    sys.exit("no results/womd_partA_<agent>.json files found; run womd_pilot.py --agent ... first")

rows = []
for a, fams in runs.items():
    for fam in FAMS:
        s = fams.get(fam)
        if s:
            rows.append({"agent": a, "family": fam, "agents": s["agents"], "delta": s.get("delta"), "omega": s.get("omega"),
                         "tail_R": s.get("tail_R"), "sim_ran_into_ahead": s.get("sim_rate_hit_ahead"),
                         "real_ran_into_ahead": s.get("real_rate_hit_ahead"), "achieved": s.get("achieved"),
                         "required": s.get("required"), "decision": s.get("decision", "")})
with open(RES / "agent_comparison.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

metrics = [("tail_R", "Tail ratio R  (1 = right; log scale)"), ("delta", "Bias \u03b4  (1 = right; log scale)"),
           ("omega", "Scatter \u03c9  (0 = right)"), ("sim_ran_into_ahead", "Ran into something ahead (% of vehicles)")]
colors = {"log": "#8a8f98", "idm_cautious": "#2F6DB5", "idm": "#E0A33A", "constant_speed": "#B4561C"}
labels = {"car_following": "Car following", "lane_change": "Lane change", "intersection": "Intersection",
          "vru_interaction": "Pedestrian / cyclist", "POOLED": "Pooled"}
# acceptance limits from credibility.CRED_THRESHOLDS (assumed): level 1 (loosest) and level 4 (required here)
L1 = {"tail_R": 5.0, "delta": 2.0, "omega": 1.0}; L4 = {"tail_R": 1.5, "delta": 1.15, "omega": 0.25}
fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), sharey=True)
y = np.arange(len(FAMS)) + np.array([0, 0, 0, 0, 0.35])[:len(FAMS)]; h = 0.8 / len(runs)
for ax, (m, title) in zip(axes, metrics):
    if m in L1:
        if m == "omega":
            ax.axvspan(0, L1[m], color="#2F6DB5", alpha=0.06, lw=0); ax.axvspan(0, L4[m], color="#2F6DB5", alpha=0.12, lw=0)
        else:
            ax.axvspan(1 / L1[m], L1[m], color="#2F6DB5", alpha=0.06, lw=0); ax.axvspan(1 / L4[m], L4[m], color="#2F6DB5", alpha=0.12, lw=0)
    for k, a in enumerate(runs):
        vals = [runs[a].get(fam, {}).get({"sim_ran_into_ahead": "sim_rate_hit_ahead"}.get(m, m)) for fam in FAMS]
        vals = [np.nan if v is None else (100 * v if m == "sim_ran_into_ahead" else v) for v in vals]
        if m in ("tail_R", "delta"):      # bars on a log axis start at 1: they show the direction and size of the error
            vals = np.array(vals, float)
            ax.barh(y + (k - (len(runs) - 1) / 2) * h, vals - 1, left=1, height=h, color=colors[a], label=a)
        else:
            ax.barh(y + (k - (len(runs) - 1) / 2) * h, vals, height=h, color=colors[a], label=a)
    if m in ("tail_R", "delta"):
        ax.set_xscale("log"); ax.axvline(1, color="k", lw=0.8)
        ticks = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8] if m == "delta" else [0.25, 0.5, 1, 2, 4, 8]
        ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}" for t in ticks]); ax.minorticks_off()
        ax.set_xlim(0.04 if m == "delta" else 0.2, 8)
    if m == "sim_ran_into_ahead":
        ax.set_xlim(0, None)
    ax.set_title(title, loc="left", fontsize=10); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_yticks(y, [labels.get(f, f) for f in FAMS]); axes[0].invert_yaxis()
for t in axes[0].get_yticklabels():
    if t.get_text() == "Pooled": t.set_fontweight("bold")
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=9, loc="upper right", ncol=len(runs))
fig.text(0.01, 0.015, "Shaded: acceptance range for credibility level 1 (light) and level 4 (darker, required for this decision); "
         "assumed thresholds. 'log' replays the recording (sanity check); its ran-into value is the real noise floor. "
         "WOMD v1.1 validation, 2,000 scenes.", fontsize=8)
plt.tight_layout(rect=(0, 0.06, 1, 0.92)); plt.savefig(RES / "agent_comparison.png", dpi=160)

print(f"{'agent':<15} {'family':<16} {'delta':>6} {'omega':>6} {'R':>6} {'ran-into':>9}  decision")
for r in rows:
    f = lambda v, p=3: "  n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{p}f}"
    print(f"{r['agent']:<15} {r['family']:<16} {f(r['delta']):>6} {f(r['omega']):>6} {f(r['tail_R'], 2):>6} "
          f"{f(None if r['sim_ran_into_ahead'] is None else 100 * r['sim_ran_into_ahead'], 1):>8}%  {r['decision']}")
print("wrote results/agent_comparison.csv and results/agent_comparison.png")
