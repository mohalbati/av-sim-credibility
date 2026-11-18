"""Data figures for docs/technical_report.md, drawn from the aggregate results in results/.

Needs only the aggregate files (womd_partA_<setting>.json, agent_comparison.csv, pilot_results.json), so it runs
without the dataset. The concept sketches in docs/figures (ttc, families, cdf, scatter, attribution, loop) are static.
Run: python scripts/make_figures.py
"""
import sys, json, csv, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from avcred import credibility, evidence

RES, FIG = ROOT / "results", ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
BLUE, GREY, ORANGE, RUST, LIGHT = "#2F6DB5", "#8a8f98", "#E0A33A", "#B4561C", "#d9dde3"
NAMES = {"car_following": "Car following", "lane_change": "Lane change", "intersection": "Intersection",
         "vru_interaction": "Pedestrian / cyclist", "other": "Other", "POOLED": "Pooled"}
SETTINGS = ["log", "idm_cautious", "idm", "constant_speed"]
COL = {"log": GREY, "idm_cautious": BLUE, "idm": ORANGE, "constant_speed": RUST}


def part_a(setting):
    return {r["family"]: r for r in json.load(open(RES / f"womd_partA_{setting}.json"))["partA"]}


def save(fig, name):
    fig.savefig(FIG / name, dpi=170, bbox_inches="tight"); plt.close(fig); print("wrote docs/figures/" + name)


# 1. vehicles per family
idm = part_a("idm")
fams = ["car_following", "vru_interaction", "other", "intersection", "lane_change"]
n = [idm[f]["agents"] for f in fams]; tot = idm["POOLED"]["agents"]
fig, ax = plt.subplots(figsize=(6.4, 2.4))
ax.barh(range(len(fams)), n, color=[BLUE] + [LIGHT] * 4)
for i, v in enumerate(n):
    ax.text(v + 60, i, f"{v:,} ({100 * v / tot:.0f}%)", va="center", fontsize=9)
ax.set_yticks(range(len(fams)), [NAMES[f] for f in fams]); ax.invert_yaxis(); ax.set_xlim(0, 7000)
ax.set_xlabel(f"moving vehicles (total {tot:,}; 2,000 WOMD v1.1 validation scenes)")
save(fig, "counts.png")

# 2. collisions by cause, default IDM
fams = ["vru_interaction", "intersection", "lane_change", "car_following", "other", "POOLED"]
ahead = [100 * idm[f]["sim_rate_hit_ahead"] for f in fams]
by_sim = [100 * idm[f]["sim_rate_struck_behind_by_sim"] for f in fams]
by_log = [100 * idm[f]["sim_rate_struck_behind_by_log"] for f in fams]
real_ahead = [100 * idm[f]["real_rate_hit_ahead"] for f in fams]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
y = np.arange(len(fams))
ax.barh(y, ahead, color=BLUE, label="simulated: ran into something ahead")
ax.barh(y, by_sim, left=ahead, color="#a9c3e6", label="simulated: struck from behind by another simulated car")
ax.barh(y, by_log, left=np.add(ahead, by_sim), color=LIGHT, label="simulated: struck from behind by a replayed road user")
ax.scatter(real_ahead, y, marker="|", s=160, color="k", zorder=3, label="recorded: ran into something ahead")
for i, f in enumerate(fams):
    ax.text(100 * idm[f]["sim_collision_rate"] + 0.3, i, f"{100 * idm[f]['sim_collision_rate']:.1f}%", va="center", fontsize=8.5)
ax.set_yticks(y, [NAMES[f] for f in fams]); ax.invert_yaxis(); ax.set_xlim(0, 16)
ax.set_xlabel("share of moving vehicles whose first collision had this cause (%)")
ax.legend(frameon=False, fontsize=7.8, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
save(fig, "collisions.png")

# 3. four settings by family (delta, omega, R, ran into ahead)
rows = list(csv.DictReader(open(RES / "agent_comparison.csv")))
fams = ["car_following", "lane_change", "intersection", "vru_interaction", "POOLED"]
th = credibility.CRED_THRESHOLDS
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
ypos = np.arange(len(fams)) + np.array([0, 0, 0, 0, 0.35]); h = 0.2
for ax, (m, title) in zip(axes, [("tail_R", "tail ratio R (1 = match, log scale)"), ("delta", "bias δ (1 = match, log scale)"),
                                 ("omega", "scatter ω (0 = match)"), ("sim_ran_into_ahead", "ran into something ahead (%)")]):
    if m == "omega":
        ax.axvspan(0, th[1]["w"], color=BLUE, alpha=0.06, lw=0); ax.axvspan(0, th[4]["w"], color=BLUE, alpha=0.13, lw=0)
    elif m in ("tail_R", "delta"):
        k = "t" if m == "tail_R" else "b"
        ax.axvspan(1 / th[1][k], th[1][k], color=BLUE, alpha=0.06, lw=0); ax.axvspan(1 / th[4][k], th[4][k], color=BLUE, alpha=0.13, lw=0)
    for j, s in enumerate(SETTINGS):
        vals = []
        for f in fams:
            r = next(r for r in rows if r["agent"] == s and r["family"] == f)
            v = float(r[m]); vals.append(100 * v if m == "sim_ran_into_ahead" else v)
        off = (j - 1.5) * h
        if m in ("tail_R", "delta"):
            ax.barh(ypos + off, np.array(vals) - 1, left=1, height=h, color=COL[s], label=s)
        else:
            ax.barh(ypos + off, vals, height=h, color=COL[s], label=s)
    if m in ("tail_R", "delta"):
        ax.set_xscale("log"); ax.axvline(1, color="k", lw=0.7)
        t = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8] if m == "delta" else [0.25, 0.5, 1, 2, 4, 8]
        ax.set_xticks(t, [f"{v:g}" for v in t]); ax.minorticks_off(); ax.set_xlim(0.04 if m == "delta" else 0.2, 8)
    ax.set_title(title, loc="left", fontsize=9.5)
axes[0].set_yticks(ypos, [NAMES[f] for f in fams]); axes[0].invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8.5, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))
save(fig, "comparison.png")

# 4. credibility ladder, car following
cf = {s: next(r for r in rows if r["agent"] == s and r["family"] == "car_following") for s in SETTINGS}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True)
for lvl in range(1, 6):
    b, w = th[lvl]["b"], th[lvl]["w"]
    a1.barh(lvl, np.log(b) * 2, left=-np.log(b), height=0.6, color=BLUE, alpha=0.10 + 0.12 * lvl)
    a2.barh(lvl, w, height=0.6, color=BLUE, alpha=0.10 + 0.12 * lvl)
for s in ["idm_cautious", "idm", "constant_speed"]:
    a1.axvline(np.log(float(cf[s]["delta"])), color=COL[s], lw=1.6, label=f"{s}: δ = {float(cf[s]['delta']):.3f}")
    a2.axvline(float(cf[s]["omega"]), color=COL[s], lw=1.6, label=f"{s}: ω = {float(cf[s]['omega']):.3f}")
t = [0.05, 0.1, 0.25, 0.5, 1, 2]
a1.set_xticks(np.log(t), [f"{v:g}" for v in t]); a1.set_xlim(np.log(0.04), np.log(2.6)); a1.set_xlabel("bias δ (log scale)")
a2.set_xlim(0, 2.6); a2.set_xlabel("scatter ω")
a1.set_yticks(range(1, 6), [f"level {i}" for i in range(1, 6)]); a1.invert_yaxis()
a1.legend(frameon=False, fontsize=7.5, loc="lower left"); a2.legend(frameon=False, fontsize=7.5, loc="lower right")
save(fig, "ladder.png")

# 5. Rule 2 weight vs omega
pooled = {s: float(next(r for r in rows if r["agent"] == s and r["family"] == "POOLED")["omega"]) for s in SETTINGS}
w = np.linspace(0, 2.6, 300)
fig, ax = plt.subplots(figsize=(6.2, 2.8))
ax.plot(w, evidence.a_from_omega(w), color=GREY)
for s in ["idm_cautious", "idm", "constant_speed"]:
    a = evidence.a_from_omega(pooled[s])
    dy = {"idm_cautious": 30, "idm": 16, "constant_speed": 8}[s]
    ax.plot(pooled[s], a, "o", color=COL[s]); ax.annotate(f"{s}: ω = {pooled[s]:.2f}, a = {a:.3f}", (pooled[s], a), xytext=(-10, dy), textcoords="offset points", fontsize=8.5)
ax.set_xlabel("pooled scatter ω of the setting"); ax.set_ylabel("weight a on simulated exposure")
save(fig, "aomega.png")

# 6. power-prior worked example
cases = [("road only (a = 0)", 6.5, 271.3, GREY), ("sim corrected by R, a = 0.35", 0.35 * 60 + 6.5, 0.35 * 2713 + 271.3, BLUE),
         ("sim corrected by R, a = 1", 60 + 6.5, 2713 + 271.3, "#7fa7da"), ("sim not corrected, a = 0.35", 0.35 * 240 + 6.5, 0.35 * 2713 + 271.3, RUST)]
x = np.linspace(0, 0.11, 600)
fig, ax = plt.subplots(figsize=(6.6, 2.9))
for lab, sh, rate, c in cases:
    g = stats.gamma(sh, scale=1 / rate)
    ax.plot(x, g.pdf(x), color=c, lw=1.6, label=f"{lab}: mean {g.mean():.4f}")
ax.set_xlabel("event rate λ (events per unit of exposure)"); ax.set_yticks([]); ax.set_ylabel("posterior density")
ax.legend(frameon=False, fontsize=8)
save(fig, "posterior.png")

# 7. synthetic pilot
pil = json.load(open(RES / "pilot_results.json"))
fams = [r["family"] for r in pil["partA"]]
fig, ax = plt.subplots(figsize=(6.2, 2.5))
for i, r in enumerate(pil["partA"]):
    ax.plot(r["true_delta"], i, "o", mfc="white", mec=GREY, ms=8); ax.plot(r["delta"], i, "o", color=BLUE, ms=5)
    ax.annotate(f"{r['delta']:.3f}", (r["delta"], i), xytext=(10, -3), textcoords="offset points", fontsize=8.5)
ax.plot(pil["pooled"]["delta"], len(fams), "o", color=BLUE, ms=5)
ax.annotate(f"{pil['pooled']['delta']:.3f}", (pil["pooled"]["delta"], len(fams)), xytext=(10, -3), textcoords="offset points", fontsize=8.5)
ax.set_yticks(range(len(fams) + 1), [NAMES[f] for f in fams] + ["Pooled"]); ax.invert_yaxis()
ax.set_xlabel("bias δ: planted (open circle) and recovered (dot)"); ax.set_xlim(0.9, 1.75)
save(fig, "synthetic.png")
