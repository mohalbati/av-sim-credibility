"""Tests against known answers. Run: python tests/test_core.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy import stats
from avcred import measures, scenarios, validation, credibility, evidence, womd

rng = np.random.default_rng(1)


def test_ttc_drac():
    assert np.isclose(measures.time_to_collision(20, 25, 15), 2.0)
    assert np.isclose(measures.drac(20, 25, 15), 2.5)
    assert np.isinf(measures.time_to_collision(20, 15, 25))          # opening gap
    assert measures.time_to_collision(0, 10, 5) == 0.0               # contact


def test_area_metric_known():
    assert np.isclose(validation.area_metric([1.0], [3.0]), 2.0)     # two point masses 2 apart
    a = rng.normal(0, 1, 200_000); b = rng.normal(0.5, 1, 200_000)
    assert abs(validation.area_metric(a, b) - 0.5) < 0.01            # pure shift: area = shift
    assert validation.area_metric(a, a) == 0.0
    lo, hi = validation.signed_area(a, b)
    assert lo > 0.49 and hi < 0.01                                   # sim (a) sits left of real (b)


def test_bias_uncertainty_recovers_truth():
    real = rng.lognormal(0, 0.5, 4000)
    sim = real * np.exp(rng.normal(np.log(1.5), 0.2, real.size))
    r = validation.bias_uncertainty(sim, real, n_boot=500)
    assert abs(r["delta"] - 1.5) < 0.02 and abs(r["omega"] - 0.2) < 0.01
    assert r["delta_ci"][0] < 1.5 < r["delta_ci"][1]


def test_tail_ratio_same_distribution():
    y1 = rng.gamma(4, 0.8, 20_000); y2 = rng.gamma(4, 0.8, 20_000)
    r = validation.tail_ratio(y1, y2, tau=0.8, n_boot=300)
    assert r["ci"][0] < 1 < r["ci"][1]


def test_gpd_lower_tail():
    y = rng.gamma(4, 0.8, 50_000)
    true = stats.gamma.cdf(0.5, 4, scale=0.8)
    est = validation.tail_prob_gpd(y, 0.5, q=0.05)
    assert 0.5 < est / true < 2.0                                     # extrapolation within a factor of 2


def test_credibility_levels():
    cou = credibility.ContextOfUse("cut-in conflict rate", "high", "high")
    assert credibility.model_risk(cou) == 5
    assert credibility.achieved_level(1.05, 0.1, 1.1) == 5
    assert credibility.achieved_level(2.5, 0.1, 1.0) == 0             # bias outside every level
    assert credibility.assess(cou, 1.3, 0.3, 1.8)["decision"] == "do not credit"


def test_evidence_power_prior():
    p0 = evidence.posterior(6, 271.3, a=0.0)
    assert np.isclose(p0.mean(), 6.5 / 271.3)                         # a = 0 -> Jeffreys road-only posterior
    p1 = evidence.posterior(6, 271.3, k_sim=60, M_sim=2713, a=1.0)
    assert p1.std() < p0.std()                                        # simulation narrows the interval
    # consistent simulation -> high weight; conflicting simulation -> low weight
    a_ok = evidence.a_empirical(6, 271.3, k_sim=60, M_sim=2713)       # same rate 0.022
    a_bad = evidence.a_empirical(6, 271.3, k_sim=600, M_sim=2713)     # 10x the rate
    assert a_ok > 0.5 and a_bad < 0.05
    assert evidence.a_from_omega(0.3) == 0.5


def test_scenario_rules():
    f = scenarios.SceneFeatures(vru_within_m=40, in_intersection=False, lateral_shift_m=3.0, lead_vehicle_present=True)
    assert scenarios.classify(f) == "lane_change"
    f.vru_within_m = 5
    assert scenarios.classify(f) == "vru_interaction"


def test_womd_ttc():
    t = np.arange(10) * 0.1; z = np.zeros((3, 10))
    x = np.vstack([20 * t, 30 + 10 * t, 100 + 0 * t]); y = np.vstack([0 * t, 0.5 + 0 * t, 10 + 0 * t])
    vx = np.vstack([20 + 0 * t, 10 + 0 * t, 0 * t])
    ttc, lead, seen = womd.min_ttc_per_follower(x, y, z, vx, z, np.array([5, 5, 1.]), np.array([2, 2, 1.]),
                                               np.ones((3, 10), bool), np.array([1, 1, 2]))
    assert np.isclose(ttc[0], 1.6) and lead[0] == 1 and np.isinf(ttc[1])   # gap 25 - 10*0.9 = 16 m at 10 m/s


def test_first_collision():
    T, N = 3, 3
    ov = np.zeros((T, N, N), bool); ov[1, 0, 1] = ov[1, 1, 0] = True       # objects 0 and 1 touch at t = 1
    x = np.array([[0, 0, 0], [5, 4, 4], [50, 50, 50.]]); y = np.zeros((N, T)); yaw = np.zeros((N, T))
    where, partner = womd.first_collision(ov, x, y, yaw, reactive=np.array([True, False, True]))
    assert where[0] == "ahead" and partner[0] == "log_replay"              # 0 ran into the replayed object 1
    assert where[1] == "behind" and partner[1] == "reactive" and where[2] == "none"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t(); print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
