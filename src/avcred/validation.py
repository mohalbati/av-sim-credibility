"""Step 4 (Part A): validation metrics between simulated and real values of a safety measure.

- area_metric:        area between the two empirical CDFs (Ferson, Oberkampf & Ginzburg, 2008), in the measure's units
- bias_uncertainty:   paired sim/real values for the same scene; ln(sim/real) ~ N(ln delta, omega^2)
                      (the bias / relative-uncertainty idea of NUREG-1824 Supp. 1, as used in the author's thesis, Ch. 3)
- tail_ratio:         R = P_sim(Y < tau) / P_real(Y < tau), empirical or generalized-Pareto tail
"""
import numpy as np
from scipy import stats


def area_metric(sim, real):
    """Exact area between the empirical CDFs of two samples: integral of |F_sim(y) - F_real(y)| dy."""
    s = np.sort(np.asarray(sim, float)); r = np.sort(np.asarray(real, float))
    s = s[np.isfinite(s)]; r = r[np.isfinite(r)]
    grid = np.unique(np.concatenate([s, r]))
    if grid.size < 2:
        return 0.0
    Fs = np.searchsorted(s, grid[:-1], side="right") / s.size
    Fr = np.searchsorted(r, grid[:-1], side="right") / r.size
    return float(np.sum(np.abs(Fs - Fr) * np.diff(grid)))


def signed_area(sim, real):
    """Split the area metric by direction: (area where the simulation sits left of reality, area where it sits right)."""
    s = np.sort(np.asarray(sim, float)); r = np.sort(np.asarray(real, float))
    grid = np.unique(np.concatenate([s, r]))
    Fs = np.searchsorted(s, grid[:-1], side="right") / s.size
    Fr = np.searchsorted(r, grid[:-1], side="right") / r.size
    w = np.diff(grid)
    return float(np.sum(np.clip(Fs - Fr, 0, None) * w)), float(np.sum(np.clip(Fr - Fs, 0, None) * w))


def bias_uncertainty(sim, real, n_boot=2000, seed=0, conf=0.95):
    """Paired values (same scene): delta = exp(mean ln(sim/real)), omega = SD of ln(sim/real), with bootstrap intervals.

    delta = 1 means unbiased; delta = 2 means the simulation predicts about twice the real value.
    omega is the scatter of the model error (a relative uncertainty)."""
    sim = np.asarray(sim, float); real = np.asarray(real, float)
    ok = (sim > 0) & (real > 0) & np.isfinite(sim) & np.isfinite(real)
    lr = np.log(sim[ok] / real[ok])
    if lr.size < 3:
        raise ValueError("need at least 3 valid pairs")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, lr.size, (n_boot, lr.size))
    bm = lr[idx].mean(1); bs = lr[idx].std(1, ddof=1)
    a = (1 - conf) / 2
    return {"n": int(lr.size), "delta": float(np.exp(lr.mean())), "omega": float(lr.std(ddof=1)),
            "delta_ci": tuple(np.exp(np.quantile(bm, [a, 1 - a]))), "omega_ci": tuple(np.quantile(bs, [a, 1 - a]))}


def tail_prob_empirical(y, tau):
    """P(Y < tau) with a Jeffreys-style correction so zero counts don't give zero: (k + 1/2) / (n + 1)."""
    y = np.asarray(y, float)
    return (np.sum(y < tau) + 0.5) / (y.size + 1)


def tail_prob_gpd(y, tau, q=0.10):
    """P(Y < tau) for a LOWER tail (e.g. small TTC) from a generalized Pareto fit to the lowest q fraction.

    Fits GPD to u - y for y below the q-quantile u, then P(Y < tau) = q * P(u - Y > u - tau | Y < u)."""
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    u = np.quantile(y, q)
    exc = u - y[y < u]
    if tau >= u:
        return float(np.mean(y < tau))
    if exc.size < 10:
        raise ValueError("too few tail points for a GPD fit")
    xi, _, sig = stats.genpareto.fit(exc, floc=0)
    return float(q * stats.genpareto.sf(u - tau, xi, 0, sig))


def tail_ratio(sim, real, tau, method="empirical", n_boot=1000, seed=0, conf=0.95):
    """R = P_sim(Y < tau) / P_real(Y < tau) with a bootstrap interval. R < 1: the simulation under-represents the tail."""
    f = tail_prob_empirical if method == "empirical" else tail_prob_gpd
    sim = np.asarray(sim, float); real = np.asarray(real, float)
    R = f(sim, tau) / f(real, tau)
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_boot):
        try:
            out.append(f(rng.choice(sim, sim.size), tau) / f(rng.choice(real, real.size), tau))
        except ValueError:
            continue
    a = (1 - conf) / 2
    return {"R": float(R), "ci": tuple(np.quantile(out, [a, 1 - a])) if out else (np.nan, np.nan)}


def validate_family(sim_scene, real_scene, tau, n_boot=1000, seed=0):
    """All Part-A metrics for one scenario family, from per-scene values (paired by index)."""
    sim_scene = np.asarray(sim_scene, float); real_scene = np.asarray(real_scene, float)
    left, right = signed_area(sim_scene, real_scene)
    return {"area": area_metric(sim_scene, real_scene), "area_sim_low": left, "area_sim_high": right,
            **{f"bias_{k}": v for k, v in bias_uncertainty(sim_scene, real_scene, n_boot, seed).items()},
            **{f"tail_{k}": v for k, v in tail_ratio(sim_scene, real_scene, tau, n_boot=n_boot, seed=seed).items()}}
