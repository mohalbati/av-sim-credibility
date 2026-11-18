"""Step 6 (Part B): from validation to a safety estimate.

A simulation estimate for scenario family s (k_sim events in M_sim of simulated exposure) is
(1) corrected by the validated tail ratio R and (2) used as a power prior with weight a in [0, 1]:

    lambda | data ~ Gamma(a * k_sim_corr + k_road + 1/2,  a * M_sim + M_road)

a = 0 ignores simulation (Jeffreys posterior on road data only); a = 1 treats simulation like road data.
Simulation credit = a * M_sim, the road-equivalent exposure simulation is worth.

Three candidate rules for a are provided and compared, not assumed:
  a_from_level   - credibility level from credibility.py mapped to a weight (assumed mapping)
  a_from_omega   - weight shrinks as the validated relative uncertainty omega grows (assumed functional form)
  a_empirical    - empirical-Bayes power prior: a maximizing the marginal likelihood of the road data
                   (Gravestock & Held, 2017), i.e. how consistent the road data are with the simulation
"""
import numpy as np
from scipy import stats, special


def corrected_sim_events(k_sim, R):
    """Correct simulated event counts for under/over-representation of the tail: k / R."""
    return k_sim / R


def posterior(k_road, M_road, k_sim=0.0, M_sim=0.0, a=0.0):
    """Gamma posterior (shape, rate) for an event rate with a Jeffreys base prior and a power prior from simulation."""
    shape = a * k_sim + k_road + 0.5
    rate = a * M_sim + M_road
    return stats.gamma(shape, scale=1 / rate)


def summarize(post, conf=0.95):
    a = (1 - conf) / 2
    lo, med, hi = post.ppf([a, 0.5, 1 - a])
    return {"mean": float(post.mean()), "median": float(med), "lo": float(lo), "hi": float(hi)}


def simulation_credit(a, M_sim):
    return a * M_sim


LEVEL_TO_A = {0: 0.0, 1: 0.05, 2: 0.15, 3: 0.35, 4: 0.6, 5: 0.9}   # assumed mapping, reported and varied


def a_from_level(level):
    return LEVEL_TO_A[int(level)]


def a_from_omega(omega, omega_ref=0.3):
    """Assumed form: a = 1 / (1 + (omega / omega_ref)^2); a = 0.5 when omega equals omega_ref."""
    return 1.0 / (1.0 + (omega / omega_ref) ** 2)


def log_marginal_road(k_road, M_road, k_sim, M_sim, a):
    """log P(k_road | a): Poisson road count under the Gamma(a k_sim + 1/2, a M_sim) prior -> negative binomial.
    For a = 0 the prior is improper, so a is searched on (0, 1]."""
    al = a * k_sim + 0.5; be = a * M_sim
    return (special.gammaln(k_road + al) - special.gammaln(al) - special.gammaln(k_road + 1)
            + al * np.log(be / (be + M_road)) + k_road * np.log(M_road / (be + M_road)))


def a_empirical(k_road, M_road, k_sim, M_sim, grid=None):
    """Empirical-Bayes power prior: a in (0, 1] maximizing the road data's marginal likelihood."""
    grid = np.linspace(0.001, 1.0, 1000) if grid is None else grid
    ll = np.array([log_marginal_road(k_road, M_road, k_sim, M_sim, a) for a in grid])
    return float(grid[int(np.argmax(ll))])
