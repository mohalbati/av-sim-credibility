"""Step 3: safety-relevant measures from trajectories.

All functions work on NumPy arrays sampled at a fixed time step (WOMD and Waymax use 10 Hz, dt = 0.1 s).
Distances in metres, speeds in m/s, accelerations in m/s^2, times in seconds.
"""
import numpy as np


def time_to_collision(gap, v_follow, v_lead):
    """TTC = gap / closing speed, defined only while the follower is faster (closing > 0); otherwise +inf."""
    gap = np.asarray(gap, float)
    closing = np.asarray(v_follow, float) - np.asarray(v_lead, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where((closing > 0) & (gap > 0), gap / closing, np.inf)
    return np.where(gap <= 0, 0.0, ttc)            # already in contact -> TTC 0


def drac(gap, v_follow, v_lead):
    """Deceleration rate to avoid collision: closing^2 / (2 gap) while closing > 0, else 0."""
    gap = np.asarray(gap, float)
    closing = np.asarray(v_follow, float) - np.asarray(v_lead, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((closing > 0) & (gap > 0), closing ** 2 / (2 * gap), 0.0)
    return np.where(gap <= 0, np.inf, out)


def time_headway(gap, v_follow):
    """Time headway = gap / follower speed (inf when the follower is stopped)."""
    v = np.asarray(v_follow, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, np.asarray(gap, float) / v, np.inf)


def peak_deceleration(speed, dt=0.1):
    """Largest deceleration (positive number) from a speed trace."""
    a = np.diff(np.asarray(speed, float)) / dt
    return float(max(0.0, -a.min())) if a.size else 0.0


def post_encroachment_time(t_first_leaves, t_second_enters):
    """PET for crossing paths: time between the first road user leaving the conflict area and the second entering."""
    return float(t_second_enters - t_first_leaves)


def scene_summary(gap, v_follow, v_lead, dt=0.1):
    """Per-scene safety summary for a following pair: minimum TTC, minimum headway, maximum DRAC, peak deceleration."""
    return {
        "min_ttc": float(np.min(time_to_collision(gap, v_follow, v_lead))),
        "min_headway": float(np.min(time_headway(gap, v_follow))),
        "max_drac": float(np.max(drac(gap, v_follow, v_lead))),
        "peak_decel": peak_deceleration(v_follow, dt),
    }
