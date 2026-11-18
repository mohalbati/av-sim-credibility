"""Per-agent safety measures from Waymo Open Motion Dataset trajectories (pure NumPy; no Waymax import).

Arrays are (N objects, T steps) at 10 Hz, as in Waymax trajectories: x, y, yaw, vel_x, vel_y, length, width, valid.
Object types follow Waymax/WOMD ids: 1 vehicle, 2 pedestrian, 3 cyclist.

min_ttc_per_follower: for every object i, the smallest 1-D time-to-collision to an object j that is
  - ahead of i (positive longitudinal offset in i's frame),
  - laterally overlapping i's path (|lateral offset| < half-widths summed),
  - heading the same way (|yaw difference| < 45 deg) if j is a vehicle; any heading for pedestrians/cyclists,
  TTC = bumper gap / closing speed along i's heading, defined only before contact (gap > 0); inf if not closing.
  Contacts (collisions) are measured separately with Waymax's rotated-box overlap check.
  This is a car-following-type TTC. It does not capture crossing conflicts (PET would; later step).
"""
import numpy as np

TYPE_VEH, TYPE_PED, TYPE_CYC = 1, 2, 3


def min_ttc_per_follower(x, y, yaw, vx, vy, L, W, valid, types, max_ttc=10.0, max_yaw_diff=np.pi / 4):
    """Returns (min_ttc[N], lead_type_at_min[N], lead_seen[N]). L, W are per-object (N,)."""
    N, T = x.shape
    out = np.full(N, np.inf); lead_type = np.zeros(N, int); seen = np.zeros(N, bool)
    eye = np.eye(N, dtype=bool); half_L = (L[:, None] + L[None, :]) / 2; half_W = (W[:, None] + W[None, :]) / 2
    not_vehicle = (types != TYPE_VEH)[None, :]
    for t in range(T):
        c, s = np.cos(yaw[:, t]), np.sin(yaw[:, t])
        dx = x[None, :, t] - x[:, None, t]; dy = y[None, :, t] - y[:, None, t]       # [i, j]: j relative to i
        lon = dx * c[:, None] + dy * s[:, None]
        lat = -dx * s[:, None] + dy * c[:, None]
        dyaw = np.abs(np.angle(np.exp(1j * (yaw[None, :, t] - yaw[:, None, t]))))
        ok = (valid[:, None, t] & valid[None, :, t] & ~eye & (lon > 0) & (np.abs(lat) < half_W)
              & ((dyaw < max_yaw_diff) | not_vehicle))
        seen |= ok.any(1)
        vi = vx[:, t] * c + vy[:, t] * s
        vj = vx[None, :, t] * c[:, None] + vy[None, :, t] * s[:, None]
        closing = vi[:, None] - vj; gap = lon - half_L
        with np.errstate(divide="ignore", invalid="ignore"):
            ttc = np.where((gap > 0) & (closing > 1e-6), gap / closing, np.inf)   # contacts are counted separately
        ttc = np.where(ok, ttc, np.inf)
        j = np.argmin(ttc, axis=1); m = ttc[np.arange(N), j]
        better = m < out
        out[better] = m[better]; lead_type[better] = types[j[better]]
    out[out > max_ttc] = np.inf
    return out, lead_type, seen


def agent_features(i, x, y, yaw, valid, types):
    """Log-based features for scenario-family rules: lateral shift (lane change), turning (intersection proxy),
    closest distance to a pedestrian or cyclist."""
    idx = np.where(valid[i])[0]
    x0, y0, h0 = x[i, idx[0]], y[i, idx[0]], yaw[i, idx[0]]
    dx = x[i, idx] - x0; dy = y[i, idx] - y0
    lat = -dx * np.sin(h0) + dy * np.cos(h0)
    dyaw = float(np.abs(np.angle(np.exp(1j * (yaw[i, idx] - h0)))).max())
    vru = np.where(np.isin(types, [TYPE_PED, TYPE_CYC]))[0]
    d_vru = np.inf
    for k in vru:
        both = valid[i] & valid[k]
        if both.any():
            d_vru = min(d_vru, float(np.min(np.hypot(x[i, both] - x[k, both], y[i, both] - y[k, both]))))
    return {"lateral_shift_m": float(np.abs(lat).max()) if dyaw < np.radians(20) else 0.0,
            "turning": dyaw > np.radians(45), "vru_within_m": d_vru}


def first_collision(ov, x, y, yaw, reactive):
    """Attribute each object's FIRST collision in the window.

    ov: (T, N, N) pairwise box overlaps; x, y, yaw: (N, T); reactive: (N,) True for simulator-controlled objects.
    Returns (where[N], partner[N]):
      where   'none' | 'ahead' (the partner's centre is in front of i: i ran into it) | 'behind' (i was struck from behind)
      partner '' | 'reactive' (another simulated agent) | 'log_replay' (an agent replaying its log, which cannot react)
    'behind' + 'log_replay' is a set-up artifact: a non-reactive replayed agent drove into a simulated vehicle that slowed."""
    T, N, _ = ov.shape
    where = np.array(["none"] * N, dtype=object); partner = np.array([""] * N, dtype=object)
    for i in range(N):
        ts, js = np.nonzero(ov[:, i, :])            # row-major: sorted by time
        if ts.size == 0:
            continue
        t, j = ts[0], js[0]
        lon = (x[j, t] - x[i, t]) * np.cos(yaw[i, t]) + (y[j, t] - y[i, t]) * np.sin(yaw[i, t])
        where[i] = "ahead" if lon > 0 else "behind"
        partner[i] = "reactive" if reactive[j] else "log_replay"
    return where, partner
