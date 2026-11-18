"""Step 1: rule-based scenario typing.

Rules work on a small set of per-scene features, so the same code can serve other trajectory datasets.
Thresholds are explicit, documented assumptions; change them in RULES and report the change.
"""
from dataclasses import dataclass

FAMILIES = ("vru_interaction", "intersection", "lane_change", "car_following", "other")


@dataclass
class SceneFeatures:
    vru_within_m: float          # closest pedestrian/cyclist distance to the subject vehicle over the scene (m)
    in_intersection: bool        # subject vehicle path enters a mapped intersection
    lateral_shift_m: float       # largest lateral move of the subject (or a cut-in agent) across a lane centre (m)
    lead_vehicle_present: bool   # a same-lane vehicle ahead for most of the scene


RULES = {
    "vru_distance_m": 15.0,      # a pedestrian or cyclist within 15 m counts as an interaction
    "lane_change_shift_m": 2.5,  # about two-thirds of a 3.7 m lane width
}


def classify(f: SceneFeatures, rules: dict = RULES) -> str:
    """Return one primary family, in priority order: VRU interaction > intersection > lane change > car following."""
    if f.vru_within_m <= rules["vru_distance_m"]:
        return "vru_interaction"
    if f.in_intersection:
        return "intersection"
    if f.lateral_shift_m >= rules["lane_change_shift_m"]:
        return "lane_change"
    if f.lead_vehicle_present:
        return "car_following"
    return "other"
