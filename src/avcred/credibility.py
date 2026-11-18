"""Step 5 (Part B): risk-informed credibility scoring, adapted from the logic of ASME V&V 40-2018.

V&V 40 sets the credibility a model needs from its *model risk* = model influence x decision consequence,
then compares that with the credibility achieved. The numerical thresholds below are this project's
assumptions (stated, changeable), not values from the standard.
"""
from dataclasses import dataclass

LEVELS = {"low": 1, "medium": 2, "high": 3}

# model risk (1..5) from influence x consequence, as in the V&V 40 risk-matrix pattern
RISK = {(1, 1): 1, (1, 2): 2, (1, 3): 3, (2, 1): 2, (2, 2): 3, (2, 3): 4, (3, 1): 3, (3, 2): 4, (3, 3): 5}

# validation thresholds an achieved credibility level must meet (assumed; report them)
#   delta within [1/b, b], omega below w, tail ratio R within [1/t, t]
CRED_THRESHOLDS = {
    1: {"b": 2.0, "w": 1.0, "t": 5.0},
    2: {"b": 1.5, "w": 0.6, "t": 3.0},
    3: {"b": 1.25, "w": 0.4, "t": 2.0},
    4: {"b": 1.15, "w": 0.25, "t": 1.5},
    5: {"b": 1.10, "w": 0.15, "t": 1.25},
}


@dataclass
class ContextOfUse:
    question: str          # e.g. "conflict rate in cut-ins for release gating"
    influence: str         # low: one of several evidence sources; high: simulation is the main evidence
    consequence: str       # low: comfort; high: serious-injury outcome


def model_risk(cou: ContextOfUse) -> int:
    return RISK[(LEVELS[cou.influence], LEVELS[cou.consequence])]


def achieved_level(delta, omega, tail_R, applicability_ok=True) -> int:
    """Highest level whose thresholds are all met (0 if none). Applicability (validation scenes cover the use) is required."""
    if not applicability_ok:
        return 0
    lvl = 0
    for L in sorted(CRED_THRESHOLDS):
        th = CRED_THRESHOLDS[L]
        within = lambda x, b: (1 / b) <= x <= b
        if within(delta, th["b"]) and omega <= th["w"] and within(tail_R, th["t"]):
            lvl = L
    return lvl


def assess(cou: ContextOfUse, delta, omega, tail_R, applicability_ok=True) -> dict:
    """Required vs achieved credibility and the resulting decision for one scenario family."""
    req = model_risk(cou); ach = achieved_level(delta, omega, tail_R, applicability_ok)
    if ach >= req:
        decision = "credit"
    elif ach >= req - 1 and ach > 0:
        decision = "credit with model-error correction"
    else:
        decision = "do not credit"
    return {"question": cou.question, "required": req, "achieved": ach, "decision": decision}
