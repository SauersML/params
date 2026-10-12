"""Absolute output qualification for auxiliary reward, independent of the start's quality.

This prevents bonuses for insufficient outputs. It does not establish an ordering
between insufficient circuits or replace intervention qualification.
"""

import math

POINT_TARGET = 0.9
GATE_SLOPE = 4.0


def qualify(point_score: float | None, phi: float | None, phi_tau: float | None,
            dominance: float | None = 0.0, slope: float = GATE_SLOPE, floor: float = POINT_TARGET) -> dict:
    """Qualification and the existing smooth phi gate, with explicit failure reasons.

    g applies to quality bonuses; dominance_gate applies to the separate dominance
    bonus. Both are zero unless point recovery reaches the absolute target and the
    phi gate is defined. Noise and a weak start cannot lower the point target.
    """
    if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError("point target must be finite and between zero and one")
    if not math.isfinite(slope) or slope <= 0.0:
        raise ValueError("gate slope must be finite and positive")
    reasons = []
    if point_score is None or not math.isfinite(point_score):
        reasons.append("point_unresolved")
    elif point_score < floor:
        reasons.append("point_insufficient")
    elif point_score > 1.0:
        reasons.append("point_invalid")
    phi_gate = None
    if phi is None or phi_tau is None or not math.isfinite(phi) or not math.isfinite(phi_tau):
        reasons.append("phi_gate_unresolved")
    else:
        # Branch on the sign so exp never sees a positive exponent. Overflow of
        # a finite difference/product to infinity correctly saturates the gate.
        logit = math.log(2.0) * slope * (phi - phi_tau)
        if logit >= 0.0:
            phi_gate = 1.0 / (1.0 + math.exp(-logit))
        else:
            exp_logit = math.exp(logit)
            phi_gate = exp_logit / (1.0 + exp_logit)
    eligible = not reasons
    return {"eligible": eligible, "reasons": reasons, "point_target": floor,
            "phi_gate": phi_gate, "g": phi_gate if eligible else 0.0,
            "dominance_gate": 1.0 if eligible else 0.0,
            "D": dominance if eligible and dominance is not None and math.isfinite(dominance) else 0.0,
            "faithful": eligible and phi >= phi_tau}


def gated_auxiliary(gate: dict, quality: float) -> float:
    """g Q + D after qualification; reject all auxiliary reward when ineligible."""
    if not gate["eligible"]:
        return 0.0
    return gate["g"] * quality + gate["D"]
