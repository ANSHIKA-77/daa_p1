# dynamic_weights.py

DEFAULT_WEIGHTS = {
    "stability":             0.4,
    "response_time":         0.3,
    "resource_efficiency":   0.2,
    "excess_alert_frequency": 0.1,
}

# Thresholds
ICU_BED_LOW_THRESHOLD    = 0.2   # <= 20% beds free → critical resource pressure
ICU_BED_MEDIUM_THRESHOLD = 0.5   # <= 50% beds free → moderate pressure

ALERT_FATIGUE_HIGH  = 5
ALERT_FATIGUE_MED   = 2

RISK_LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}


def _clamp(val, lo=0.0, hi=1.0):
    return max(lo, min(hi, val))


def adjust_weights(
    icu_bed_availability: float,   # 0.0 – 1.0  (fraction of beds free)
    patient_risk_label: str,       # "Low" | "Medium" | "High"
    alert_frequency: int,          # raw alert count for this patient
    base_weights: dict = None,
) -> dict:
   
    w = dict(base_weights) if base_weights else dict(DEFAULT_WEIGHTS)

    risk_level = RISK_LABEL_MAP.get(patient_risk_label, 0)

    # --- ICU bed pressure ---
    if icu_bed_availability <= ICU_BED_LOW_THRESHOLD:
        w["resource_efficiency"]   = _clamp(w["resource_efficiency"]   + 0.10)
        w["response_time"]         = _clamp(w["response_time"]         - 0.05)
        w["stability"]             = _clamp(w["stability"]             - 0.05)
    elif icu_bed_availability <= ICU_BED_MEDIUM_THRESHOLD:
        w["resource_efficiency"]   = _clamp(w["resource_efficiency"]   + 0.05)
        w["stability"]             = _clamp(w["stability"]             - 0.05)

    # --- Patient risk level ---
    if risk_level == 2:       # High
        w["response_time"]               = _clamp(w["response_time"]               + 0.10)
        w["excess_alert_frequency"]      = _clamp(w["excess_alert_frequency"]      - 0.05)
        w["stability"]                   = _clamp(w["stability"]                   - 0.05)
    elif risk_level == 1:     # Medium
        w["response_time"]               = _clamp(w["response_time"]               + 0.05)
        w["stability"]                   = _clamp(w["stability"]                   - 0.05)

    # --- Alert fatigue ---
    if alert_frequency >= ALERT_FATIGUE_HIGH:
        w["excess_alert_frequency"]  = _clamp(w["excess_alert_frequency"]  + 0.10)
        w["response_time"]           = _clamp(w["response_time"]           - 0.05)
        w["stability"]               = _clamp(w["stability"]               - 0.05)
    elif alert_frequency >= ALERT_FATIGUE_MED:
        w["excess_alert_frequency"]  = _clamp(w["excess_alert_frequency"]  + 0.05)
        w["stability"]               = _clamp(w["stability"]               - 0.05)

    # --- Renormalize so weights always sum to 1.0 ---
    total = sum(w.values())
    if total > 0:
        w = {k: round(v / total, 6) for k, v in w.items()}

    return w


def get_weights_for_patient(
    patient_id: int,
    icu_bed_availability: float,
    risk_label: str,
    alert_fatigue_map: dict,
    base_weights: dict = None,
) -> dict:
    """Convenience wrapper used by patient_prioritization.py."""
    af = alert_fatigue_map.get(patient_id, 0)
    return adjust_weights(
        icu_bed_availability=icu_bed_availability,
        patient_risk_label=risk_label,
        alert_frequency=af,
        base_weights=base_weights,
    )


# ── quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scenarios = [
        (0.15, "High",   6),
        (0.40, "Medium", 2),
        (0.80, "Low",    0),
        (0.10, "High",   0),
        (0.60, "Low",    8),
    ]

    print(f"{'ICU':>6} {'Risk':<8} {'AF':>4}  "
          f"{'stab':>6} {'resp':>6} {'res':>6} {'alert':>6}")
    print("-" * 52)
    for icu, risk, af in scenarios:
        w = adjust_weights(icu, risk, af)
        print(f"{icu:>6.0%} {risk:<8} {af:>4}  "
              f"{w['stability']:>6.3f} {w['response_time']:>6.3f} "
              f"{w['resource_efficiency']:>6.3f} "
              f"{w['excess_alert_frequency']:>6.3f}")