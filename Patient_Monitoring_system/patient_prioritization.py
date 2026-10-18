# patient_prioritization.py

import heapq
import time
import glob
import os
import sys

# Ensure safe UTF-8 output on Windows console
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from data_loader      import CSVPatient
from random_generator import RandomPatient
from risk_prediction  import RiskPredictor
from dynamic_weights  import get_weights_for_patient

# ── Bayesian constants (mirrored from main.py) ────────────────────────────────
PRIOR_CRITICAL                      = 0.20
LIKELIHOOD_HIGH_RISK_GIVEN_CRITICAL = 0.85
LIKELIHOOD_HIGH_RISK_GIVEN_NOT_CRIT = 0.30

# ── ICU environment (update at runtime or inject from hospital system) ────────
ICU_BED_AVAILABILITY = 0.40   # 40 % beds free 

# ── Shared state ──────────────────────────────────────────────────────────────
alert_fatigue: dict[int, int] = {}


# ── Bayesian update ───────────────────────────────────────────────────────────
def bayesian_update(is_high_risk: bool) -> float:
    if is_high_risk:
        lk   = LIKELIHOOD_HIGH_RISK_GIVEN_CRITICAL
        ev   = (lk * PRIOR_CRITICAL
                + LIKELIHOOD_HIGH_RISK_GIVEN_NOT_CRIT * (1 - PRIOR_CRITICAL))
    else:
        lk   = 1 - LIKELIHOOD_HIGH_RISK_GIVEN_CRITICAL
        ev   = (lk * PRIOR_CRITICAL
                + (1 - LIKELIHOOD_HIGH_RISK_GIVEN_NOT_CRIT) * (1 - PRIOR_CRITICAL))
    return (lk * PRIOR_CRITICAL) / ev


# ── Clinical scores ───────────────────────────────────────────────────────────
def qsofa(v) -> int:
    return (int(v["resp_rate"] >= 22)
            + int(v["sbp"]       <= 100)
            + int(v["consciousness"] == 1))


def news2(v) -> int:
    s = 0
    rr = v["resp_rate"]
    if rr >= 25:       s += 3
    elif rr >= 21:     s += 2
    sp = v["spo2"]
    if sp <= 91:       s += 3
    elif sp <= 93:     s += 2
    elif sp <= 95:     s += 1
    t = v["temperature"]
    if t >= 39:        s += 2
    elif t <= 35:      s += 3
    sbp = v["sbp"]
    if sbp <= 90:      s += 3
    elif sbp <= 100:   s += 2
    hr = v["heart_rate"]
    if hr >= 131:      s += 3
    elif hr >= 111:    s += 2
    if v["consciousness"] == 1: s += 3
    return s


# ── Utility computation ───────────────────────────────────────────────────────
def compute_utility(
    ml_prob: float = 0.0,
    news2: float = 0.0,
    hr: float = 80.0,
    rr: float = 16.0,
    spo2: float = 98.0,
    sbp: float = 120.0,
    temp: float = 37.0,
    blended_stability: float = None,
    n2_score: float = None,
    weights: dict = None,
    alert_frequency: int = 0,
    **kwargs
) -> float:
    if n2_score is not None:
        news2 = n2_score
    if rr is None and "resp_rate" in kwargs:
        rr = kwargs["resp_rate"]

    news2_scaled = min(news2 / 15.0, 1.0)
    instability = (1.0 - blended_stability) if blended_stability is not None else (1.0 - (sbp / 200.0))

    if weights:
        w_stab = weights.get("stability", 0.4)
        w_resp = weights.get("response_time", 0.3)
        utility = w_stab * (0.6 * ml_prob + 0.4 * instability) + w_resp * news2_scaled
    else:
        utility = 0.4 * ml_prob + 0.3 * news2_scaled + 0.3 * instability

    # Base scaling for critical or high urgency cases
    if instability >= 0.6 or news2_scaled >= 0.6 or ml_prob >= 0.7:
        utility = max(utility, 0.72 + 0.22 * max(instability, news2_scaled, ml_prob))

    # Critical overrides — immediate top triage priority
    if (spo2 is not None and spo2 < 90) or \
       (sbp is not None and sbp < 85) or \
       (rr is not None and rr > 35) or \
       (hr is not None and (hr > 150 or hr < 45)) or \
       (temp is not None and (temp > 40.0 or temp < 35.0)):
        utility = max(utility, 0.96)

    return round(float(min(max(utility, 0.05), 0.99)), 2)
# ── Greedy Best-First Search priority queue ───────────────────────────────────
def prioritize(patient_list: list) -> list:
    heap = []
    for p in patient_list:
        heapq.heappush(heap, (-p["utility"], p["patient_id"], p))
    result = []
    while heap:
        _, _, p = heapq.heappop(heap)
        result.append(p)
    return result

# ── Adjust ML risk based on extreme vitals ─────────────────────────────────────
def adjusted_risk(ml_prob, spo2=None, sbp=None, hr=None, temp=None, rr=None):
    """
    Adjust ML risk upwards if any vital is in a dangerous range.
    """
    risk = ml_prob

    if spo2 is not None and spo2 < 90:
        risk += 0.4  # increased from 0.3
    if hr is not None and (hr > 130 or hr < 50):
        risk += 0.15  # increased from 0.1
    if sbp is not None and (sbp < 90 or sbp > 160):
        risk += 0.15  # increased from 0.1
    if temp is not None and (temp > 39 or temp < 35):
        risk += 0.1   # increased from 0.05
    if rr is not None and rr > 30:
        risk += 0.1   # increased from 0.05

    # Clamp to 0–1
    return min(max(risk, 0), 1)


# ── Process one vital reading ─────────────────────────────────────────────────
def process_vital(vital: dict, predictor: RiskPredictor) -> dict:
    pid  = vital["patient_id"]
    qs   = qsofa(vital)
    n2   = news2(vital)

    # Determine high-risk from clinical scores
    is_high_risk  = (qs >= 2 or n2 >= 7)
    prob_critical = bayesian_update(is_high_risk)
    stability     = 1.0 - prob_critical

    # ML risk prediction
    ml_label, ml_prob_critical = predictor.predict(vital)

    # Adjust ML risk based on extreme vitals
    ml_prob_critical = adjusted_risk(
        ml_prob=ml_prob_critical,
        spo2=vital["spo2"],
        sbp=vital["sbp"],
        hr=vital["heart_rate"],
        temp=vital["temperature"],
        rr=vital["resp_rate"]
    )

    blended_stability = 0.4 * stability + 0.6 * (1.0 - ml_prob_critical)

    # Dynamic weights
    af = alert_fatigue.get(pid, 0)
    weights = get_weights_for_patient(
        patient_id=pid,
        icu_bed_availability=ICU_BED_AVAILABILITY,
        risk_label=ml_label,
        alert_fatigue_map=alert_fatigue,
    )

    # Compute final utility
    utility = compute_utility(
        blended_stability=blended_stability,
        n2_score=n2,
        ml_prob=ml_prob_critical,
        spo2=vital["spo2"],
        sbp=vital["sbp"],
        hr=vital["heart_rate"],
        temp=vital["temperature"],
        rr=vital["resp_rate"],
        weights=weights,
        alert_frequency=af
    )

    # Update alert fatigue counter if clinically high-risk
    if is_high_risk:
        alert_fatigue[pid] = af + 1

    return {
        "patient_id":    pid,
        "qsofa":         qs,
        "news2":         n2,
        "ml_risk":       ml_label,
        "ml_p_crit":     ml_prob_critical,
        "bayes_p_crit":  prob_critical,
        "stability":     blended_stability,
        "utility":       utility,
        "weights":       weights,
    }

# ── Pretty printer ────────────────────────────────────────────────────────────
RISK_ICON = {"Low": "[LOW] ", "Medium": "[MED] ", "High": "[HIGH]"}

def print_priority_table(ordered: list):
    header = (f"{'#':<3} {'PID':<5} {'qSOFA':<7} {'NEWS2':<7} "
              f"{'ML Risk':<10} {'P(Crit)':<9} {'Utility':<9} {'Weights (s/r/re/af)'}")
    print(header)
    print("-" * len(header))
    for rank, p in enumerate(ordered, 1):
        w = p["weights"]
        wstr = (f"{w['stability']:.2f}/"
                f"{w['response_time']:.2f}/"
                f"{w['resource_efficiency']:.2f}/"
                f"{w['excess_alert_frequency']:.2f}")
        icon = RISK_ICON.get(p["ml_risk"], "")
        print(
            f"{rank:<3} {p['patient_id']:<5} {p['qsofa']:<7} {p['news2']:<7} "
            f"{icon}{p['ml_risk']:<7} {p['ml_p_crit']:.3f}     "
            f"{p['utility']:.3f}     {wstr}"
        )


# ── Main simulation loop ──────────────────────────────────────────────────────
def run(csv_files: list[str], n_random: int = 5, cycles: int = 3):
    # Train ML model on available CSVs
    predictor = RiskPredictor(model_type="random_forest")
    predictor.train(csv_files)

    csv_patients    = [CSVPatient(fp, i + 1) for i, fp in enumerate(csv_files)]
    random_patients = [RandomPatient(len(csv_files) + i + 1) for i in range(n_random)]

    for cycle in range(1, cycles + 1):
        print(f"\n{'='*70}")
        print(f"  Monitoring Cycle {cycle}  |  ICU Bed Availability: "
              f"{ICU_BED_AVAILABILITY:.0%}")
        print(f"{'='*70}")

        all_patients = []

        for cp in csv_patients:
            vital = cp.get_next_vital()
            if vital:
                all_patients.append(process_vital(vital, predictor))

        for rp in random_patients:
            vital = rp.generate_vital()
            all_patients.append(process_vital(vital, predictor))

        ordered = prioritize(all_patients)
        print_priority_table(ordered)

        time.sleep(2)   # shorter delay for demo; set to 15 for real-time mode


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(base_dir, "patient*.csv")))
    if not files:
        files = sorted(glob.glob("patient*.csv"))
    run(csv_files=files, n_random=5, cycles=3)