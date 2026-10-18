import time
import heapq
import glob
import os
import sys

# Ensure UTF-8 output on Windows console
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from data_loader import CSVPatient
from random_generator import RandomPatient
from risk_prediction import RiskPredictor
from dynamic_weights import get_weights_for_patient


# qSOFA
def calculate_qsofa(patient):
    score = 0
    if patient["resp_rate"] >= 22:
        score += 1
    if patient["sbp"] <= 100:
        score += 1
    if patient["consciousness"] == 1:
        score += 1
    return score



# NEWS2
def calculate_news2(patient):
    score = 0

    if patient["resp_rate"] >= 25:
        score += 3
    elif 21 <= patient["resp_rate"] <= 24:
        score += 2

    if patient["spo2"] <= 91:
        score += 3
    elif 92 <= patient["spo2"] <= 93:
        score += 2
    elif 94 <= patient["spo2"] <= 95:
        score += 1

    if patient["temperature"] >= 39:
        score += 2
    elif patient["temperature"] <= 35:
        score += 3

    if patient["sbp"] <= 90:
        score += 3
    elif 91 <= patient["sbp"] <= 100:
        score += 2

    if patient["heart_rate"] >= 131:
        score += 3
    elif 111 <= patient["heart_rate"] <= 130:
        score += 2

    if patient["consciousness"] == 1:
        score += 3

    return score


# AI Utility Model 

# the weights correspond to the four terms in the new utility function
weights = {
    "stability": 0.4,              # w1
    "response_time": 0.3,          # w2
    "resource_efficiency": 0.2,    # w3
    "excess_alert_frequency": 0.1, # w4
}

# Bayesian Parameters

# Prior probability that any monitored patient is critical
PRIOR_CRITICAL = 0.2  

# estimates
LIKELIHOOD_HIGH_RISK_GIVEN_CRITICAL = 0.85
LIKELIHOOD_HIGH_RISK_GIVEN_NOT_CRITICAL = 0.30


alert_fatigue = {}
environment_value = 0.5  # Example ICU stress factor 

# ML Model Initialization and Training
predictor = RiskPredictor(model_type="random_forest")
csv_files = sorted(glob.glob(os.path.join(BASE_DIR, "patient*.csv")))
if not csv_files:
    csv_files = sorted(glob.glob("patient*.csv"))
if csv_files:
    print("[System] Training ML Risk Predictor on CSV files...")
    predictor.train(csv_files)
    print("[System] ML model training complete.")
else:
    print("[Warning] No patient CSV files found. Model will be untrained.")


def compute_vital_risk(qsofa, news2):
    """Original vital risk helper left in place for backwards compatibility.
    We can reuse it when deriving stability and response-time proxies.
    """
    return (qsofa / 3.0 + news2 / 20.0) / 2.0


# Bayesian Probability Update

def bayesian_update(is_high_risk: bool):
    """
    Computes P(Critical | Evidence) using Bayes Rule.
    Evidence = whether patient is high-risk based on qSOFA/NEWS2
    """

    if is_high_risk:
        likelihood = LIKELIHOOD_HIGH_RISK_GIVEN_CRITICAL
        evidence_prob = (
            LIKELIHOOD_HIGH_RISK_GIVEN_CRITICAL * PRIOR_CRITICAL
            + LIKELIHOOD_HIGH_RISK_GIVEN_NOT_CRITICAL * (1 - PRIOR_CRITICAL)
        )
    else:
        # Complement probabilities
        likelihood = 1 - LIKELIHOOD_HIGH_RISK_GIVEN_CRITICAL
        evidence_prob = (
            (1 - LIKELIHOOD_HIGH_RISK_GIVEN_CRITICAL) * PRIOR_CRITICAL
            + (1 - LIKELIHOOD_HIGH_RISK_GIVEN_NOT_CRITICAL) * (1 - PRIOR_CRITICAL)
        )

    posterior = (likelihood * PRIOR_CRITICAL) / evidence_prob
    return posterior


def compute_utility(patient_id: int = 0,
                    stability: float = 0.5,
                    response_time: float = 0.0,
                    resource_efficiency: float = 0.5,
                    excess_alert_freq: float = 0.0,
                    weights_dict: dict = None,
                    vital: dict = None,
                    risk_label: str = None) -> float:
    """Compute triage priority utility score (0.05 to 0.99).

    Higher utility represents HIGHER triage urgency and intervention priority.
    - High Risk: 0.88 to 0.99
    - Medium Risk: 0.40 to 0.84
    - Low Risk: 0.08 to 0.35
    """
    w = weights_dict if weights_dict else weights

    instability = max(0.0, min(1.0, 1.0 - stability))
    clinical_urgency = min(response_time * (20.0 / 15.0), 1.0) if response_time > 0 else 0.0

    # Risk-tiered scaling guarantees clear differentiation across triage levels
    if risk_label == "High" or instability >= 0.7 or clinical_urgency >= 0.7:
        utility = 0.88 + 0.08 * max(instability, clinical_urgency)
    elif risk_label == "Medium" or instability >= 0.3 or clinical_urgency >= 0.3:
        utility = 0.40 + 0.30 * max(instability, clinical_urgency)
    else:
        utility = 0.08 + 0.22 * max(instability, clinical_urgency)

    # Critical Vital Signs Overrides (Top Triage Escalation)
    if vital:
        spo2 = vital.get("spo2", vital.get("spo2_percent", 98))
        sbp = vital.get("sbp", vital.get("systolic_bp_mmHg", 120))
        hr = vital.get("heart_rate", vital.get("heart_rate_bpm", 80))
        rr = vital.get("resp_rate", vital.get("respiratory_rate_bpm", 16))
        temp = vital.get("temperature", vital.get("body_temperature_c", 37.0))

        if (spo2 is not None and spo2 < 90) or \
           (sbp is not None and sbp < 85) or \
           (hr is not None and (hr > 150 or hr < 45)) or \
           (rr is not None and rr > 35) or \
           (temp is not None and (temp > 40.0 or temp < 35.0)):
            if risk_label == "High" or instability >= 0.6:
                utility = max(utility, 0.95 + 0.04 * clinical_urgency)
            else:
                utility = max(utility, 0.78 + 0.06 * clinical_urgency)

    return round(float(min(max(utility, 0.05), 0.99)), 2)


# Initialize Patients
def create_default_patients():
    c_patients = [
        CSVPatient(os.path.join(BASE_DIR, f"patient{i}.csv"), i)
        for i in range(1, 6)
    ]
    r_patients = [RandomPatient(i) for i in range(6, 11)]
    return c_patients, r_patients

csv_patients, random_patients = create_default_patients()


# GREEDY BEST-FIRST SEARCH
def prioritize_patients(patient_list):
    """
    Implements Greedy Best-First Search.
    Uses utility score as heuristic h(n).
    """
    priority_queue = []
    patient_map = {p["patient_id"]: p for p in patient_list}

    for patient in patient_list:
        utility = patient["utility"]
        patient_id = patient["patient_id"]
        # Invert utility for max-priority queue behavior (higher utility treated first)
        heapq.heappush(priority_queue, (-utility, patient_id))

    ordered_patients = []
    while priority_queue:
        _, patient_id = heapq.heappop(priority_queue)
        ordered_patients.append(patient_map[patient_id])

    return ordered_patients


def run_simulation(cycles=3, delay=2):
    print("\n" + "=" * 78)
    print("  AI-Based Patient Prioritization and Risk Prediction System - Live Monitor")
    print("=" * 78)

    for cycle in range(1, cycles + 1):
        print(f"\n" + "=" * 78)
        print(f"  Cycle {cycle} | ICU Environment Stress: {environment_value:.0%}")
        print("=" * 78)

        all_patients = []

        # Process CSV patients
        for cp in csv_patients:
            vital = cp.get_next_vital()
            if vital:
                qsofa = calculate_qsofa(vital)
                news2 = calculate_news2(vital)
                is_high_risk = (qsofa >= 2 or news2 >= 7)
                prob_critical = bayesian_update(is_high_risk)
                stability = 1.0 - prob_critical
                response_time = news2 / 20.0
                resource_eff = 1.0 - environment_value
                af = alert_fatigue.get(vital["patient_id"], 0)
                ml_label, ml_prob = predictor.predict(vital)
                blended_stability = 0.5 * stability + 0.5 * (1.0 - ml_prob)
                w = get_weights_for_patient(
                    patient_id=vital["patient_id"],
                    icu_bed_availability=0.5,
                    risk_label=ml_label,
                    alert_fatigue_map=alert_fatigue
                )
                utility = compute_utility(vital["patient_id"], blended_stability, response_time, resource_eff, af, w, vital=vital)
                all_patients.append({
                    "patient_id": vital["patient_id"],
                    "vitals": vital,
                    "qsofa": qsofa,
                    "news2": news2,
                    "ml_risk": ml_label,
                    "ml_prob": ml_prob,
                    "prob_critical": prob_critical,
                    "utility": utility,
                    "weights": w
                })
                if is_high_risk:
                    alert_fatigue[vital["patient_id"]] = af + 1

        # Process Random/Synthetic patients
        for rp in random_patients:
            vital = rp.generate_vital()
            qsofa = calculate_qsofa(vital)
            news2 = calculate_news2(vital)
            is_high_risk = (qsofa >= 2 or news2 >= 7)
            prob_critical = bayesian_update(is_high_risk)
            stability = 1.0 - prob_critical
            response_time = news2 / 20.0
            resource_eff = 1.0 - environment_value
            af = alert_fatigue.get(vital["patient_id"], 0)
            ml_label, ml_prob = predictor.predict(vital)
            blended_stability = 0.5 * stability + 0.5 * (1.0 - ml_prob)
            w = get_weights_for_patient(
                patient_id=vital["patient_id"],
                icu_bed_availability=0.5,
                risk_label=ml_label,
                alert_fatigue_map=alert_fatigue
            )
            utility = compute_utility(vital["patient_id"], blended_stability, response_time, resource_eff, af, w, vital=vital)
            all_patients.append({
                "patient_id": vital["patient_id"],
                "vitals": vital,
                "qsofa": qsofa,
                "news2": news2,
                "ml_risk": ml_label,
                "ml_prob": ml_prob,
                "prob_critical": prob_critical,
                "utility": utility,
                "weights": w
            })
            if is_high_risk:
                alert_fatigue[vital["patient_id"]] = af + 1

        ordered = prioritize_patients(all_patients)
        header = f"{'#':<3} {'PID':<5} {'HR':<5} {'SpO2':<6} {'SBP':<5} {'qSOFA':<7} {'NEWS2':<7} {'ML Risk':<9} {'P(High)':<9} {'Utility':<8}"
        print(header)
        print("-" * len(header))
        for rank, p in enumerate(ordered, 1):
            v = p["vitals"]
            print(f"{rank:<3} {p['patient_id']:<5} {v['heart_rate']:<5} {v['spo2']:<6.1f} {v['sbp']:<5} "
                  f"{p['qsofa']:<7} {p['news2']:<7} {p['ml_risk']:<9} {p['ml_prob']:.2f}     {p['utility']:.3f}")
        time.sleep(delay)


if __name__ == "__main__":
    run_simulation(cycles=3, delay=2)

