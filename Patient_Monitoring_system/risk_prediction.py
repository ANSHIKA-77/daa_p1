import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLS = [
    "respiratory_rate_bpm", "systolic_bp_mmHg", "heart_rate_bpm",
    "spo2_percent", "body_temperature_c", "consciousness"
]

def load_csv(file_path):
    rows = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def extract_features(rows):
    X = []
    for row in rows:
        X.append([float(row[c]) for c in FEATURE_COLS])
    return np.array(X)

def assign_risk_label(row):
    resp = float(row["respiratory_rate_bpm"])
    sbp  = float(row["systolic_bp_mmHg"])
    hr   = float(row["heart_rate_bpm"])
    spo2 = float(row["spo2_percent"])
    temp = float(row["body_temperature_c"])
    cons = int(row["consciousness"])

    score = 0
    if resp >= 25: score += 2
    elif resp >= 22: score += 1
    if sbp <= 90: score += 2
    elif sbp <= 100: score += 1
    if hr >= 131: score += 2
    elif hr >= 111: score += 1
    if spo2 <= 91: score += 2
    elif spo2 <= 95: score += 1
    if temp >= 39 or temp <= 35: score += 1
    if cons == 1: score += 2

    if score >= 6: return 2    # High
    elif score >= 3: return 1  # Medium
    else: return 0             # Low

LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}

class RiskPredictor:
    def __init__(self, model_type="random_forest"):
        self.scaler = StandardScaler()
        if model_type == "logistic":
            self.model = LogisticRegression(max_iter=1000, multi_class='multinomial')
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.trained = False

    def train(self, file_paths):
        all_rows = []
        for fp in file_paths:
            all_rows.extend(load_csv(fp))

        X = extract_features(all_rows)
        y = np.array([assign_risk_label(r) for r in all_rows])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)
        self.model.fit(X_train, y_train)
        self.trained = True

        y_pred = self.model.predict(X_test)
        print("[RiskPredictor] Training complete.")
        print(classification_report(y_test, y_pred,
              target_names=["Low", "Medium", "High"], zero_division=0))

    def predict(self, vital: dict):
        """
        vital: dict with keys matching FEATURE_COLS names OR the internal keys
        Returns: (label_str, prob_critical_float)
        """
        key_map = {
            "respiratory_rate_bpm": "resp_rate",
            "systolic_bp_mmHg": "sbp",
            "heart_rate_bpm": "heart_rate",
            "spo2_percent": "spo2",
            "body_temperature_c": "temperature",
            "consciousness": "consciousness"
        }
        features = []
        for col in FEATURE_COLS:
            alt = key_map[col]
            val = vital.get(col, vital.get(alt, 0))
            features.append(float(val))

        X = self.scaler.transform([features])
        label_idx = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        prob_critical = proba[2]  # P(High)
        return LABEL_MAP[label_idx], prob_critical

    def predict_batch(self, vitals: list):
        return [self.predict(v) for v in vitals]


if __name__ == "__main__":
    import glob
    files = glob.glob("patient*.csv")
    if not files:
        files = ["patient_vitals_sample.csv"]

    predictor = RiskPredictor(model_type="random_forest")
    predictor.train(files)

    sample = {
        "resp_rate": 26, "sbp": 88, "heart_rate": 120,
        "spo2": 92, "temperature": 38.5, "consciousness": 1
    }
    label, prob = predictor.predict(sample)
    print(f"\nSample prediction -> Risk: {label}, P(Critical): {prob:.3f}")