import random

class RandomPatient:
    def __init__(self, patient_id):
        self.patient_id = patient_id

    def generate_vital(self):
        return {
            "patient_id": self.patient_id,
            "resp_rate": random.randint(10, 35),
            "sbp": random.randint(80, 140),
            "heart_rate": random.randint(60, 140),
            "spo2": random.randint(85, 100),
            "temperature": round(random.uniform(35.0, 40.0), 1),
            "consciousness": random.randint(0, 1)
        }
