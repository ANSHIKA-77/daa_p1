import csv

class CSVPatient:
    def __init__(self, file_path, patient_id):
        self.file_path = file_path
        self.patient_id = patient_id
        self.data = []
        self.index = 0
        self.load_data()

    def load_data(self):
        with open(self.file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.data.append({
                    "patient_id": self.patient_id,
                    "resp_rate": int(row["respiratory_rate_bpm"]),
                    "sbp": int(row["systolic_bp_mmHg"]),
                    "heart_rate": int(row["heart_rate_bpm"]),
                    "spo2": float(row["spo2_percent"]),
                    "temperature": float(row["body_temperature_c"]),
                    "consciousness": int(row["consciousness"])
                })

    def get_next_vital(self):
        if not self.data:
            return None
        vital = self.data[self.index]
        self.index = (self.index + 1) % len(self.data)
        return vital
