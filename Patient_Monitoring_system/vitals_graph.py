import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

VITALS_TO_PLOT = [
    ("heart_rate_bpm",      "Heart Rate",          "bpm",   "tab:red"),
    ("spo2_percent",        "SpO2",                "%",     "tab:blue"),
    ("systolic_bp_mmHg",    "Systolic BP",         "mmHg",  "tab:green"),
    ("respiratory_rate_bpm","Respiratory Rate",    "bpm",   "tab:orange"),
    ("body_temperature_c",  "Body Temperature",    "°C",    "tab:purple"),
]

def load_vitals(file_path):
    rows = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def make_timestamps(n, interval_seconds=15):
    base = datetime.now()
    return [base + timedelta(seconds=i * interval_seconds) for i in range(n)]

def plot_patient_vitals(file_path, patient_id=None, save_path=None, show=True):
    rows = load_vitals(file_path)
    if not rows:
        print(f"[VitalsGraph] No data in {file_path}")
        return

    timestamps = make_timestamps(len(rows))
    label = patient_id if patient_id else file_path

    fig, axes = plt.subplots(len(VITALS_TO_PLOT), 1,
                             figsize=(12, 3 * len(VITALS_TO_PLOT)),
                             sharex=True)
    fig.suptitle(f"Patient Vitals Over Time — Patient {label}", fontsize=14, fontweight='bold')

    for ax, (col, title, unit, color) in zip(axes, VITALS_TO_PLOT):
        values = [float(r[col]) for r in rows if col in r]
        ax.plot(timestamps[:len(values)], values, color=color, linewidth=1.8, marker='o', markersize=3)
        ax.set_ylabel(f"{title}\n({unit})", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        # Reference bands
        if col == "spo2_percent":
            ax.axhline(95, color='orange', linestyle=':', linewidth=1, label='SpO2 warn (95%)')
            ax.axhline(91, color='red',    linestyle=':', linewidth=1, label='SpO2 crit (91%)')
            ax.legend(fontsize=7, loc='lower right')
        elif col == "systolic_bp_mmHg":
            ax.axhline(100, color='orange', linestyle=':', linewidth=1, label='SBP warn (100)')
            ax.axhline(90,  color='red',    linestyle=':', linewidth=1, label='SBP crit (90)')
            ax.legend(fontsize=7, loc='lower right')
        elif col == "heart_rate_bpm":
            ax.axhline(130, color='orange', linestyle=':', linewidth=1, label='HR warn (130)')
            ax.legend(fontsize=7, loc='upper right')

    axes[-1].set_xlabel("Time", fontsize=9)
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VitalsGraph] Saved to {save_path}")
    if show:
        plt.show()
    plt.close()

def plot_multi_patient_comparison(file_paths, vital_col="heart_rate_bpm",
                                   vital_label="Heart Rate (bpm)",
                                   save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10.colors

    for idx, fp in enumerate(file_paths):
        rows = load_vitals(fp)
        if not rows: continue
        timestamps = make_timestamps(len(rows))
        values = [float(r[vital_col]) for r in rows if vital_col in r]
        pid = idx + 1
        ax.plot(timestamps[:len(values)], values,
                label=f"Patient {pid}", color=colors[idx % len(colors)],
                linewidth=1.5, marker='o', markersize=3)

    ax.set_title(f"Multi-Patient Comparison — {vital_label}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Time")
    ax.set_ylabel(vital_label)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VitalsGraph] Comparison saved to {save_path}")
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    import glob
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(base_dir, "patient*.csv")))
    if not files:
        files = sorted(glob.glob("patient*.csv"))
    if not files:
        files = ["patient_vitals_sample.csv"]

    for i, fp in enumerate(files, 1):
        plot_patient_vitals(fp, patient_id=i, show=True)

    if len(files) > 1:
        plot_multi_patient_comparison(files, show=True)