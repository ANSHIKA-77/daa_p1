# AI-Based Patient Prioritization and Risk Prediction System

An intelligent healthcare system that combines clinical decision-support algorithms with machine learning to prioritize patients based on real-time vital signs and risk assessment.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Clinical Scoring Systems](#clinical-scoring-systems)
- [Machine Learning Models](#machine-learning-models)
- [Configuration](#configuration)

## Overview

This project implements an AI-driven patient prioritization system designed for emergency departments and ICU settings. It uses a multi-layered approach combining:

1. **Clinical Scoring Systems**: qSOFA and NEWS2 for evidence-based risk assessment
2. **Machine Learning**: Logistic Regression and Random Forest for predictive risk modeling
3. **Bayesian Inference**: Probabilistic reasoning for clinical decision support
4. **Dynamic Weighting**: Adaptive algorithms that adjust to ICU capacity and workload

## Features

- **Multi-Model Risk Assessment**: Combines clinical scores with ML predictions for comprehensive evaluation
- **Real-Time Vital Signs Monitoring**: Tracks heart rate, respiratory rate, SpO2, blood pressure, temperature, and consciousness level
- **Dynamic Prioritization**: Adjusts priority weights based on:
  - ICU bed availability
  - Alert fatigue metrics
  - Patient risk levels
  - Clinical stability
- **Web-Based Dashboard**: Real-time visualization of patient data, clinical scores, and priority rankings
- **Bayesian Updates**: Calculates posterior probabilities of critical status
- **Flexible Data Input**: Supports both CSV data and synthetic patient generation
- **Historical Analysis**: Tracks vital signs and scores over time

## System Architecture

### Core Components

1. **data_loader.py** - Loads patient data from CSV files
2. **random_generator.py** - Generates synthetic patient vital signs for simulation
3. **risk_prediction.py** - Machine learning models for risk classification
4. **patient_prioritization.py** - Priority queue and Bayesian calculations
5. **dynamic_weights.py** - Weight adjustment based on hospital conditions
6. **vitals_graph.py** - Visualization of vital signs trends
7. **web_server.py** - Flask-based web dashboard for real-time monitoring
8. **main.py** - Orchestrates the entire system

## Installation

### Requirements

- Python 3.7+
- Flask
- scikit-learn
- NumPy
- Matplotlib

### Setup

1. Clone or download the project
2. Install dependencies:
   ```bash
   pip install flask scikit-learn numpy matplotlib
   ```
3. Ensure CSV patient files (patient1.csv - patient5.csv) are in the project directory

## Usage

### Running the System

1. **Start the main system**:
   ```bash
   python main.py
   ```
   This will:
   - Load patient data from CSV files
   - Generate synthetic patients for simulation
   - Train ML models on vital signs data
   - Begin real-time patient monitoring and prioritization
   - Output priority queue and risk assessments

2. **Access the Web Dashboard**:
   ```bash
   python web_server.py
   ```
   Then open `index.html` in a web browser to view:
   - Real-time vital signs for all patients
   - Clinical scores (qSOFA, NEWS2)
   - ML risk predictions
   - Priority queue rankings
   - Historical charts

## Project Structure

```
AI_PROJECT1/
├── main.py                      # Main orchestrator
├── patient_prioritization.py    # Prioritization algorithms
├── risk_prediction.py           # ML models (LR, RF)
├── data_loader.py               # CSV data handling
├── random_generator.py          # Synthetic patient generation
├── dynamic_weights.py           # Dynamic weight adjustment
├── vitals_graph.py              # Visualization utilities
├── web_server.py                # Flask dashboard
├── index.html                   # Web interface
├── patient1.csv - patient5.csv  # Patient vital signs data
└── README.md                    # This file
```

## Clinical Scoring Systems

### qSOFA (Quick SOFA)
Rapid assessment for sepsis risk. Scores 0-3 based on:
- Respiratory rate ≥ 22 (1 point)
- Systolic blood pressure ≤ 100 mmHg (1 point)
- Altered consciousness (1 point)

**Clinical Use**: Quick screening for sepsis-related organ dysfunction

### NEWS2 (National Early Warning Score 2)
Comprehensive acuity score. Weighted scoring for:
- Respiratory rate
- SpO2 level
- Body temperature
- Systolic blood pressure
- Heart rate
- Consciousness level
- Oxygen supplementation

**Clinical Use**: Standardized assessment of patient acuity and deterioration risk

## Machine Learning Models

### Feature Set
- Respiratory rate (breaths per minute)
- Systolic blood pressure (mmHg)
- Heart rate (beats per minute)
- SpO2 (blood oxygen percentage)
- Body temperature (°C)
- Consciousness level (0/1)

### Risk Categories
- **Low**: Score 0-2 (managed on general ward)
- **Medium**: Score 3-5 (close monitoring, potential ICU)
- **High**: Score ≥6 (immediate ICU consideration)

### Model Performance
Both Logistic Regression and Random Forest are trained on patient data with cross-validation. Performance metrics are generated during training for model validation.

## Configuration

### Dynamic Weight Parameters (dynamic_weights.py)

Default weight distribution:
- **stability** (0.4): Patient physiological stability
- **response_time** (0.3): Urgency of intervention needed
- **resource_efficiency** (0.2): ICU bed utilization
- **excess_alert_frequency** (0.1): Alert fatigue management

### Bayesian Priors (patient_prioritization.py)
- Prior probability of critical status: 20%
- Likelihood ratios for risk assessment
- ICU bed availability tracking

### Thresholds
- ICU bed low threshold: ≤20% available
- ICU bed medium threshold: ≤50% available
- Alert fatigue high: >5 alerts per patient
- Alert fatigue medium: >2 alerts per patient

## Clinical Validation

This system is designed to complement clinical judgment, not replace it. All recommendations should be reviewed by qualified healthcare professionals. The combination of multiple assessment methods (clinical scores + ML) provides more robust decision support than any single approach.

## Future Enhancements

- Integration with hospital EHR systems
- Real-time vital signs from monitoring devices
- Improved alert fatigue models
- Additional clinical scoring systems
- Predictive deterioration modeling
- Multi-hospital resource coordination

## License

This is an academic project for AI coursework.

## Contact

For questions or issues, please refer to the project documentation or contact the development team.

---

**Note**: This system is a demonstration of AI in healthcare. Always follow institutional protocols and clinical guidelines when making patient care decisions.
