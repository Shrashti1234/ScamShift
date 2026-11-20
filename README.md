# ScamShift: MLOps SMS Scam Detection & Automated Retraining Pipeline

ScamShift is a lightweight, production-ready MLOps project that flags suspicious messages (SMS/UPI scams in Hinglish & English) and provides an automated model-quality monitoring, retraining, and promotion workflow.

---

## 📁 Project Architecture

```
Scamshift/
├── .github/workflows/
│   └── ci.yaml                  # Phase 3: GitHub Actions CI workflow
├── data/
│   ├── SMSSpamCollection        # Phase 1: Baseline SMS spam dataset
│   └── incoming_batch.csv       # Task A: New incoming production data batch
├── models/
│   ├── champion_model.pkl       # Phase 1: Active deployed Champion model
│   ├── champion_evaluation.json # Champion baseline test metrics
│   ├── candidate_model.pkl      # Task C: Retrained Candidate model artifact
│   ├── candidate_evaluation.json# Candidate evaluation metadata
│   └── quality_gate_decision.json# Task D: Quality Gate audit record
├── tests/
│   ├── test_api.py              # Phase 3: Automated FastAPI endpoint tests
│   └── test_tasks.py            # Automated tests for Tasks A, B, C, D
├── .gitignore                   # Ignored files (caches, logs, DBs)
├── Dockerfile                   # Phase 2: Docker container packaging
├── main.py                      # Phase 1: FastAPI serving API
├── notes.txt                    # KIET MLOps curriculum notes
├── requirements.txt             # Project dependencies
├── train.py                     # Training, monitoring, & retraining pipeline
└── README.md                    # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
python -m venv kiet-env
# Windows:
.\kiet-env\Scripts\activate
# Linux / macOS:
source kiet-env/bin/activate

pip install -r requirements.txt
```

### 2. Train Baseline Champion Model
```bash
python train.py --champion
```
*Trains a TF-IDF + Logistic Regression pipeline on historical SMS data and saves `models/champion_model.pkl`.*

### 3. Run FastAPI Prediction Service
```bash
uvicorn main:app --reload --port 8000
```
* Interactive Swagger docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Health check: `GET http://127.0.0.1:8000/health`
* Prediction: `POST http://127.0.0.1:8000/predict` with `{"text": "Aapka account freeze ho gaya, OTP share karein"}`

### 4. Run Automated Tests
```bash
pytest tests/ -v
```

---

## 📊 MLOps Practical Assessment Tasks (in `train.py`)

All assessment tasks are implemented as separate, modular functions inside [`train.py`](train.py):

| Task | Function in `train.py` | Direct Command | Description |
| :--- | :--- | :--- | :--- |
| **Task A** | `task_a_prepare_data()` | `python train.py --task A` | Ingests `data/incoming_batch.csv`, validates schema (`id`, `text`, `label`), filters corrupted rows, drops duplicates, and prints class distribution (50% Genuine, 50% Scam). |
| **Task B** | `task_b_evaluate_model()` | `python train.py --task B` | Evaluates deployed champion on new batch. Demonstrates concept drift: Accuracy: **60.0%**, Precision: **63.6%**, Recall: **46.7%**, F1: **53.8%**. Saves report. |
| **Task C** | `task_c_retrain_model()` | `python train.py --task C` | Combines baseline data (3,869 rows) with incoming batch (60 rows) = 3,929 rows. Retrains candidate model (100% Recall on new batch, 93.8% historical retention). Logs to MLflow. |
| **Task D** | `task_d_quality_gate()` | `python train.py --task D` | Evaluates 4 quality criteria (F1 improvement, Recall $+53.3\text{ pp}$, False Alarm $\le 5\%$, Benchmark retention loss $\le 2\text{ pp}$). Records `ELIGIBLE_FOR_PROMOTION` audit record. |
| **Task E** | `run_retraining_pipeline()` | `python train.py --pipeline` | Orchestrates Tasks A $\rightarrow$ B $\rightarrow$ C $\rightarrow$ D end-to-end and outputs execution evidence. |

---

## 🐳 Docker Virtualization
```bash
docker build -t scamshift-api:latest .
docker run -d -p 8000:8000 --name scamshift-container scamshift-api:latest
```
