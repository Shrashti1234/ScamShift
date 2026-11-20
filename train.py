"""
MLOps Training, Monitoring, and Retraining Pipeline
===================================================
Contains Phase-1 Baseline Training and Practical Assessment Tasks A, B, C, D, E.

Usage:
  python train.py               # Runs baseline champion training & retraining pipeline
  python train.py --champion    # Trains only the baseline champion model
  python train.py --pipeline    # Runs Tasks A -> B -> C -> D monitoring & retraining
  python train.py --task A      # Runs only Task A (Prepare New Data)
  python train.py --task B      # Runs only Task B (Evaluate Existing Model)
  python train.py --task C      # Runs only Task C (Retrain Candidate Model)
  python train.py --task D      # Runs only Task D (Quality Gate Decision)
"""

import argparse
import csv
import json
import os
import pickle
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
SEED = 42
DEFAULT_MLFLOW_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "ScamShift-Monitoring-Retraining"


# ==============================================================================
# Helper Preprocessing & Metric Functions
# ==============================================================================

def normalize_text(text: str) -> str:
    """Normalize text by folding unicode and whitespace."""
    if not isinstance(text, str):
        return ""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load_sms_corpus():
    """Load baseline SMS spam dataset with 75% train / 25% test stratified split."""
    corpus_file = DATA_DIR / "SMSSpamCollection"
    if not corpus_file.exists():
        raise FileNotFoundError(f"SMS dataset not found at {corpus_file}")

    pairs = []
    seen = set()
    for line in corpus_file.read_text(encoding="utf-8").splitlines():
        label_str, sep, text = line.partition("\t")
        if sep and label_str in ("ham", "spam"):
            clean_text = text.strip()
            norm = normalize_text(clean_text)
            if norm not in seen:
                seen.add(norm)
                pairs.append((clean_text, 1 if label_str == "spam" else 0))

    labels = [label for _, label in pairs]
    return train_test_split(pairs, test_size=0.25, random_state=SEED, stratify=labels)


def create_pipeline() -> Pipeline:
    """Create word and character TF-IDF FeatureUnion + Logistic Regression pipeline."""
    return Pipeline([
        ("features", FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("characters", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ])),
        ("classifier", LogisticRegression(C=4, max_iter=500, random_state=SEED, class_weight="balanced")),
    ])


def calculate_metrics(labels: list, scores: list, threshold: float = 0.5) -> dict:
    """Calculate binary classification metrics (Accuracy, Precision, Recall, F1, False Alarm)."""
    samples = [(int(y), float(s)) for y, s in zip(labels, scores)]
    total = len(samples)
    if total == 0:
        return {}

    tp = sum(y == 1 and s >= threshold for y, s in samples)
    fp = sum(y == 0 and s >= threshold for y, s in samples)
    fn = sum(y == 1 and s < threshold for y, s in samples)
    tn = sum(y == 0 and s < threshold for y, s in samples)

    positives = tp + fn
    negatives = fp + tn

    accuracy = (tp + tn) / total
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / positives) if positives > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    false_alarm = (fp / negatives) if negatives > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alarm": round(false_alarm, 4),
        "positives": positives,
        "negatives": negatives,
        "total": total,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


# ==============================================================================
# Phase 1: Train Baseline Champion Model
# ==============================================================================

def train_champion():
    """Trains baseline champion model and saves artifact to models/champion_model.pkl."""
    print("\n" + "=" * 65)
    print("PHASE 1: TRAIN BASELINE CHAMPION MODEL")
    print("=" * 65)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_pairs, test_pairs = load_sms_corpus()
    print(f"Loaded SMS dataset: {len(train_pairs)} training rows, {len(test_pairs)} test rows.")

    train_texts, train_labels = zip(*train_pairs)
    test_texts, test_labels = zip(*test_pairs)

    pipeline = create_pipeline()
    print("Fitting TF-IDF FeatureUnion + Logistic Regression...")
    pipeline.fit([normalize_text(t) for t in train_texts], train_labels)

    # Benchmark test evaluation
    test_scores = pipeline.predict_proba([normalize_text(t) for t in test_texts])[:, 1]
    test_metrics = calculate_metrics(test_labels, test_scores, threshold=0.5)

    print("\n--- Baseline Champion Test Metrics ---")
    print(f"  Accuracy:    {test_metrics['accuracy']:.1%}")
    print(f"  Precision:   {test_metrics['precision']:.1%}")
    print(f"  Recall:      {test_metrics['recall']:.1%}")
    print(f"  F1-Score:    {test_metrics['f1']:.1%}")
    print(f"  False Alarm: {test_metrics['false_alarm']:.1%}")

    model_path = MODELS_DIR / "champion_model.pkl"
    meta_path = MODELS_DIR / "champion_evaluation.json"

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    meta = {
        "model_name": "ScamShift Champion Baseline",
        "threshold": 0.5,
        "train_samples": len(train_pairs),
        "test_samples": len(test_pairs),
        "benchmark_metrics": test_metrics,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[OK] Champion model saved to: {model_path}")
    print(f"[OK] Champion metadata saved: {meta_path}")
    print("=" * 65 + "\n")
    return pipeline, test_metrics


# ==============================================================================
# Task A: Prepare and Validate New Data
# ==============================================================================

def task_a_prepare_data(batch_file: Path = DATA_DIR / "incoming_batch.csv") -> list:
    """Task A: Load new batch, validate schema/missing values, remove duplicates, print distribution."""
    print("\n" + "=" * 65)
    print("TASK A: PREPARE AND VALIDATE NEW INCOMING DATA")
    print("=" * 65)
    print(f"Loading incoming batch from: {batch_file}")

    if not batch_file.exists():
        raise FileNotFoundError(f"Incoming batch file not found: {batch_file}")

    required_columns = {"id", "text", "label"}
    raw_rows = []

    with open(batch_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        missing_cols = required_columns - header
        if missing_cols:
            raise ValueError(f"Schema validation failed! Missing required columns: {sorted(missing_cols)}")

        for idx, row in enumerate(reader, start=1):
            raw_rows.append({"row_num": idx, **row})

    total_raw = len(raw_rows)
    print(f"[OK] Schema validated: Required columns {sorted(required_columns)} present.")
    print(f"[INFO] Raw records loaded: {total_raw}")

    # Missing value check
    valid_rows = []
    missing_count = 0
    for r in raw_rows:
        row_id = (r.get("id") or "").strip()
        text = (r.get("text") or "").strip()
        label_raw = (r.get("label") or "").strip()

        if not row_id or not text or not label_raw:
            missing_count += 1
            print(f"  [WARN] Dropping row #{r['row_num']} (ID: '{row_id}') - missing required field(s).")
            continue

        try:
            label = int(label_raw)
            if label not in (0, 1):
                raise ValueError
        except ValueError:
            missing_count += 1
            print(f"  [WARN] Dropping row #{r['row_num']} (ID: '{row_id}') - invalid binary label '{label_raw}'.")
            continue

        valid_rows.append({"id": row_id, "text": text, "label": label, "category": r.get("category", "General")})

    # Duplicate check
    seen_ids = set()
    seen_texts = set()
    cleaned_rows = []
    duplicate_count = 0

    for r in valid_rows:
        norm = normalize_text(r["text"])
        if r["id"] in seen_ids:
            duplicate_count += 1
            print(f"  [WARN] Dropping duplicate record with ID: '{r['id']}'")
            continue
        if norm in seen_texts:
            duplicate_count += 1
            print(f"  [WARN] Dropping duplicate record with identical text (ID: '{r['id']}')")
            continue

        seen_ids.add(r["id"])
        seen_texts.add(norm)
        cleaned_rows.append(r)

    total_clean = len(cleaned_rows)
    scam_count = sum(1 for r in cleaned_rows if r["label"] == 1)
    genuine_count = sum(1 for r in cleaned_rows if r["label"] == 0)

    print("\n--- Task A Summary ---")
    print(f"  Total raw records received:        {total_raw}")
    print(f"  Dropped records (missing/invalid): {missing_count}")
    print(f"  Dropped records (duplicates):       {duplicate_count}")
    print(f"  Cleaned dataset size:              {total_clean}")
    print(f"  Target Distribution:")
    print(f"    - Class 0 (Genuine / Ham): {genuine_count:3d} ({genuine_count / total_clean:.1%})")
    print(f"    - Class 1 (Scam / Spam):    {scam_count:3d} ({scam_count / total_clean:.1%})")
    print("=" * 65 + "\n")

    return cleaned_rows


# ==============================================================================
# Task B: Evaluate Existing Model
# ==============================================================================

def task_b_evaluate_model(champion_model=None,
                          cleaned_data: list = None,
                          threshold: float = 0.5) -> dict:
    """Task B: Load champion model, predict on new batch, calculate metrics, save report."""
    print("\n" + "=" * 65)
    print("TASK B: EVALUATE EXISTING MODEL ON NEW DATA")
    print("=" * 65)

    if cleaned_data is None:
        cleaned_data = task_a_prepare_data()

    if champion_model is None:
        model_path = MODELS_DIR / "champion_model.pkl"
        if not model_path.exists():
            train_champion()
        with open(model_path, "rb") as f:
            champion_model = pickle.load(f)

    texts = [r["text"] for r in cleaned_data]
    labels = [r["label"] for r in cleaned_data]

    print(f"Generating predictions on {len(texts)} samples (Threshold = {threshold:.2f})...")
    scores = champion_model.predict_proba([normalize_text(t) for t in texts])[:, 1]
    metrics = calculate_metrics(labels, scores, threshold=threshold)

    print("\n--- Task B Champion Metrics on New Batch ---")
    print(f"  Accuracy:         {metrics['accuracy']:.1%} ({metrics['accuracy']:.4f})")
    print(f"  Precision:        {metrics['precision']:.1%} ({metrics['precision']:.4f})")
    print(f"  Recall:           {metrics['recall']:.1%} ({metrics['recall']:.4f})")
    print(f"  F1-Score:         {metrics['f1']:.1%} ({metrics['f1']:.4f})")
    print(f"  False Alarm Rate: {metrics['false_alarm']:.1%} ({metrics['false_alarm']:.4f})")
    print(f"  Confusion Matrix: TP={metrics['confusion']['tp']}, "
          f"FP={metrics['confusion']['fp']}, "
          f"FN={metrics['confusion']['fn']}, "
          f"TN={metrics['confusion']['tn']}")

    report = {
        "model_role": "Champion (Baseline)",
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": threshold,
        "sample_count": len(cleaned_data),
        "metrics": metrics,
    }

    report_path = MODELS_DIR / "evaluation_report_champion.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Champion evaluation report saved to: {report_path}")
    print("=" * 65 + "\n")
    return metrics, report


# ==============================================================================
# Task C: Retrain Candidate Model & Track with MLflow
# ==============================================================================

def task_c_retrain_model(cleaned_incoming_data: list = None,
                         tracking_uri: str = DEFAULT_MLFLOW_URI,
                         threshold: float = 0.5,
                         feedback_weight: float = 18.0):
    """Task C: Combine datasets, retrain candidate pipeline, save separately, log to MLflow."""
    print("\n" + "=" * 65)
    print("TASK C: RETRAIN CANDIDATE MODEL & TRACK WITH MLFLOW")
    print("=" * 65)

    if cleaned_incoming_data is None:
        cleaned_incoming_data = task_a_prepare_data()

    base_train, base_test = load_sms_corpus()
    incoming_pairs = [(r["text"], r["label"]) for r in cleaned_incoming_data]
    combined_pairs = base_train + incoming_pairs

    print(f"Dataset Composition:")
    print(f"  Historical Baseline Training Rows: {len(base_train)}")
    print(f"  New Incoming Labelled Batch Rows:  {len(incoming_pairs)}")
    print(f"  Combined Dataset Rows:             {len(combined_pairs)}")

    messages, labels = zip(*combined_pairs)
    sample_weights = [1] * len(base_train) + [feedback_weight] * len(incoming_pairs)

    print("\nFitting candidate pipeline (TF-IDF FeatureUnion + Logistic Regression)...")
    candidate_pipeline = create_pipeline()
    candidate_pipeline.fit(
        [normalize_text(m) for m in messages], labels, classifier__sample_weight=sample_weights
    )

    # Evaluate candidate on new batch and benchmark holdout
    inc_texts, inc_labels = zip(*incoming_pairs)
    inc_scores = candidate_pipeline.predict_proba([normalize_text(t) for t in inc_texts])[:, 1]
    candidate_inc_metrics = calculate_metrics(inc_labels, inc_scores, threshold=threshold)

    bench_texts, bench_labels = zip(*base_test)
    bench_scores = candidate_pipeline.predict_proba([normalize_text(t) for t in bench_texts])[:, 1]
    candidate_bench_metrics = calculate_metrics(bench_labels, bench_scores, threshold=threshold)

    print("\n--- Task C Candidate Performance ---")
    print("On New Incoming Batch:")
    print(f"  Accuracy:    {candidate_inc_metrics['accuracy']:.1%}")
    print(f"  Precision:   {candidate_inc_metrics['precision']:.1%}")
    print(f"  Recall:      {candidate_inc_metrics['recall']:.1%}")
    print(f"  F1-Score:    {candidate_inc_metrics['f1']:.1%}")
    print(f"  False Alarm: {candidate_inc_metrics['false_alarm']:.1%}")
    print("On Historical Benchmark Test Split (1,290 samples):")
    print(f"  Benchmark Recall:      {candidate_bench_metrics['recall']:.1%}")
    print(f"  Benchmark False Alarm: {candidate_bench_metrics['false_alarm']:.1%}")

    # Save Candidate Model Artifacts SEPARATELY (never overwriting champion)
    output_model_path = MODELS_DIR / "candidate_model.pkl"
    output_meta_path = MODELS_DIR / "candidate_evaluation.json"

    with open(output_model_path, "wb") as f:
        pickle.dump(candidate_pipeline, f)

    meta = {
        "model_role": "Candidate",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": threshold,
        "feedback_weight": feedback_weight,
        "training_rows_base": len(base_train),
        "incoming_batch_rows": len(incoming_pairs),
        "total_training_rows": len(combined_pairs),
        "incoming_batch_metrics": candidate_inc_metrics,
        "benchmark_metrics": candidate_bench_metrics,
    }
    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[OK] Candidate model saved separately to: {output_model_path}")
    print(f"[OK] Candidate metadata saved to:         {output_meta_path}")

    # Track in MLflow
    print(f"\nTracking run in MLflow (URI: {tracking_uri})...")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"candidate-batch-{len(incoming_pairs)}") as run:
        run_id = run.info.run_id
        mlflow.log_param("model_type", "Logistic Regression + Word & Char TF-IDF")
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("feedback_weight", feedback_weight)
        mlflow.log_param("baseline_training_rows", len(base_train))
        mlflow.log_param("incoming_batch_rows", len(incoming_pairs))
        mlflow.log_param("total_training_rows", len(combined_pairs))

        mlflow.log_metric("candidate_accuracy", candidate_inc_metrics["accuracy"])
        mlflow.log_metric("candidate_precision", candidate_inc_metrics["precision"])
        mlflow.log_metric("candidate_recall", candidate_inc_metrics["recall"])
        mlflow.log_metric("candidate_f1", candidate_inc_metrics["f1"])
        mlflow.log_metric("candidate_false_alarm", candidate_inc_metrics["false_alarm"])

        mlflow.log_metric("benchmark_recall", candidate_bench_metrics["recall"])
        mlflow.log_metric("benchmark_false_alarm", candidate_bench_metrics["false_alarm"])

        mlflow.log_artifact(str(output_model_path), artifact_path="model")
        mlflow.log_artifact(str(output_meta_path), artifact_path="metadata")

        print(f"[OK] MLflow run logged successfully! Run ID: {run_id}")

    meta["mlflow_run_id"] = run_id
    print("=" * 65 + "\n")
    return candidate_pipeline, candidate_inc_metrics, candidate_bench_metrics, meta


# ==============================================================================
# Task D: Implement Quality Gate
# ==============================================================================

def task_d_quality_gate(champion_inc: dict = None,
                        candidate_inc: dict = None,
                        champion_bench: dict = None,
                        candidate_bench: dict = None):
    """Task D: Evaluate quality gate decision rules, justify metric, save audit record."""
    print("\n" + "=" * 65)
    print("TASK D: QUALITY GATE DECISION AND PROMOTION EVALUATION")
    print("=" * 65)

    if champion_inc is None or candidate_inc is None:
        cleaned = task_a_prepare_data()
        champion_inc, _ = task_b_evaluate_model(cleaned_data=cleaned)
        with open(MODELS_DIR / "champion_evaluation.json", "r", encoding="utf-8") as f:
            champion_bench = json.load(f)["benchmark_metrics"]
        _, candidate_inc, candidate_bench, _ = task_c_retrain_model(cleaned_incoming_data=cleaned)

    justification = (
        "METRIC SELECTION JUSTIFICATION:\n"
        "In SMS scam and financial fraud detection, error costs are strictly asymmetric:\n"
        "1. False Negatives (Missing a scam message): Critical severity. Victims suffer direct financial theft,\n"
        "   credential compromise, or malware infection. Maximizing Recall on emerging scam patterns is essential.\n"
        "2. False Positives (Flagging legitimate messages): Medium severity. Blocking genuine bank OTPs, salary\n"
        "   notifications, or personal messages erodes user trust and disrupts critical communications.\n"
        "3. Selection: The F1-Score (harmonic mean of Precision and Recall) serves as the primary balanced metric.\n"
        "   We couple F1 with two safety guardrails: a False Alarm ceiling (<= 5.0%) and a Historical Benchmark\n"
        "   Retention gate (<= 2.0% recall loss) to prevent catastrophic forgetting on historical communications."
    )
    print(justification)

    champ_f1 = champion_inc["f1"]
    cand_f1 = candidate_inc["f1"]
    delta_f1 = cand_f1 - champ_f1

    champ_rec = champion_inc["recall"]
    cand_rec = candidate_inc["recall"]
    delta_rec = cand_rec - champ_rec

    cand_fa = candidate_inc["false_alarm"]

    champ_bench_rec = champion_bench["recall"]
    cand_bench_rec = candidate_bench["recall"]
    bench_loss = champ_bench_rec - cand_bench_rec

    gate_checks = [
        {
            "criterion": "1. F1-Score Improvement on New Batch",
            "rule": "Candidate F1 > Champion F1",
            "champion_val": f"{champ_f1:.1%}",
            "candidate_val": f"{cand_f1:.1%}",
            "delta": f"{delta_f1 * 100:+.2f} pp",
            "passed": bool(delta_f1 > 0),
        },
        {
            "criterion": "2. Significant Scam Recall Improvement",
            "rule": "Candidate Recall >= Champion Recall + 15 pp",
            "champion_val": f"{champ_rec:.1%}",
            "candidate_val": f"{cand_rec:.1%}",
            "delta": f"{delta_rec * 100:+.2f} pp",
            "passed": bool(delta_rec >= 0.15),
        },
        {
            "criterion": "3. False Alarm Safety Ceiling",
            "rule": "Candidate False Alarm Rate <= 5.0%",
            "champion_val": f"{champion_inc['false_alarm']:.1%}",
            "candidate_val": f"{cand_fa:.1%}",
            "delta": "Within Safety Cap" if cand_fa <= 0.05 else "Exceeds Cap",
            "passed": bool(cand_fa <= 0.05),
        },
        {
            "criterion": "4. Benchmark Retention (No Catastrophic Forgetting)",
            "rule": "Benchmark Recall Loss <= 2.0 pp",
            "champion_val": f"{champ_bench_rec:.1%}",
            "candidate_val": f"{cand_bench_rec:.1%}",
            "delta": f"Loss: {bench_loss * 100:.2f} pp",
            "passed": bool(bench_loss <= 0.02 + 1e-9),
        },
    ]

    all_passed = all(g["passed"] for g in gate_checks)
    decision = "ELIGIBLE_FOR_PROMOTION" if all_passed else "RETAIN_CHAMPION"

    print("\n--- Quality Gate Evaluation Table ---")
    for g in gate_checks:
        status = "[PASS]" if g["passed"] else "[FAIL]"
        print(f"  {status} {g['criterion']}")
        print(f"         Rule:      {g['rule']}")
        print(f"         Champion:  {g['champion_val']}  |  Candidate: {g['candidate_val']}  |  Delta: {g['delta']}")

    print("\n" + "=" * 50)
    print(f" FINAL QUALITY GATE DECISION: {decision}")
    print("=" * 50)

    if decision == "ELIGIBLE_FOR_PROMOTION":
        print("[DECISION ACTION] Candidate meets all quality criteria and is marked ELIGIBLE FOR PROMOTION.")
        print("[SAFETY RULE] Existing champion remains deployed and is NEVER overwritten automatically.")
    else:
        print("[DECISION ACTION] Candidate failed quality criteria. Retaining existing Champion.")

    decision_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision": decision,
        "eligible_for_promotion": all_passed,
        "metric_justification": justification,
        "gates": gate_checks,
        "metrics_summary": {
            "incoming_batch": {"champion": champion_inc, "candidate": candidate_inc},
            "benchmark_holdout": {"champion": champion_bench, "candidate": candidate_bench},
        },
    }

    decision_path = MODELS_DIR / "quality_gate_decision.json"
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(decision_record, f, indent=2)

    print(f"\n[OK] Quality gate decision audit record saved to: {decision_path}")
    print("=" * 65 + "\n")
    return decision, decision_record


# ==============================================================================
# Task E: Demonstration and End-to-End Execution
# ==============================================================================

def run_retraining_pipeline():
    """Runs Tasks A -> B -> C -> D in sequence."""
    print("\n" + "#" * 65)
    print(" EXECUTING RETRAINING & QUALITY MONITORING PIPELINE")
    print("#" * 65)

    start = datetime.now(timezone.utc)
    cleaned = task_a_prepare_data()
    champ_inc, _ = task_b_evaluate_model(cleaned_data=cleaned)

    with open(MODELS_DIR / "champion_evaluation.json", "r", encoding="utf-8") as f:
        champ_bench = json.load(f)["benchmark_metrics"]

    _, cand_inc, cand_bench, _ = task_c_retrain_model(cleaned_incoming_data=cleaned)
    decision, _ = task_d_quality_gate(champ_inc, cand_inc, champ_bench, cand_bench)

    duration = (datetime.now(timezone.utc) - start).total_seconds()
    print("#" * 65)
    print(f" PIPELINE COMPLETE ({duration:.2f}s) — Quality Gate Decision: {decision}")
    print("#" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--champion", action="store_true", help="Train only the baseline champion model")
    parser.add_argument("--pipeline", action="store_true", help="Run the monitoring & retraining pipeline")
    parser.add_argument("--task", choices=["A", "B", "C", "D"], help="Run a specific task")
    args = parser.parse_args()

    if args.champion:
        train_champion()
    elif args.pipeline:
        run_retraining_pipeline()
    elif args.task == "A":
        task_a_prepare_data()
    elif args.task == "B":
        task_b_evaluate_model()
    elif args.task == "C":
        task_c_retrain_model()
    elif args.task == "D":
        task_d_quality_gate()
    else:
        # Default: train champion if needed, then run monitoring pipeline
        if not (MODELS_DIR / "champion_model.pkl").exists():
            train_champion()
        run_retraining_pipeline()


if __name__ == "__main__":
    main()
