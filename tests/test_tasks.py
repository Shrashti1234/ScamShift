"""
Automated Tests for Monitoring and Retraining Tasks A, B, C, D
=============================================================
"""

import json
from pathlib import Path
import pytest

from train import (
    task_a_prepare_data,
    task_b_evaluate_model,
    task_c_retrain_model,
    task_d_quality_gate,
)


def test_task_a_prepare_data(tmp_path):
    test_csv = tmp_path / "batch.csv"
    test_csv.write_text(
        "id,text,label\n"
        "1,Aapka account block OTP bhejo,1\n"
        "2,Meeting at 4pm,0\n"
        "1,Duplicate id,1\n"
        "3,Meeting at 4pm,0\n"
        "4,,1\n",
        encoding="utf-8"
    )
    cleaned = task_a_prepare_data(test_csv)
    assert len(cleaned) == 2
    assert {r["id"] for r in cleaned} == {"1", "2"}


def test_task_b_evaluate_existing_model():
    cleaned = task_a_prepare_data()
    metrics, rep = task_b_evaluate_model(cleaned_data=cleaned)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics


def test_task_c_retrain_model(tmp_path):
    cleaned = task_a_prepare_data()
    db_path = tmp_path / "test.db"

    model, inc_m, bench_m, meta = task_c_retrain_model(
        cleaned_incoming_data=cleaned,
        tracking_uri=f"sqlite:///{db_path.as_posix()}"
    )
    assert inc_m["recall"] > 0.90
    assert bench_m["recall"] > 0.90


def test_task_d_quality_gate():
    champ_inc = {"accuracy": 0.60, "precision": 0.63, "recall": 0.46, "f1": 0.53, "false_alarm": 0.26}
    cand_inc = {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "false_alarm": 0.0}
    champ_bench = {"recall": 0.95, "false_alarm": 0.004}
    cand_bench = {"recall": 0.94, "false_alarm": 0.004}

    decision, rec = task_d_quality_gate(
        champ_inc, cand_inc, champ_bench, cand_bench
    )
    assert decision == "ELIGIBLE_FOR_PROMOTION"
    assert rec["eligible_for_promotion"] is True
