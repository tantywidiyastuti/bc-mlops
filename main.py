"""
Main Pipeline Orchestrator: Hands-on 2 End-to-End MLOps
Menjalankan seluruh alur eksperimen dari L1 hingga L9 secara berurutan.
"""
import sys
import os

# Tambahkan project root ke sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data import prepare_and_save_data
from src.train import run_all_training
from src.evaluate import evaluate_and_select
from src.register import register_and_quality_gate
from src.serve_test import test_serving
from src.reproduce import audit_and_reproduce


def main():
    print("=" * 80)
    print(" PIPELINE END-TO-END MLOPS: MODEL EXPERIMENTATION & TRACKING")
    print(" Studi Kasus: Deteksi Tumor Ganas (Breast Cancer Wisconsin)")
    print(" Framework  : MLflow + Scikit-Learn")
    print("=" * 80)

    # L1: Data preparation, split, hash
    print("\n>>> TAHAP 1/6: PREPARASI DATA & HASHING (L1)")
    prepare_and_save_data()

    # L2-L4: Training, Baselines, Model Selection, Hyperparameter Tuning
    print("\n>>> TAHAP 2/6: TRAINING, BASELINES & EXPERIMENT TRACKING (L2 - L4)")
    run_all_training()

    # L5-L6: Programmatic selection & test set evaluation
    print("\n>>> TAHAP 3/6: PROGRAMMATIC SELECTION & TEST EVALUATION (L5 - L6)")
    best_run_id, best_model_uri, test_metrics = evaluate_and_select()

    # L7: Model registry & quality gate
    print("\n>>> TAHAP 4/6: MODEL REGISTRY & QUALITY GATE (L7)")
    version, val_status = register_and_quality_gate(best_model_uri, best_run_id)

    # L8: Serving test
    print("\n>>> TAHAP 5/6: SERVING MODEL & SIGNATURE VALIDATION (L8)")
    test_serving()

    # L9: Lineage audit & reproducibility verification
    print("\n>>> TAHAP 6/6: AUDIT LINEAGE & REPRODUCIBILITY (L9)")
    audit_results = audit_and_reproduce()

    print("\n" + "=" * 80)
    print(" SELURUH PIPELINE END-TO-END BERHASIL DIJALANKAN DENGAN SUKSES!")
    print(f" Model Champion: Version {version} | Quality Gate: {val_status.upper()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
