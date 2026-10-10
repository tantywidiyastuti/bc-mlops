"""
L7 — Model Registry + Quality Gate Otomatis
Mendaftarkan model kandidat terpilih ke MLflow Model Registry,
memberi alias @challenger, mengevaluasi Quality Gate, dan mempromosikannya ke @champion.
"""
import os
import mlflow
from mlflow import MlflowClient

from src.data import load_prepared_data
from src.tracking import setup_mlflow

REGISTERED_MODEL_NAME = "bc-malignant-classifier"
MIN_PRECISION = 0.90
MARGIN = 0.005


def get_champion_recall(client, name):
    """
    Mengambil test_recall dari model yang sedang memegang alias @champion.
    Jika belum ada champion (versi pertama), return -1.0.
    """
    try:
        champ = client.get_model_version_by_alias(name, "champion")
        champ_run = client.get_run(champ.run_id)
        return float(champ_run.data.metrics.get("test_recall", -1.0))
    except Exception:
        return -1.0


def register_and_quality_gate(best_model_uri, best_run_id):
    print("\n" + "=" * 70)
    print(" MEMULAI MODEL REGISTRY & QUALITY GATE (L7)")
    print("=" * 70)

    setup_mlflow("bc-malignant-detection")
    client = MlflowClient()

    _, _, data_md5 = load_prepared_data()

    # 1. Registrasi Model ke Model Registry
    print(f"[*] Mendaftarkan model ({best_model_uri}) dengan nama: '{REGISTERED_MODEL_NAME}'...")
    mv = mlflow.register_model(best_model_uri, REGISTERED_MODEL_NAME)
    print(f"[OK] Model Version terdaftar: Version {mv.version}")

    # 2. Update Anotasi & Tags
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=mv.version,
        description=f"Dipilih otomatis: max cv_recall dgn precision>=0.90. data_md5={data_md5}"
    )
    client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "data_md5", data_md5)
    
    # 3. Tetapkan Alias @challenger
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", mv.version)
    print(f"[*] Alias '@challenger' ditetapkan untuk versi {mv.version}")

    # 4. Ambil metrik test dari run
    run_data = client.get_run(best_run_id).data
    test_recall = float(run_data.metrics.get("test_recall", 0.0))
    test_precision = float(run_data.metrics.get("test_precision", 0.0))
    champ_rec = get_champion_recall(client, REGISTERED_MODEL_NAME)

    print(f"\n[*] Evaluasi Quality Gate:")
    print(f"    - Current Challenger Test Precision: {test_precision:.4f} (Syarat >= {MIN_PRECISION:.2f})")
    print(f"    - Current Challenger Test Recall   : {test_recall:.4f}")
    print(f"    - Current Champion Test Recall     : {champ_rec:.4f} (Margin: +{MARGIN})")

    # 5. Cek Quality Gate
    passed = (test_precision >= MIN_PRECISION) and (test_recall >= champ_rec + MARGIN)

    validation_status = "approved" if passed else "rejected"
    client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "validation_status", validation_status)
    print(f"[*] Validation Status: {validation_status.upper()}")

    # 6. Promosi ke @champion jika lolos
    if passed:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", mv.version)
        print(f"[SUCCESS] MODEL VERSI {mv.version} DIPROMOSIKAN SEBAGAI '@champion'!")
    else:
        print(f"[FAILED] Model tidak memenuhi Quality Gate, tidak dipromosikan ke champion.")

    return mv.version, validation_status


if __name__ == "__main__":
    # Test jika dijalankan langsung
    setup_mlflow("bc-malignant-detection")
    client = MlflowClient()
    runs = mlflow.search_runs(
        experiment_names=["bc-malignant-detection"],
        filter_string="tags.candidate = 'true' and metrics.cv_precision_mean >= 0.90",
        order_by=["metrics.cv_recall_mean DESC", "metrics.cv_f1_mean DESC"]
    )
    if not runs.empty:
        best_run_id = runs.iloc[0]["run_id"]
        best_model_uri = runs.iloc[0].get("tags.model_uri", f"runs:/{best_run_id}/model")
        register_and_quality_gate(best_model_uri, best_run_id)
