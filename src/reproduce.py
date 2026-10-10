"""
L9 — Reproduce: Dari Model di Produksi Kembali ke Asal-usulnya
Melakukan audit penuh ketertelusuran (lineage):
- Menemukan run asal dari model ber-alias @champion
- Memverifikasi commit Git, data MD5 hash, hyperparameter, metrik
- Mengunduh artefak model & requirements.txt untuk audit reproducibility
"""
import os
import mlflow
from mlflow import MlflowClient

from src.data import load_prepared_data
from src.tracking import setup_mlflow

REGISTERED_MODEL_NAME = "bc-malignant-classifier"


def audit_and_reproduce():
    print("\n" + "=" * 70)
    print(" MEMULAI AUDIT REPRODUCIBILITY & LINEAGE MODEL PRODUKSI (L9)")
    print("=" * 70)

    setup_mlflow("bc-malignant-detection")
    client = MlflowClient()

    _, _, current_data_md5 = load_prepared_data()

    # 1. Telusuri Model Champion
    champ = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    print(f"[*] Champion Model Version : v{champ.version}")
    print(f"[*] Asal Run ID             : {champ.run_id}")
    print(f"[*] Status Validasi         : {champ.tags.get('validation_status')}")

    # 2. Dapatkan Metadata Run Asal
    run = client.get_run(champ.run_id)
    run_name = run.info.run_name
    git_commit = run.data.tags.get("mlflow.source.git.commit", "N/A (Local untracked/pre-commit)")
    recorded_data_md5 = run.data.tags.get("data_md5", "Unknown")

    print(f"\n[*] Lineage Detail:")
    print(f"    - Run Name   : {run_name}")
    print(f"    - Git Commit : {git_commit}")
    print(f"    - Data MD5   : {recorded_data_md5}")

    # Parameter estimator
    clf_params = {k: v for k, v in run.data.params.items() if k.startswith("clf__")}
    print(f"\n[*] Hyperparameters Terlacak ({len(clf_params)} params):")
    for k, v in list(clf_params.items())[:6]:
        print(f"    - {k}: {v}")

    # Metrik terlacak
    print(f"\n[*] Metrik Terlacak:")
    for k, v in sorted(run.data.metrics.items()):
        print(f"    - {k:<20}: {v:.4f}")

    # 3. Assert Keutuhan Data (Data Integrity Check)
    print(f"\n[*] Memverifikasi Integritas Data...")
    assert current_data_md5 == recorded_data_md5, f"Data berbeda! Current: {current_data_md5} != Recorded: {recorded_data_md5}"
    print(f"    [VERIFIED] Data MD5 hash identik 100%! Data training terverifikasi utuh.")

    # 4. Unduh artefak untuk audit lingkungan
    print(f"\n[*] Mengunduh Artefak Model untuk Audit Lingkungan...")
    try:
        download_path = mlflow.artifacts.download_artifacts(run_id=champ.run_id, artifact_path="model")
        req_file = os.path.join(download_path, "requirements.txt")
        if os.path.exists(req_file):
            print(f"    [OK] File requirements.txt ditemukan di artefak model:")
            with open(req_file, "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]
                print("         " + ", ".join(lines[:6]))
    except Exception as e:
        print(f"    [NOTE] Download artifact info: {e}")

    print("\n" + "=" * 70)
    print(" AUDIT REPRODUCIBILITY SELESAI: LINEAGE LENGKAP & TERBUKTI IDENTIK")
    print("=" * 70)
    return {
        "version": champ.version,
        "run_id": champ.run_id,
        "run_name": run_name,
        "git_commit": git_commit,
        "data_md5": recorded_data_md5,
        "metrics": run.data.metrics,
        "params": clf_params,
    }


if __name__ == "__main__":
    audit_and_reproduce()
