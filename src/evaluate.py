"""
L5 — Analisis Programatik (Pilih Kandidat Terbaik)
L6 — Evaluasi Final di Test Set (Hanya Sekali)
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay, classification_report
)
import mlflow

from src.data import load_prepared_data
from src.tracking import setup_mlflow

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def evaluate_and_select():
    print("\n" + "=" * 70)
    print(" MEMULAI ANALISIS PROGRAMATIK (L5) & EVALUASI TEST SET (L6)")
    print("=" * 70)

    # Inisialisasi koneksi MLflow
    setup_mlflow("bc-malignant-detection")

    # =========================================================================
    # L5: ANALISIS PROGRAMATIK (Query MLflow Runs)
    # =========================================================================
    print("\n--- [L5] Query MLflow Runs untuk Seleksi Kandidat Terbaik ---")
    runs = mlflow.search_runs(
        experiment_names=["bc-malignant-detection"],
        filter_string=(
            "tags.candidate = 'true' "
            "and metrics.cv_precision_mean >= 0.90"
        ),
        order_by=[
            "metrics.cv_recall_mean DESC",
            "metrics.cv_f1_mean DESC"
        ],
    )

    if runs.empty:
        raise RuntimeError("Tidak ada run yang memenuhi kriteria satisficing (cv_precision_mean >= 0.90)!")

    cols = [
        "tags.mlflow.runName", "tags.model_family",
        "metrics.cv_recall_mean", "metrics.cv_recall_std", "metrics.cv_precision_mean", "metrics.cv_f1_mean"
    ]
    avail_cols = [c for c in cols if c in runs.columns]
    print("\nTop 5 Kandidat Lolos Filter Satisficing Metric (Precision >= 0.90):")
    print(runs[avail_cols].head(5).to_string(index=False))

    best_run = runs.iloc[0]
    best_run_id = best_run["run_id"]
    best_run_name = best_run.get("tags.mlflow.runName", "Unknown")
    
    # Ambil model_uri dari tag atau fallback ke runs:/{best_run_id}/model
    best_model_uri = best_run.get("tags.model_uri", f"runs:/{best_run_id}/model")
    if pd.isna(best_model_uri) or not best_model_uri:
        best_model_uri = f"runs:/{best_run_id}/model"

    print(f"\n[*] Kandidat Terpilih: {best_run_name} (Run ID: {best_run_id})")
    print(f"[*] Model URI        : {best_model_uri}")
    print(f"[*] CV Recall Mean   : {best_run['metrics.cv_recall_mean']:.4f}")
    print(f"[*] CV Precision Mean: {best_run['metrics.cv_precision_mean']:.4f}")

    # =========================================================================
    # L6: EVALUASI FINAL DI TEST SET (Hanya Sekali)
    # =========================================================================
    print("\n--- [L6] Evaluasi Final pada Test Set (Kunci Data Terbuka Sekali) ---")
    train_df, test_df, data_md5 = load_prepared_data()
    X_test = test_df.drop(columns="target")
    y_test = test_df["target"]

    print(f"[*] Memuat model dari MLflow: {best_model_uri}")
    model = mlflow.sklearn.load_model(best_model_uri)

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X_test)
    else:
        score = y_pred

    test_metrics = {
        "test_recall": float(recall_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, score)),
    }

    print("\nHasil Evaluasi pada Data Uji (Test Set):")
    for k, v in test_metrics.items():
        print(f"  {k:<16}: {v:.4f}")

    # Catat metrik test ke RUN YANG SAMA di MLflow
    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_metrics(test_metrics)
        try:
            test_ds = mlflow.data.from_pandas(
                test_df, source="data/test.csv", name="bc-test", targets="target"
            )
            mlflow.log_input(test_ds, context="testing")
        except Exception:
            pass

        # Visualisasi Confusion Matrix pada Test Set
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=["Benign (0)", "Malignant (1)"], cmap=plt.cm.Greens, ax=ax
        )
        ax.set_title(f"Test Confusion Matrix - {best_run_name}")
        plt.tight_layout()

        local_plot_path = os.path.join(PLOTS_DIR, "confusion_matrix_test.png")
        fig.savefig(local_plot_path, dpi=150)
        mlflow.log_figure(fig, "plots/confusion_matrix_test.png")
        plt.close(fig)

        test_report = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"])
        local_rep_path = os.path.join(REPORTS_DIR, "classification_report_test.txt")
        with open(local_rep_path, "w", encoding="utf-8") as f:
            f.write(test_report)
        mlflow.log_text(test_report, "reports/classification_report_test.txt")

        mlflow.set_tag("test_evaluated", "true")

    print(f"\n[OK] Metrik dan artefak test set berhasil dicatat ke Run ID {best_run_id}")
    return best_run_id, best_model_uri, test_metrics


if __name__ == "__main__":
    evaluate_and_select()
