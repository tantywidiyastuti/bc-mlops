"""
Helper Standar Logging MLflow untuk Eksperimen Deteksi Tumor Ganas
Mencatat parameter, metrik cross-validation 5-fold, artefak diagnostik, dan model.
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import mlflow
from mlflow.models import infer_signature

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# 5-fold Stratified CV identik untuk seluruh eksperimen (kontrol variabel)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = ["recall", "precision", "f1", "roc_auc"]


def setup_mlflow(experiment_name="bc-malignant-detection"):
    """
    Mengonfigurasi tracking URI ke server HTTP atau database SQLite lokal.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        # Default fallback langsung ke file SQLite di root proyek
        db_path = os.path.join(PROJECT_ROOT, "mlflow.db").replace("\\", "/")
        tracking_uri = f"sqlite:///{db_path}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri


def run_experiment(run_name, pipe, X_train, y_train, train_ds, common_tags, tags=None, note=""):
    """
    Fungsi standar eksekusi satu eksperimen:
    - Mencatat metadata, tags, deskripsi run, dan versi data
    - Evaluasi 5-Fold Stratified CV (mean & std)
    - Menghasilkan artefak out-of-fold confusion matrix & classification report
    - Melatih model pada seluruh data training dan mencatatnya ke MLflow Models
    """
    with mlflow.start_run(run_name=run_name) as run:
        # (a) KONTEKS: tags, deskripsi, versi data
        all_tags = {**common_tags, "candidate": "true", **(tags or {})}
        mlflow.set_tags(all_tags)
        if note:
            mlflow.set_tag("mlflow.note.content", note)
        
        # Log input dataset untuk tracking lineage data
        try:
            mlflow.log_input(train_ds, context="training")
        except Exception as e:
            mlflow.set_tag("dataset_log_error", str(e))

        # (b) PARAMETER: langkah pipeline dan hyperparameter estimator
        mlflow.log_param("pipeline_steps", " -> ".join(pipe.named_steps))
        clf_step = pipe[-1]
        clf_params = {f"clf__{k}": v for k, v in clf_step.get_params().items()}
        mlflow.log_params(clf_params)

        # (c) METRIK: mean & std dari 5-Fold CV
        cv = cross_validate(pipe, X_train, y_train, cv=CV, scoring=SCORING)
        cv_summary = {}
        for m in SCORING:
            mean_val = float(cv[f"test_{m}"].mean())
            std_val = float(cv[f"test_{m}"].std())
            mlflow.log_metric(f"cv_{m}_mean", mean_val)
            mlflow.log_metric(f"cv_{m}_std", std_val)
            cv_summary[f"cv_{m}_mean"] = mean_val
            cv_summary[f"cv_{m}_std"] = std_val

        fit_time = float(cv["fit_time"].mean())
        mlflow.log_metric("fit_time_s", fit_time)
        cv_summary["fit_time_s"] = fit_time

        # (d) ARTEFAK DIAGNOSTIK: Prediksi Out-Of-Fold
        # Catatan: Untuk estimator dummy tertentu yang tidak memiliki predict_proba, cross_val_predict tetap bekerja
        y_oof = cross_val_predict(pipe, X_train, y_train, cv=CV)
        
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay.from_predictions(
            y_train, y_oof, display_labels=["Benign (0)", "Malignant (1)"], cmap=plt.cm.Blues, ax=ax
        )
        ax.set_title(f"OOF Confusion Matrix - {run_name}")
        plt.tight_layout()

        # Simpan lokal dan unggah ke MLflow
        local_plot_path = os.path.join(PLOTS_DIR, f"{run_name}_oof_cm.png")
        fig.savefig(local_plot_path, dpi=150)
        mlflow.log_figure(fig, "plots/confusion_matrix_oof.png")
        plt.close(fig)

        clf_rep = classification_report(y_train, y_oof, zero_division=0, target_names=["Benign", "Malignant"])
        local_rep_path = os.path.join(REPORTS_DIR, f"{run_name}_oof_report.txt")
        with open(local_rep_path, "w", encoding="utf-8") as f:
            f.write(clf_rep)
        mlflow.log_text(clf_rep, "reports/classification_report_oof.txt")

        # (e) MODEL: Latih pada seluruh X_train, y_train
        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_train)
        signature = infer_signature(X_train, predictions)
        input_example = X_train.head(3)

        info = mlflow.sklearn.log_model(
            pipe,
            name="model",
            signature=signature,
            input_example=input_example,
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        mlflow.set_tag("model_uri", info.model_uri)

        print(f"  [RUN OK] {run_name:<26} | CV Recall: {cv_summary['cv_recall_mean']:.4f} (+/- {cv_summary['cv_recall_std']:.4f}) | CV Prec: {cv_summary['cv_precision_mean']:.4f} | F1: {cv_summary['cv_f1_mean']:.4f}")

        return run.info.run_id, cv_summary
