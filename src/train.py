"""
L2 - L4: Eksperimen Baseline, Model Selection, dan Hyperparameter Tuning
Menjalankan seluruh run eksperimen ke MLflow:
- L2: Baseline Dummy, LogReg no-scale (H1), LogReg scaled (H2)
- L3: Model Selection (KNN, Decision Tree, Random Forest, Gradient Boosting, SVC)
- L4: Hyperparameter Tuning SVC dengan Nested Runs (GridSearchCV) (H3)
"""
import os
import pandas as pd
import mlflow
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from src.data import load_prepared_data
from src.tracking import setup_mlflow, run_experiment, CV, SCORING

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_all_training():
    print("\n" + "=" * 70)
    print(" MEMULAI PIPELINE TRAINING & EXPERIMENTATION (L2 - L4)")
    print("=" * 70)

    # 1. Konfigurasi MLflow
    tracking_uri = setup_mlflow("bc-malignant-detection")
    print(f"[*] MLflow Tracking URI: {tracking_uri}")

    # 2. Muat data L1
    train_df, test_df, data_md5 = load_prepared_data()
    X_train = train_df.drop(columns="target")
    y_train = train_df["target"]

    # Buat objek dataset MLflow untuk lineage
    try:
        train_ds = mlflow.data.from_pandas(
            train_df, source="data/train.csv", name="bc-train", targets="target"
        )
    except Exception:
        train_ds = None

    common_tags = {
        "data_md5": data_md5,
        "split_seed": "42",
        "dataset": "sklearn-breast-cancer",
    }

    # =========================================================================
    # L2: BASELINE EXPERIMENTS (H1 & H2)
    # =========================================================================
    print("\n--- [L2] Menjalankan Baseline Experiments ---")

    # 00: Dummy Classifier
    run_experiment(
        run_name="00-dummy-most-frequent",
        pipe=Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
        X_train=X_train,
        y_train=y_train,
        train_ds=train_ds,
        common_tags=common_tags,
        tags={"stage": "baseline", "model_family": "dummy"},
        note="Batas bawah: selalu menebak kelas mayoritas (benign).",
    )

    # 01: LogReg No-Scale (Menguji Hipotesis H1)
    run_experiment(
        run_name="01-logreg-noscale",
        pipe=Pipeline([("clf", LogisticRegression(max_iter=5000, random_state=42))]),
        X_train=X_train,
        y_train=y_train,
        train_ds=train_ds,
        common_tags=common_tags,
        tags={"stage": "baseline", "model_family": "logreg"},
        note="H1: model linear tanpa preprocessing.",
    )

    # 02: LogReg Scaled (Menguji Hipotesis H2)
    run_experiment(
        run_name="02-logreg-scaled",
        pipe=Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=42)),
        ]),
        X_train=X_train,
        y_train=y_train,
        train_ds=train_ds,
        common_tags=common_tags,
        tags={"stage": "baseline", "model_family": "logreg"},
        note="H2: hanya menambah StandardScaler, variabel lain tetap.",
    )

    # =========================================================================
    # L3: MODEL SELECTION (5 Algoritma Sederhana)
    # =========================================================================
    print("\n--- [L3] Menjalankan Model Selection (5 Algoritma) ---")
    candidates = {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42),
        "gb": GradientBoostingClassifier(random_state=42),
        "svc": SVC(kernel="rbf", random_state=42),
    }

    for i, (family, clf) in enumerate(candidates.items(), start=3):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        run_experiment(
            run_name=f"{i:02d}-{family}-default",
            pipe=pipe,
            X_train=X_train,
            y_train=y_train,
            train_ds=train_ds,
            common_tags=common_tags,
            tags={"stage": "model-selection", "model_family": family},
            note="Hyperparameter default; preprocessing identik untuk semua.",
        )

    # =========================================================================
    # L4: HYPERPARAMETER TUNING DENGAN NESTED RUNS (SVC GridSearch) (H3)
    # =========================================================================
    print("\n--- [L4] Menjalankan Hyperparameter Tuning dengan Nested Runs (H3) ---")
    pipe_svc = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", random_state=42))])
    grid = {
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", 0.01, 0.001],
    }  # 12 kombinasi

    gs = GridSearchCV(pipe_svc, grid, cv=CV, scoring=SCORING, refit="recall", n_jobs=-1)

    parent_run_name = "10-svc-gridsearch"
    with mlflow.start_run(
        run_name=parent_run_name,
        tags={
            **common_tags,
            "stage": "tuning",
            "model_family": "svc",
            "candidate": "true",
        },
    ) as parent_run:
        if train_ds is not None:
            try:
                mlflow.log_input(train_ds, context="training")
            except Exception:
                pass
        mlflow.log_dict(grid, "search_space.json")
        mlflow.set_tag("mlflow.note.content", "H3: Tuning hyperparameter SVC via GridSearchCV (12 kombinasi).")

        print("  [*] Menjalankan GridSearchCV fit...")
        gs.fit(X_train, y_train)
        res = gs.cv_results_

        print(f"  [*] Mencatat 12 child runs (trials)...")
        for i, params in enumerate(res["params"]):
            child_name = f"svc-trial-{i:02d}"
            with mlflow.start_run(run_name=child_name, nested=True):
                mlflow.log_params(params)
                mlflow.log_metrics({f"cv_{m}_mean": float(res[f"mean_test_{m}"][i]) for m in SCORING})
                mlflow.log_metrics({f"cv_{m}_std": float(res[f"std_test_{m}"][i]) for m in SCORING})

        # Parent run: Ringkasan konfigurasi & metrik terbaik
        b = gs.best_index_
        mlflow.log_params(gs.best_params_)
        for m in SCORING:
            mlflow.log_metric(f"cv_{m}_mean", float(res[f"mean_test_{m}"][b]))
            mlflow.log_metric(f"cv_{m}_std", float(res[f"std_test_{m}"][b]))

        # Log model terbaik ke parent run
        best_model = gs.best_estimator_
        sig = infer_signature(X_train, best_model.predict(X_train))
        info = mlflow.sklearn.log_model(
            best_model,
            name="model",
            signature=sig,
            input_example=X_train.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        mlflow.set_tag("model_uri", info.model_uri)

        print(f"  [PARENT RUN OK] Best Params: {gs.best_params_}")
        print(f"                  CV Recall  : {res['mean_test_recall'][b]:.4f} (+/- {res['std_test_recall'][b]:.4f})")
        print(f"                  CV Prec    : {res['mean_test_precision'][b]:.4f}")
        print(f"                  CV F1      : {res['mean_test_f1'][b]:.4f}")

    print("\n" + "=" * 70)
    print(" SELURUH EKSPERIMEN L2 - L4 SELESAI DICATAT KE MLFLOW!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_training()
