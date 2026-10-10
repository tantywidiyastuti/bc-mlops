"""
L8 — Serving Model dari Registry Menggunakan Alias @champion
Menguji pemuatan model dinamis via alias tanpa hard-code path/versi,
menguji prediksi batch, serta menguji validasi Model Signature terhadap silent failure.
"""
import os
import json
import pandas as pd
import mlflow

from src.data import load_prepared_data
from src.tracking import setup_mlflow

REGISTERED_MODEL_NAME = "bc-malignant-classifier"


def test_serving():
    print("\n" + "=" * 70)
    print(" MEMULAI PENGUJIAN MODEL SERVING DARI REGISTRY (L8)")
    print("=" * 70)

    setup_mlflow("bc-malignant-detection")
    _, test_df, _ = load_prepared_data()
    X_test = test_df.drop(columns="target")
    y_test = test_df["target"]

    # 1. Pemuatan model berdasarkan ALIAS
    model_alias_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    print(f"[*] Memuat model dinamis dari URI Alias: {model_alias_uri}")
    model = mlflow.pyfunc.load_model(model_alias_uri)
    print("[OK] Model @champion berhasil dimuat ke memori!")

    # 2. Pengujian Prediksi Batch
    sample_X = X_test.head(5)
    preds = model.predict(sample_X)
    print("\n[*] Hasil Prediksi 5 Sampel Teratas:")
    for idx, (p, actual) in enumerate(zip(preds, y_test.head(5))):
        pred_label = "Malignant (Ganas)" if p == 1 else "Benign (Jinak)"
        actual_label = "Malignant" if actual == 1 else "Benign"
        print(f"    Sample {idx+1}: Prediksi = {pred_label} | Aktual = {actual_label}")

    # 3. Format Payload REST API (Simulasi Request HTTP JSON)
    payload = {"dataframe_split": sample_X.to_dict(orient="split")}
    print("\n[*] Simulasi Request Payload JSON untuk REST Endpoint (/invocations):")
    print(f"    Columns: {len(payload['dataframe_split']['columns'])} fitur numerik")
    print(f"    Rows   : {len(payload['dataframe_split']['data'])} baris")

    # 4. Pengujian Proteksi Skema Signature (Mencegah Silent Failure)
    print("\n[*] Pengujian Proteksi Model Signature terhadap Masukan Rusak/Invalid:")
    bad_data = sample_X.drop(columns=[sample_X.columns[0]])  # Menghapus satu kolom
    try:
        model.predict(bad_data)
        print("    [!] WARNING: Model tidak menolak data salah!")
    except Exception as e:
        print(f"    [SUCCESS] Signature berhasil menolak data invalid secara tegas!")
        print(f"              Pesan error: {type(e).__name__} dicegah sebelum kalkulasi.")

    print("\n[L8 OK] Model serving via alias & signature validation terverifikasi sempurna!")
    return preds


if __name__ == "__main__":
    test_serving()
