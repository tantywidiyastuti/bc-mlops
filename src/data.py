"""
L1 — Data: Load, Preprocess, Split Tetap (Stratified), dan Hashing (MD5)
"""
import os
import hashlib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def prepare_and_save_data():
    """
    Memuat dataset Wisconsin Breast Cancer, membalik target (1=malignant),
    menghitung MD5 hash seluruh data, membagi train/test (80/20 stratified),
    dan menyimpannya ke format CSV.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Load dataset Breast Cancer
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()

    # 2. Inversi label: 1 = malignant (tumor ganas), 0 = benign (jinak)
    df["target"] = 1 - df["target"]

    # 3. Sidik jari data (MD5 hash)
    data_md5 = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

    # 4. Split tetap 80% train, 20% test stratified
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["target"], random_state=42
    )

    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("=" * 60)
    print(f"[L1] Data Preparation Berhasil Diselesaikan!")
    print(f"     Total Data : {len(df)} baris x {df.shape[1]} kolom")
    print(f"     Train Set  : {len(train_df)} baris (80%) -> {train_path}")
    print(f"     Test Set   : {len(test_df)} baris (20%) -> {test_path}")
    print(f"     Data MD5   : {data_md5}")
    print(f"     Distribusi Train: Malignant={train_df['target'].sum()}, Benign={(train_df['target']==0).sum()}")
    print("=" * 60)

    return train_df, test_df, data_md5


def load_prepared_data():
    """
    Memuat data train dan test yang sudah tersimpan, serta mengembalikan
    X_train, y_train, X_test, y_test, dan data_md5.
    """
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        return prepare_and_save_data()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Rekonstruksi seluruh df untuk menghitung ulang md5 hash
    df_combined = pd.concat([train_df, test_df]).sort_index()
    # Atur ulang sesuai hash asli
    data_md5 = hashlib.md5(pd.util.hash_pandas_object(df_combined, index=True).values).hexdigest()

    return train_df, test_df, data_md5


if __name__ == "__main__":
    prepare_and_save_data()
