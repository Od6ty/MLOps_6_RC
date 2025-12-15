# src/models/train.py
from pathlib import Path
from typing import Dict, Any

import joblib
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

from features.preprocessing import build_dataset  # src/features/preprocessing.py


def train_new_model(
    data_raw_dir: Path,
    data_processed_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Training SVM dan Naive Bayes sebagai kandidat CANARY.
    Simpan:
      - scaler.joblib
      - svm_canary.joblib
      - nb_canary.joblib
    Return dict metrik sederhana.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    data_processed_dir.mkdir(parents=True, exist_ok=True)  # placeholder jika perlu

    X_train, X_test, y_train, y_test, scaler, classes = build_dataset(
        data_raw_dir=data_raw_dir
    )

    # -------- SVM --------
    svm_clf = svm.SVC(kernel="rbf", probability=True)
    svm_clf.fit(X_train, y_train)
    y_pred_svm = svm_clf.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)

    # -------- Naive Bayes --------
    nb_clf = GaussianNB()
    nb_clf.fit(X_train, y_train)
    y_pred_nb = nb_clf.predict(X_test)
    acc_nb = accuracy_score(y_test, y_pred_nb)

    # Simpan artefak
    joblib.dump(scaler, output_dir / "scaler.joblib")
    joblib.dump(classes, output_dir / "classes.joblib")
    joblib.dump(svm_clf, output_dir / "svm_canary.joblib")
    joblib.dump(nb_clf, output_dir / "nb_canary.joblib")

    print("[train] SVM accuracy:", acc_svm)
    print("[train] Naive Bayes accuracy:", acc_nb)

    # (opsional) simpan classification_report ke file
    report_svm = classification_report(y_test, y_pred_svm, zero_division=0)
    report_nb = classification_report(y_test, y_pred_nb, zero_division=0)
    (output_dir / "report_svm.txt").write_text(report_svm)
    (output_dir / "report_nb.txt").write_text(report_nb)

    return {
        "acc_svm": acc_svm,
        "acc_nb": acc_nb,
        "classes": classes,
    }


def train_pipeline_with_registry(
    data_raw_dir: Path,
    data_processed_dir: Path,
) -> None:
    """
    Contoh pipeline produksi:
      1. Train model baru (SVM + NB).
      2. Log ke MLflow (opsional).
      3. Tandai satu model sebagai kandidat canary.
    Detail MLflow diatur di src/models/registry.py.
    """
    # Kamu bisa menambahkan integrasi MLflow di sini.
    # Misal:
    # import mlflow
    # with mlflow.start_run() as run:
    #   metrics = train_new_model(...)
    #   mlflow.log_metric("acc_svm", metrics["acc_svm"])
    #   ...
    #   mlflow.sklearn.log_model(...)
    #   run_id = run.info.run_id
    #   -> panggil registry.set_canary_from_run(run_id)
    raise NotImplementedError(
        "Implementasi lengkap pipeline + MLflow belum diisi. "
        "Gunakan train_new_model() sebagai baseline."
    )
