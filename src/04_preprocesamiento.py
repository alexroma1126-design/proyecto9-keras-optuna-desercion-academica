from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path.cwd()
RAW = ROOT / "data/raw/student_dropout_success.csv"
OUT = ROOT / "data/processed/students"
MODELS = ROOT / "models/students"
REPORTS = ROOT / "results/metrics/students"

for folder in [OUT, MODELS, REPORTS]:
    folder.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

df = pd.read_csv(RAW)
X = df.drop(columns="Target")

label_map = {
    "Dropout": 0,
    "Enrolled": 1,
    "Graduate": 2
}

y = df["Target"].map(label_map)

assert y.notna().all(), "Se encontro una clase inesperada"
assert X.notna().all().all(), "Hay valores faltantes"
assert len(X) == 4424
assert X.shape[1] == 36

y = y.astype("int32").to_numpy()

# Codigos nominales documentados por UCI
categorical = [
    "Marital Status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation"
]

assert all(c in X.columns for c in categorical)

# Grupos temporales
sem1 = [c for c in X if "1st sem" in c]
sem2 = [c for c in X if "2nd sem" in c]

macro = [
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]

assert len(sem1) == 6
assert len(sem2) == 6
assert all(c in X.columns for c in macro)

base = [
    c for c in X.columns
    if c not in sem1 + sem2 + macro
]

scenarios = {
    "A_sin_semestres": base,
    "B_primer_semestre": base + sem1,
    "C_dos_semestres": base + sem1 + sem2,
    "D_completo": list(X.columns)
}

expected = {
    "A_sin_semestres": 21,
    "B_primer_semestre": 27,
    "C_dos_semestres": 33,
    "D_completo": 36
}

for name, cols in scenarios.items():
    assert len(cols) == expected[name]
    assert len(cols) == len(set(cols))

# Division unica para todos los escenarios
indices = np.arange(len(df))

trainval_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

train_idx, val_idx = train_test_split(
    trainval_idx,
    test_size=0.25,
    random_state=SEED,
    stratify=y[trainval_idx]
)

assert len(set(train_idx) & set(val_idx)) == 0
assert len(set(train_idx) & set(test_idx)) == 0
assert len(set(val_idx) & set(test_idx)) == 0

splits = {
    "train": train_idx,
    "validation": val_idx,
    "test": test_idx
}

split_labels = np.empty(len(df), dtype=object)

for name, idx in splits.items():
    split_labels[idx] = name

pd.DataFrame({
    "row_index": indices,
    "split": split_labels
}).to_csv(OUT / "indices_division.csv", index=False)

print("\n========== DIVISION DEL DATASET ==========")

for name, idx in splits.items():
    counts = np.bincount(y[idx], minlength=3)
    print(
        f"{name:12s}: {len(idx):4d} registros "
        f"| clases={counts.tolist()}"
    )

metadata = {
    "dataset": "UCI 697",
    "seed": SEED,
    "label_mapping": label_map,
    "split": {
        name: int(len(idx))
        for name, idx in splits.items()
    },
    "categorical_columns": categorical,
    "scenarios": {}
}

for name, columns in scenarios.items():

    cat_cols = [c for c in columns if c in categorical]
    num_cols = [c for c in columns if c not in categorical]

    transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32
                ),
                cat_cols
            ),
            (
                "numeric",
                StandardScaler(),
                num_cols
            )
        ],
        remainder="drop",
        sparse_threshold=0
    )

    # Ajuste EXCLUSIVO con entrenamiento
    X_train = transformer.fit_transform(
        X.iloc[train_idx][columns]
    ).astype("float32")

    X_val = transformer.transform(
        X.iloc[val_idx][columns]
    ).astype("float32")

    X_test = transformer.transform(
        X.iloc[test_idx][columns]
    ).astype("float32")

    assert X_train.shape[1] == X_val.shape[1]
    assert X_train.shape[1] == X_test.shape[1]

    assert np.isfinite(X_train).all()
    assert np.isfinite(X_val).all()
    assert np.isfinite(X_test).all()

    np.savez_compressed(
        OUT / f"{name}.npz",
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y[train_idx],
        y_val=y[val_idx],
        y_test=y[test_idx]
    )

    joblib.dump(
        transformer,
        MODELS / f"preprocessor_{name}.joblib"
    )

    metadata["scenarios"][name] = {
        "original_features": len(columns),
        "encoded_features": int(X_train.shape[1]),
        "categorical_features": cat_cols,
        "numeric_features": num_cols
    }

    print(
        f"\n{name}"
        f"\n  Variables originales: {len(columns)}"
        f"\n  Variables codificadas: {X_train.shape[1]}"
        f"\n  Entrenamiento: {X_train.shape}"
        f"\n  Validacion: {X_val.shape}"
        f"\n  Prueba: {X_test.shape}"
    )

with open(
    REPORTS / "preprocesamiento.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("\nFASE 4 COMPLETADA CORRECTAMENTE")
