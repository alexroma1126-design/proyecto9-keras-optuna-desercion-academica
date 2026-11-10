import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score

ROOT = Path.cwd()

RAW = ROOT / "data/raw/student_dropout_success.csv"
DATA = ROOT / "data/processed/students/D_completo.npz"

MODEL = ROOT / (
    "models/students/robustez/Moderado_seed_42.keras"
)

PREPROCESSOR = ROOT / (
    "models/students/preprocessor_D_completo.joblib"
)

METRICS = ROOT / "results/metrics/students"
FIGURES = ROOT / "results/figures/students"

METRICS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

for file in [RAW, DATA, MODEL, PREPROCESSOR]:
    if not file.exists():
        raise FileNotFoundError(file)

print("\n========== FASE 9 ==========")
print("Analisis de importancia por permutacion")
print("Modelo: D - Balanceo moderado")
print("Conjunto: Validacion")
print("Prueba final: NO utilizada")

# Cargar datos originales
df = pd.read_csv(RAW)

labels = {
    "Dropout": 0,
    "Enrolled": 1,
    "Graduate": 2
}

X = df.drop(columns="Target")
y = df["Target"].map(labels).to_numpy(dtype=np.int32)

# Reconstruir exactamente la division original
indices = np.arange(len(df))

trainval_idx, _ = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y
)

_, val_idx = train_test_split(
    trainval_idx,
    test_size=0.25,
    random_state=42,
    stratify=y[trainval_idx]
)

X_val = X.iloc[val_idx].copy()
y_val = y[val_idx]

preprocessor = joblib.load(PREPROCESSOR)

with np.load(DATA) as saved:
    saved_X = saved["X_val"]
    saved_y = saved["y_val"]

check_X = preprocessor.transform(
    X_val
).astype("float32")

assert np.array_equal(y_val, saved_y)
assert np.allclose(check_X, saved_X, atol=1e-5)

print("Validacion de particiones: CORRECTA")
print("Registros de validacion:", len(y_val))
print("Variables originales:", X_val.shape[1])

# Cargar modelo ya entrenado
tf.keras.backend.clear_session()

model = tf.keras.models.load_model(
    MODEL, compile=False
)

def predict(dataframe):
    array = preprocessor.transform(
        dataframe
    ).astype("float32")

    probabilities = model(
        array, training=False
    ).numpy()

    return np.argmax(probabilities, axis=1)

# Rendimiento original
original_pred = predict(X_val)

base_f1 = f1_score(
    y_val,
    original_pred,
    average="macro",
    zero_division=0
)

base_recall = recall_score(
    y_val,
    original_pred,
    labels=[0, 1, 2],
    average=None,
    zero_division=0
)[1]

print("\nMacro-F1 original:", round(base_f1, 4))
print("Recall Enrolled:", round(base_recall, 4))

# Permutacion de cada variable original
rng = np.random.default_rng(2026)

repeticiones = 5
results = []

print("\nAnalizando las 36 variables...")

for number, column in enumerate(X_val.columns, 1):

    delta_f1 = []
    delta_recall = []

    for repetition in range(repeticiones):

        modified = X_val.copy()

        modified[column] = rng.permutation(
            X_val[column].to_numpy()
        )

        pred = predict(modified)

        score = f1_score(
            y_val,
            pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_val,
            pred,
            labels=[0, 1, 2],
            average=None,
            zero_division=0
        )[1]

        delta_f1.append(base_f1 - score)
        delta_recall.append(base_recall - recall)

    results.append({
        "variable": column,
        "delta_f1_media": float(np.mean(delta_f1)),
        "delta_f1_std": float(
            np.std(delta_f1, ddof=1)
        ),
        "delta_recall_enrolled_media": float(
            np.mean(delta_recall)
        )
    })

    print(
        f"[{number:02d}/36] {column}",
        flush=True
    )

importance = pd.DataFrame(results)

importance = importance.sort_values(
    "delta_f1_media",
    ascending=False
)

output = METRICS / "importancia_permutacion.csv"

importance.to_csv(output, index=False)

# Grafico con las doce variables principales
top = importance.head(12).sort_values(
    "delta_f1_media"
)

fig, ax = plt.subplots(figsize=(12, 7))

ax.barh(
    top["variable"],
    top["delta_f1_media"]
)

ax.axvline(0, linewidth=1)

ax.set_xlabel(
    "Reduccion media del Macro-F1"
)

ax.set_title(
    "Importancia de variables - Modelo D Moderado"
)

fig.tight_layout()

figure = FIGURES / "09_importancia_permutacion.png"

fig.savefig(
    figure,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("\n========== TOP 12 VARIABLES ==========")

print(
    importance[
        [
            "variable",
            "delta_f1_media",
            "delta_recall_enrolled_media"
        ]
    ].head(12).round(5).to_string(index=False)
)

print("\nCSV:", output)
print("Grafico:", figure)
print("\nFASE 9 COMPLETADA CORRECTAMENTE")
