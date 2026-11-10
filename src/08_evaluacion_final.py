import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    recall_score,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/students"
MODELS = ROOT / "models/students"
METRICS = ROOT / "results/metrics/students"
FIGURES = ROOT / "results/figures/students"

FINAL = METRICS / "evaluacion_final_test.csv"
PROTOCOL = METRICS / "protocolo_final.json"

CLASSES = ["Dropout", "Enrolled", "Graduate"]
SCENARIOS = [
    "A_sin_semestres",
    "B_primer_semestre",
    "C_dos_semestres",
    "D_completo"
]

tf.get_logger().setLevel("ERROR")

# Evitar repetir la evaluacion ya cerrada
if FINAL.exists():
    print("\nLa evaluacion final ya existe.")
    print("No se repetira el analisis del test.")
    print(pd.read_csv(FINAL).round(4).to_string(index=False))
    sys.exit(0)

# Registrar las decisiones anteriores a la prueba
protocol = {
    "dataset": "UCI 697",
    "test_size": 885,
    "primary_scenario": "D_completo",
    "primary_model": "Moderado",
    "primary_seed": 42,
    "selection_basis":
        "Mayor macro-F1 promedio en validacion "
        "de tres semillas entre estrategias de pesos",
    "primary_metric": "macro_f1",
    "primary_comparator": "Optuna",
    "secondary_model": "Balanceado",
    "secondary_objective": "Recall Enrolled",
    "bootstrap_resamples": 2000,
    "note":
        "No seleccionar ni reajustar modelos "
        "utilizando los resultados de test"
}

if PROTOCOL.exists():
    saved = json.loads(
        PROTOCOL.read_text(encoding="utf-8")
    )
    if saved != protocol:
        raise RuntimeError(
            "El protocolo existente no coincide. "
            "Revisar antes de evaluar."
        )
else:
    PROTOCOL.write_text(
        json.dumps(
            protocol, indent=2, ensure_ascii=False
        ),
        encoding="utf-8"
    )

specs = []

for scenario in SCENARIOS:
    specs.append((
        scenario, "Base",
        MODELS / f"baseline_{scenario}.keras"
    ))
    specs.append((
        scenario, "Optuna",
        MODELS / f"optuna_{scenario}.keras"
    ))

for strategy in ["Moderado", "Balanceado"]:
    specs.append((
        "D_completo",
        strategy,
        MODELS / "robustez" /
        f"{strategy}_seed_42.keras"
    ))

# Comprobar archivos antes de acceder a test
for _, _, path in specs:
    if not path.exists():
        raise FileNotFoundError(path)

results = []
per_class = []
predictions = {}
reference_y = None

print("\n========== EVALUACION FINAL ==========")
print("Modelos:", len(specs))
print("Metrica principal: Macro-F1")
print("Protocolo:", PROTOCOL.name)

for scenario, name, model_path in specs:

    with np.load(DATA / f"{scenario}.npz") as data:
        X_test = data["X_test"].astype("float32")
        y_test = data["y_test"].astype("int32")

    if reference_y is None:
        reference_y = y_test.copy()
    else:
        assert np.array_equal(
            reference_y, y_test
        ), "Las particiones test no coinciden"

    assert len(y_test) == 885
    assert set(np.unique(y_test)) == {0, 1, 2}

    tf.keras.backend.clear_session()

    model = tf.keras.models.load_model(
        model_path, compile=False
    )

    assert model.input_shape[-1] == X_test.shape[1]

    probabilities = model.predict(
        X_test, verbose=0
    )

    predictions_test = np.argmax(
        probabilities, axis=1
    )

    predictions[(scenario, name)] = predictions_test

    report = classification_report(
        y_test,
        predictions_test,
        labels=[0, 1, 2],
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )

    result = {
        "escenario": scenario,
        "modelo": name,
        "accuracy": accuracy_score(
            y_test, predictions_test
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions_test
        ),
        "macro_f1": f1_score(
            y_test,
            predictions_test,
            average="macro",
            zero_division=0
        ),
        "recall_dropout": report["Dropout"]["recall"],
        "recall_enrolled": report["Enrolled"]["recall"],
        "recall_graduate": report["Graduate"]["recall"]
    }

    results.append(result)

    for cls in CLASSES:
        per_class.append({
            "escenario": scenario,
            "modelo": name,
            "clase": cls,
            "precision": report[cls]["precision"],
            "recall": report[cls]["recall"],
            "f1": report[cls]["f1-score"],
            "support": report[cls]["support"]
        })

    # Matrices de confusion de los modelos D
    if scenario == "D_completo":
        fig, ax = plt.subplots(figsize=(7, 6))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions_test,
            labels=[0, 1, 2],
            display_labels=CLASSES,
            cmap="Blues",
            colorbar=False,
            ax=ax
        )

        ax.set_title(
            f"Test - D_completo - {name}"
        )
        fig.tight_layout()
        fig.savefig(
            FIGURES / f"test_matriz_D_{name}.png",
            dpi=180
        )
        plt.close(fig)

    print(
        f"{scenario:20s} {name:12s} "
        f"F1={result['macro_f1']:.4f} "
        f"Recall Enrolled={result['recall_enrolled']:.4f}"
    )

results_df = pd.DataFrame(results)

# Comparacion pareada de modelos fijados previamente
y = reference_y

p_primary = predictions[
    ("D_completo", "Moderado")
]
p_comparator = predictions[
    ("D_completo", "Optuna")
]

rng = np.random.default_rng(2026)
n = len(y)
n_boot = 2000

f1_differences = []
enrolled_differences = []

for _ in range(n_boot):

    sample = rng.integers(0, n, size=n)

    actual = y[sample]
    primary = p_primary[sample]
    comparator = p_comparator[sample]

    delta_f1 = (
        f1_score(
            actual, primary,
            average="macro", zero_division=0
        )
        -
        f1_score(
            actual, comparator,
            average="macro", zero_division=0
        )
    )

    recall_primary = recall_score(
        actual, primary,
        labels=[0, 1, 2],
        average=None, zero_division=0
    )[1]

    recall_comparator = recall_score(
        actual, comparator,
        labels=[0, 1, 2],
        average=None, zero_division=0
    )[1]

    f1_differences.append(delta_f1)
    enrolled_differences.append(
        recall_primary - recall_comparator
    )

ci_f1 = np.percentile(
    f1_differences, [2.5, 97.5]
)

ci_enrolled = np.percentile(
    enrolled_differences, [2.5, 97.5]
)

bootstrap = {
    "comparison": "D Moderado minus D Optuna",
    "confidence_level": 0.95,
    "method": "paired_percentile_bootstrap",
    "resamples": n_boot,
    "macro_f1_delta_ci": ci_f1.tolist(),
    "recall_enrolled_delta_ci": ci_enrolled.tolist(),
    "interpretation":
        "Intervalos exploratorios de una unica "
        "particion test; no implican causalidad"
}

results_df.to_csv(FINAL, index=False)

pd.DataFrame(per_class).to_csv(
    METRICS / "evaluacion_final_por_clase.csv",
    index=False
)

pd.DataFrame({
    "y_real": y,
    "D_optuna": p_comparator,
    "D_moderado": p_primary,
    "D_balanceado": predictions[
        ("D_completo", "Balanceado")
    ]
}).to_csv(
    METRICS / "predicciones_finales_D.csv",
    index=False
)

(METRICS / "bootstrap_final.json").write_text(
    json.dumps(
        bootstrap, indent=2, ensure_ascii=False
    ),
    encoding="utf-8"
)

# Figura comparativa del conjunto de prueba
pivot = results_df[
    results_df["modelo"].isin(["Base", "Optuna"])
].pivot(
    index="escenario",
    columns="modelo",
    values="macro_f1"
).reindex(SCENARIOS)

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(SCENARIOS))
width = 0.35

ax.bar(
    x - width/2,
    pivot["Base"],
    width,
    label="Keras base"
)
ax.bar(
    x + width/2,
    pivot["Optuna"],
    width,
    label="Optuna"
)

ax.set_xticks(x)
ax.set_xticklabels(["A", "B", "C", "D"])
ax.set_ylim(0, 1)
ax.set_ylabel("Macro-F1")
ax.set_title("Comparacion final en test")
ax.legend()

fig.tight_layout()
fig.savefig(
    FIGURES / "test_comparacion_escenarios.png",
    dpi=180
)
plt.close(fig)

print("\n========== RESULTADOS FINALES ==========")

print(
    results_df.round(4).to_string(index=False)
)

print("\n========== BOOTSTRAP 95 % ==========")
print(
    "Diferencia macro-F1, Moderado - Optuna:",
    np.round(ci_f1, 4)
)
print(
    "Diferencia recall Enrolled, Moderado - Optuna:",
    np.round(ci_enrolled, 4)
)

print("\nEVALUACION FINAL COMPLETADA")
print("No usar test para nuevas selecciones")
