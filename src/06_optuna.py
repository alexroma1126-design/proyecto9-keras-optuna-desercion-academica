import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import json
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

DATA = ROOT / "data/processed/students"
MODELS = ROOT / "models/students"
METRICS = ROOT / "results/metrics/students"
FIGURES = ROOT / "results/figures/students"

SCENARIOS = [
    "A_sin_semestres",
    "B_primer_semestre",
    "C_dos_semestres",
    "D_completo"
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--escenario",
    choices=SCENARIOS,
    default="D_completo"
)
parser.add_argument(
    "--ensayos", type=int, default=20
)
args = parser.parse_args()

if args.ensayos < 1:
    raise ValueError("El numero de ensayos debe ser positivo")

SCENARIO = args.escenario
SEED = 42

tf.get_logger().setLevel("ERROR")
optuna.logging.set_verbosity(optuna.logging.WARNING)

np.random.seed(SEED)

with np.load(DATA / f"{SCENARIO}.npz") as d:
    X_train = d["X_train"].astype("float32")
    X_val = d["X_val"].astype("float32")
    y_train = d["y_train"].astype("int32")
    y_val = d["y_val"].astype("int32")

assert X_train.shape[1] == X_val.shape[1]

BEST_MODEL = MODELS / f"optuna_{SCENARIO}.keras"
BEST_HISTORY = METRICS / f"optuna_historia_{SCENARIO}.csv"

# Almacenamiento persistente
database = f"optuna_{SCENARIO}.db"
storage = (
    "sqlite:///results/metrics/students/" + database
)

sampler = optuna.samplers.TPESampler(
    seed=SEED,
    n_startup_trials=5
)

study = optuna.create_study(
    study_name=f"optimizacion_v1_{SCENARIO}",
    direction="maximize",
    sampler=sampler,
    pruner=optuna.pruners.NopPruner(),
    storage=storage,
    load_if_exists=True
)

completed = [
    t for t in study.trials
    if t.state == optuna.trial.TrialState.COMPLETE
]

if completed and not BEST_MODEL.exists():
    raise RuntimeError(
        "Existe un estudio previo pero falta su "
        "modelo guardado. Revisar antes de continuar."
    )

def objective(trial):

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    n_layers = trial.suggest_int(
        "n_layers", 1, 3
    )

    dropout = trial.suggest_float(
        "dropout", 0.0, 0.4, step=0.1
    )

    l2_value = trial.suggest_float(
        "l2", 1e-6, 1e-3, log=True
    )

    learning_rate = trial.suggest_float(
        "learning_rate", 1e-4, 5e-3, log=True
    )

    batch_size = trial.suggest_categorical(
        "batch_size", [16, 32, 64]
    )

    model = tf.keras.Sequential()
    model.add(
        tf.keras.Input(shape=(X_train.shape[1],))
    )

    for i in range(n_layers):
        units = trial.suggest_categorical(
            f"units_{i}", [16, 32, 64, 128]
        )

        model.add(
            tf.keras.layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(
                    l2_value
                )
            )
        )

        if dropout > 0:
            model.add(
                tf.keras.layers.Dropout(dropout)
            )

    model.add(
        tf.keras.layers.Dense(
            3, activation="softmax"
        )
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=batch_size,
        callbacks=[early_stop],
        shuffle=True,
        verbose=0
    )

    proba = model.predict(X_val, verbose=0)
    pred = np.argmax(proba, axis=1)

    score = float(
        f1_score(
            y_val, pred,
            average="macro",
            zero_division=0
        )
    )

    accuracy = float(
        accuracy_score(y_val, pred)
    )

    epochs = len(history.history["loss"])

    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("epochs", epochs)
    trial.set_user_attr(
        "model_parameters", int(model.count_params())
    )

    previous = [
        t.value
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
    ]

    previous_best = max(previous, default=-1.0)

    if score > previous_best:
        model.save(BEST_MODEL)

        pd.DataFrame(
            history.history
        ).to_csv(BEST_HISTORY, index=False)

    print(
        f"Ensayo {trial.number:03d} | "
        f"F1 macro={score:.4f} | "
        f"Accuracy={accuracy:.4f} | "
        f"Epocas={epochs}",
        flush=True
    )

    del model
    return score

previous_trials = len(study.trials)
remaining = max(0, args.ensayos - previous_trials)

print("\n========== OPTIMIZACION ==========")
print("Escenario:", SCENARIO)
print("Entradas:", X_train.shape[1])
print("Ensayos ya registrados:", previous_trials)
print("Nuevos ensayos:", remaining)
print("Metodo: TPE")
print("Objetivo: maximizar macro-F1")

if remaining > 0:
    study.optimize(
        objective,
        n_trials=remaining,
        n_jobs=1,
        gc_after_trial=True
    )

completed = [
    t for t in study.trials
    if t.state == optuna.trial.TrialState.COMPLETE
]

if not completed:
    raise RuntimeError(
        "Ningun ensayo finalizo correctamente"
    )

best = study.best_trial

# Exportar todos los ensayos
records = []

for t in study.trials:
    row = {
        "numero": t.number,
        "estado": t.state.name,
        "f1_macro": t.value
    }
    row.update(t.params)
    row.update({
        f"extra_{k}": v
        for k, v in t.user_attrs.items()
    })
    records.append(row)

trials_df = pd.DataFrame(records)
trials_df.to_csv(
    METRICS / f"optuna_ensayos_{SCENARIO}.csv",
    index=False
)

# Evaluacion del modelo guardado en validacion
best_model = tf.keras.models.load_model(BEST_MODEL)
proba = best_model.predict(X_val, verbose=0)
pred = np.argmax(proba, axis=1)

accuracy = float(accuracy_score(y_val, pred))
f1_macro = float(
    f1_score(
        y_val, pred,
        average="macro",
        zero_division=0
    )
)

labels = ["Dropout", "Enrolled", "Graduate"]

report = classification_report(
    y_val,
    pred,
    labels=[0, 1, 2],
    target_names=labels,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).T.to_csv(
    METRICS / f"optuna_reporte_{SCENARIO}.csv"
)

# Comparacion con la referencia
baselines = pd.read_csv(
    METRICS / "comparacion_baselines_validacion.csv"
)

baseline = baselines.loc[
    baselines["escenario"] == SCENARIO
].iloc[0]

baseline_f1 = float(baseline["f1_macro_val"])

summary = {
    "scenario": SCENARIO,
    "method": "Optuna TPESampler",
    "optimization_metric": "validation_macro_f1",
    "seed": SEED,
    "trials_completed": len(completed),
    "best_trial": int(best.number),
    "best_parameters": best.params,
    "best_f1_macro": f1_macro,
    "best_accuracy": accuracy,
    "baseline_f1_macro": baseline_f1,
    "delta_f1": f1_macro - baseline_f1,
    "test_used": False
}

with open(
    METRICS / f"optuna_resumen_{SCENARIO}.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary, f, indent=2, ensure_ascii=False
    )

# Evolucion de la optimizacion
ordered = sorted(completed, key=lambda t: t.number)
numbers = [t.number for t in ordered]
values = [t.value for t in ordered]
best_so_far = np.maximum.accumulate(values)

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(numbers, values, label="Ensayos")
ax.plot(
    numbers, best_so_far,
    label="Mejor acumulado"
)
ax.axhline(
    baseline_f1,
    linestyle="--",
    label="Modelo base"
)
ax.set(
    title=f"Optuna - {SCENARIO}",
    xlabel="Numero de ensayo",
    ylabel="Macro-F1 de validacion"
)
ax.legend()
fig.tight_layout()
fig.savefig(
    FIGURES / f"optuna_evolucion_{SCENARIO}.png",
    dpi=180
)
plt.close(fig)

# Matriz de confusion del mejor modelo
fig, ax = plt.subplots(figsize=(7, 6))

ConfusionMatrixDisplay.from_predictions(
    y_val,
    pred,
    labels=[0, 1, 2],
    display_labels=labels,
    cmap="Blues",
    colorbar=False,
    ax=ax
)

ax.set_title(f"Optuna - {SCENARIO}")
fig.tight_layout()
fig.savefig(
    FIGURES / f"optuna_matriz_{SCENARIO}.png",
    dpi=180
)
plt.close(fig)

print("\n========== MEJOR MODELO ==========")
print("Mejor ensayo:", best.number)
print("Macro-F1 base:", round(baseline_f1, 4))
print("Macro-F1 Optuna:", round(f1_macro, 4))
print("Accuracy Optuna:", round(accuracy, 4))
print("Mejora absoluta F1:", round(f1_macro-baseline_f1, 4))

print("\nMejores hiperparametros:")
for name, value in best.params.items():
    print(f"  {name}: {value}")

print("\nArchivos de Optuna guardados correctamente")
print("Datos de prueba NO utilizados")
print("FASE 6 COMPLETADA")

gc.collect()
