import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/students"
MODELS = ROOT / "models/students"
METRICS = ROOT / "results/metrics/students"
FIGURES = ROOT / "results/figures/students"

for folder in (MODELS, METRICS, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    "A_sin_semestres",
    "B_primer_semestre",
    "C_dos_semestres",
    "D_completo"
]

CLASSES = ["Dropout", "Enrolled", "Graduate"]
SEED = 42
rows = []

for scenario in SCENARIOS:

    print(f"\n========== {scenario} ==========")

    path = DATA / f"{scenario}.npz"

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path) as data:
        X_train = data["X_train"].astype("float32")
        y_train = data["y_train"].astype("int32")
        X_val = data["X_val"].astype("float32")
        y_val = data["y_val"].astype("int32")

    assert X_train.shape[0] == len(y_train)
    assert X_val.shape[0] == len(y_val)
    assert X_train.shape[1] == X_val.shape[1]
    assert set(np.unique(y_train)) == {0, 1, 2}
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_val).all()

    # Modelo de referencia: clase mayoritaria
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_val)

    # Reiniciar estado y semillas entre experimentos
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    # Arquitectura fija de Keras
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ], name=f"baseline_{scenario}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=12,
        restore_best_weights=True
    )

    # Entrenamiento sin utilizar los datos de prueba
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        shuffle=True,
        verbose=0,
        callbacks=[stopping]
    )

    probabilities = model.predict(X_val, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    report = classification_report(
        y_val,
        predictions,
        labels=[0, 1, 2],
        target_names=CLASSES,
        zero_division=0,
        output_dict=True
    )

    scores = {
        "escenario": scenario,
        "entradas_codificadas": int(X_train.shape[1]),
        "parametros": int(model.count_params()),
        "epocas_ejecutadas": int(
            len(history.history["loss"])
        ),
        "mejor_epoca_val_loss": int(
            np.argmin(history.history["val_loss"]) + 1
        ),
        "accuracy_dummy": float(
            accuracy_score(y_val, dummy_pred)
        ),
        "f1_macro_dummy": float(
            f1_score(
                y_val, dummy_pred,
                average="macro", zero_division=0
            )
        ),
        "accuracy_val": float(
            accuracy_score(y_val, predictions)
        ),
        "balanced_accuracy_val": float(
            balanced_accuracy_score(y_val, predictions)
        ),
        "f1_macro_val": float(
            f1_score(
                y_val, predictions,
                average="macro", zero_division=0
            )
        ),
        "f1_ponderado_val": float(
            f1_score(
                y_val, predictions,
                average="weighted", zero_division=0
            )
        ),
        "log_loss_val": float(
            log_loss(
                y_val, probabilities, labels=[0, 1, 2]
            )
        ),
        "recall_dropout": float(
            report["Dropout"]["recall"]
        ),
        "recall_enrolled": float(
            report["Enrolled"]["recall"]
        ),
        "recall_graduate": float(
            report["Graduate"]["recall"]
        )
    }

    rows.append(scores)

    # Guardar modelo y resultados
    model.save(
        MODELS / f"baseline_{scenario}.keras"
    )

    hist = pd.DataFrame(history.history)
    hist.insert(
        0, "epoca", np.arange(1, len(hist) + 1)
    )

    hist.to_csv(
        METRICS / f"historia_{scenario}.csv",
        index=False
    )

    pd.DataFrame(report).T.to_csv(
        METRICS / f"reporte_{scenario}.csv"
    )

    # Curvas de aprendizaje
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(
        hist["epoca"], hist["loss"], label="Train"
    )
    axes[0].plot(
        hist["epoca"], hist["val_loss"],
        label="Validacion"
    )
    axes[0].set(
        title="Perdida", xlabel="Epoca", ylabel="Loss"
    )
    axes[0].legend()

    axes[1].plot(
        hist["epoca"], hist["accuracy"],
        label="Train"
    )
    axes[1].plot(
        hist["epoca"], hist["val_accuracy"],
        label="Validacion"
    )
    axes[1].set(
        title="Accuracy",
        xlabel="Epoca",
        ylabel="Accuracy"
    )
    axes[1].legend()

    fig.suptitle(f"Aprendizaje - {scenario}")
    fig.tight_layout()
    fig.savefig(
        FIGURES / f"curvas_{scenario}.png",
        dpi=170
    )
    plt.close(fig)

    # Matriz de confusion
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_val,
        predictions,
        labels=[0, 1, 2],
        display_labels=CLASSES,
        cmap="Blues",
        colorbar=False,
        values_format="d",
        ax=ax
    )

    ax.set_title(
        f"Matriz de confusion (validacion) - {scenario}"
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES / f"matriz_{scenario}.png",
        dpi=170
    )
    plt.close(fig)

    print(
        f"Accuracy dummy: {scores['accuracy_dummy']:.4f}"
    )
    print(
        f"Accuracy Keras: {scores['accuracy_val']:.4f}"
    )
    print(
        f"Macro-F1 Keras: {scores['f1_macro_val']:.4f}"
    )
    print(
        f"Recall dropout: {scores['recall_dropout']:.4f}"
    )
    print(
        f"Epocas: {scores['epocas_ejecutadas']}"
    )

# Consolidar comparacion
results = pd.DataFrame(rows)

results.to_csv(
    METRICS / "comparacion_baselines_validacion.csv",
    index=False
)

# Configuracion reproducible
with (
    METRICS / "config_baseline.json"
).open("w", encoding="utf-8") as file:

    json.dump({
        "seed": SEED,
        "tensorflow": tf.__version__,
        "architecture": [12, 8, 3],
        "activation_hidden": "relu",
        "activation_output": "softmax",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "loss": "sparse_categorical_crossentropy",
        "batch_size": 32,
        "epochs_max": 100,
        "early_stopping_patience": 12,
        "monitor": "val_loss",
        "evaluation_split": "validation",
        "test_not_used": True
    }, file, indent=2, ensure_ascii=False)

# Grafico comparativo final
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
x = np.arange(len(results))

axes[0].bar(
    x - 0.18, results["accuracy_dummy"],
    width=0.35, label="Dummy"
)
axes[0].bar(
    x + 0.18, results["accuracy_val"],
    width=0.35, label="Keras"
)
axes[0].set(
    title="Accuracy de validacion",
    ylabel="Accuracy",
    xticks=x,
    xticklabels=["A", "B", "C", "D"],
    ylim=(0, 1)
)
axes[0].legend()

axes[1].bar(
    x - 0.18, results["f1_macro_dummy"],
    width=0.35, label="Dummy"
)
axes[1].bar(
    x + 0.18, results["f1_macro_val"],
    width=0.35, label="Keras"
)
axes[1].set(
    title="F1 macro de validacion",
    ylabel="F1 macro",
    xticks=x,
    xticklabels=["A", "B", "C", "D"],
    ylim=(0, 1)
)
axes[1].legend()

fig.tight_layout()
fig.savefig(
    FIGURES / "comparacion_baselines_validacion.png",
    dpi=170
)
plt.close(fig)

print("\n========== RESUMEN FINAL ==========")
print(
    results[
        [
            "escenario",
            "accuracy_dummy",
            "accuracy_val",
            "f1_macro_val",
            "recall_dropout"
        ]
    ].to_string(index=False)
)

print(
    "\nFASE 5 COMPLETADA"
    " - CONJUNTO DE PRUEBA NO UTILIZADO"
)
