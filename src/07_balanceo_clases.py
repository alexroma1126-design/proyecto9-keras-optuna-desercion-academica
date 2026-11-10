import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/students"
METRICS = ROOT / "results/metrics/students"
FIGURES = ROOT / "results/figures/students"
MODELS = ROOT / "models/students/robustez"

MODELS.mkdir(parents=True, exist_ok=True)

SCENARIO = "D_completo"
SEEDS = [42, 123, 2026]
CLASSES = ["Dropout", "Enrolled", "Graduate"]

with np.load(DATA / f"{SCENARIO}.npz") as data:
    X_train = data["X_train"].astype("float32")
    y_train = data["y_train"].astype("int32")
    X_val = data["X_val"].astype("float32")
    y_val = data["y_val"].astype("int32")

config = json.loads(
    (METRICS / f"optuna_resumen_{SCENARIO}.json")
    .read_text(encoding="utf-8")
)

params = config["best_parameters"]

counts = np.bincount(y_train, minlength=3)
balanced = len(y_train) / (3 * counts)

strategies = {
    "Sin_pesos": np.ones(3),
    "Moderado": np.sqrt(balanced),
    "Balanceado": balanced.copy()
}

def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(X_train.shape[1],)))

    for i in range(int(params["n_layers"])):
        model.add(tf.keras.layers.Dense(
            int(params[f"units_{i}"]),
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(
                float(params["l2"])
            )
        ))

        if float(params["dropout"]) > 0:
            model.add(tf.keras.layers.Dropout(
                float(params["dropout"])
            ))

    model.add(
        tf.keras.layers.Dense(3, activation="softmax")
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(params["learning_rate"])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

results = []

print("\n========== EXPERIMENTO DE BALANCEO ==========")
print("Escenario:", SCENARIO)
print("Distribucion entrenamiento:", counts.tolist())
print("Semillas:", SEEDS)

for strategy, weights in strategies.items():

    # Normalizar para mantener peso medio cercano a uno
    weights = weights / np.average(
        weights, weights=counts
    )

    class_weights = {
        i: float(weights[i]) for i in range(3)
    }

    print(f"\nEstrategia: {strategy}")
    print("Pesos:", class_weights)

    for seed in SEEDS:

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)

        model = create_model()

        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=80,
            batch_size=int(params["batch_size"]),
            class_weight=class_weights,
            callbacks=[callback],
            verbose=0,
            shuffle=True
        )

        probabilities = model.predict(
            X_val, verbose=0
        )

        predictions = np.argmax(
            probabilities, axis=1
        )

        recalls = recall_score(
            y_val,
            predictions,
            labels=[0, 1, 2],
            average=None,
            zero_division=0
        )

        f1_classes = f1_score(
            y_val,
            predictions,
            labels=[0, 1, 2],
            average=None,
            zero_division=0
        )

        row = {
            "estrategia": strategy,
            "semilla": seed,
            "accuracy": accuracy_score(
                y_val, predictions
            ),
            "f1_macro": f1_score(
                y_val, predictions,
                average="macro",
                zero_division=0
            ),
            "recall_dropout": recalls[0],
            "recall_enrolled": recalls[1],
            "recall_graduate": recalls[2],
            "f1_enrolled": f1_classes[1],
            "epocas": len(history.history["loss"])
        }

        results.append(row)

        model.save(
            MODELS / f"{strategy}_seed_{seed}.keras"
        )

        # Matriz de confusion representativa por semilla
        if seed == 42:
            fig, ax = plt.subplots(figsize=(7, 6))

            ConfusionMatrixDisplay.from_predictions(
                y_val,
                predictions,
                labels=[0, 1, 2],
                display_labels=CLASSES,
                cmap="Blues",
                colorbar=False,
                ax=ax
            )

            ax.set_title(
                f"{strategy} - validacion - seed 42"
            )

            fig.tight_layout()
            fig.savefig(
                FIGURES / f"pesos_matriz_{strategy}.png",
                dpi=180
            )
            plt.close(fig)

        print(
            f"Semilla={seed} | "
            f"F1 macro={row['f1_macro']:.4f} | "
            f"Recall Enrolled={row['recall_enrolled']:.4f} | "
            f"Recall Dropout={row['recall_dropout']:.4f}"
        )

df = pd.DataFrame(results)

df.to_csv(
    METRICS / "experimento_pesos_ejecuciones.csv",
    index=False
)

summary = df.groupby(
    "estrategia", sort=False
).agg(
    f1_macro_promedio=("f1_macro", "mean"),
    f1_macro_desv=("f1_macro", "std"),
    recall_enrolled_promedio=("recall_enrolled", "mean"),
    recall_enrolled_desv=("recall_enrolled", "std"),
    recall_dropout_promedio=("recall_dropout", "mean"),
    recall_dropout_desv=("recall_dropout", "std"),
    accuracy_promedio=("accuracy", "mean")
).reset_index()

summary.to_csv(
    METRICS / "experimento_pesos_resumen.csv",
    index=False
)

print("\n========== RESULTADOS PROMEDIO ==========")
print(summary.round(4).to_string(index=False))

print("\n========== MODELO OPTUNA ORIGINAL ==========")

reference = pd.read_csv(
    METRICS / f"optuna_reporte_{SCENARIO}.csv",
    index_col=0
)

print("Macro-F1 original:", round(
    float(config["best_f1_macro"]), 4
))
print("Recall Enrolled original:", round(
    float(reference.loc["Enrolled", "recall"]), 4
))
print("Recall Dropout original:", round(
    float(reference.loc["Dropout", "recall"]), 4
))

print("\nFASE 7 COMPLETADA")
print("Datos de prueba NO utilizados")
