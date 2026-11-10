from pathlib import Path
from textwrap import dedent
import json
import re

ROOT = Path.cwd()
DOC = ROOT / "docs/resumen_resultados.md"
OUTPUT = ROOT / "notebooks/Proyecto_9_Redes_Neuronales_Optuna_Final.ipynb"

required = [
    DOC,
    ROOT / "data/raw/student_dropout_success.csv",
    ROOT / "results/metrics/students/preprocesamiento.json",
    ROOT / "results/metrics/students/evaluacion_final_test.csv",
    ROOT / "results/metrics/students/experimento_pesos_resumen.csv",
    ROOT / "results/metrics/students/importancia_permutacion.csv"
]

missing = [str(p) for p in required if not p.is_file()]

if missing:
    raise FileNotFoundError(
        "Archivos faltantes:\n" + "\n".join(missing)
    )

cells = []

def add_markdown(content):
    cells.append({
        "cell_type": "markdown",
        "id": f"md-{len(cells):04d}",
        "metadata": {},
        "source": dedent(content).strip() + "\n"
    })

def add_code(content):
    cells.append({
        "cell_type": "code",
        "id": f"code-{len(cells):04d}",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": dedent(content).strip() + "\n"
    })

add_markdown("""
> **Notebook académico final — Fundamentos de Inteligencia Artificial**
>
> Este cuaderno presenta los resultados del Proyecto 9.
> Utiliza experimentos ya ejecutados y documentados.
>
> No vuelve a entrenar redes neuronales ni utiliza los
> datos de prueba para seleccionar hiperparámetros.
> Los scripts experimentales están disponibles en `src/`.
""")

add_code("""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

current = Path.cwd().resolve()

ROOT = next(
    (
        path for path in (current, *current.parents)
        if (
            path / "results/metrics/students/evaluacion_final_test.csv"
        ).exists()
    ),
    None
)

if ROOT is None:
    raise FileNotFoundError(
        "Abra el notebook dentro del proyecto completo."
    )

METRICS = ROOT / "results/metrics/students"
FIGURES = ROOT / "results/figures/students"

print("Proyecto:", ROOT)
print("Resultados encontrados correctamente.")
""")

# Recuperar las secciones del informe ya elaborado
document = DOC.read_text(encoding="utf-8-sig")

sections = re.split(
    r"(?=^##\s+)",
    document,
    flags=re.MULTILINE
)

for section in sections:

    if not section.strip():
        continue

    add_markdown(section)

    title = section.strip().splitlines()[0]

    if title.startswith("## 2."):
        add_code("""
df = pd.read_csv(
    ROOT / "data/raw/student_dropout_success.csv"
)

print("Dimensiones:", df.shape)
print("Valores faltantes:", int(df.isna().sum().sum()))

distribution = df["Target"].value_counts()
display(distribution.to_frame("estudiantes"))

print("\\nPrimeras cinco observaciones:")
display(df.head())
""")

        add_markdown("""
![Distribución de clases](../results/figures/students/01_clases.png)
""")

    elif title.startswith("## 3."):
        add_code("""
info = json.loads(
    (METRICS / "preprocesamiento.json")
    .read_text(encoding="utf-8")
)

print("Distribución:")
display(pd.Series(info["split"]).to_frame("registros"))

scenarios = pd.DataFrame.from_dict(
    info["scenarios"],
    orient="index"
)

display(
    scenarios[["original_features", "encoded_features"]]
)
""")

    elif title.startswith("## 6."):
        add_code("""
base = pd.read_csv(
    METRICS / "comparacion_baselines_validacion.csv"
)

rows = []

for scenario in [
    "A_sin_semestres",
    "B_primer_semestre",
    "C_dos_semestres",
    "D_completo"
]:
    result = json.loads(
        (METRICS / f"optuna_resumen_{scenario}.json")
        .read_text(encoding="utf-8")
    )

    original = base.loc[
        base["escenario"] == scenario
    ].iloc[0]

    rows.append({
        "Escenario": scenario,
        "F1 base": original["f1_macro_val"],
        "F1 Optuna": result["best_f1_macro"],
        "Mejora (pp)": 100 * (
            result["best_f1_macro"] -
            original["f1_macro_val"]
        )
    })

validation_table = pd.DataFrame(rows)
display(validation_table.round(4))
""")

        add_markdown("""
![Comparación de modelos base](../results/figures/students/comparacion_baselines_validacion.png)
""")

    elif title.startswith("## 7."):
        add_code("""
weights = pd.read_csv(
    METRICS / "experimento_pesos_resumen.csv"
)

display(weights.round(4))
""")

        add_markdown("""
![Balanceo moderado](../results/figures/students/pesos_matriz_Moderado.png)
""")

    elif title.startswith("## 8."):
        add_code("""
final = pd.read_csv(
    METRICS / "evaluacion_final_test.csv"
)

display(
    final[
        [
            "escenario",
            "modelo",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "recall_enrolled"
        ]
    ].round(4)
)
""")

        add_markdown("""
![Evaluación final](../results/figures/students/test_comparacion_escenarios.png)

![Matriz de confusión del modelo principal](../results/figures/students/test_matriz_D_Moderado.png)
""")

    elif title.startswith("## 9."):
        add_code("""
bootstrap = json.loads(
    (METRICS / "bootstrap_final.json")
    .read_text(encoding="utf-8")
)

print("Comparación:", bootstrap["comparison"])

print(
    "IC 95% diferencia macro-F1:",
    bootstrap["macro_f1_delta_ci"]
)

print(
    "IC 95% diferencia recall Enrolled:",
    bootstrap["recall_enrolled_delta_ci"]
)
""")

    elif title.startswith("## 10."):
        add_code("""
importance = pd.read_csv(
    METRICS / "importancia_permutacion.csv"
)

display(
    importance[
        [
            "variable",
            "delta_f1_media",
            "delta_recall_enrolled_media"
        ]
    ].head(12).round(5)
)
""")

        add_markdown("""
![Importancia por permutación](../results/figures/students/09_importancia_permutacion.png)
""")

add_markdown("""
---

## Reproducibilidad y alcance

Las celdas ejecutables de este notebook reconstruyen las
tablas a partir de los archivos de resultados guardados.

Para repetir el entrenamiento y la optimización desde cero,
se deben ejecutar los scripts de `src/` siguiendo el README
y en un entorno compatible.

Los resultados de prueba corresponden a la evaluación
final ya cerrada; no deben utilizarse para escoger nuevos
hiperparámetros.

**Fuente de datos:** [UCI Machine Learning Repository,
dataset 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

OUTPUT.write_text(
    json.dumps(
        notebook,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

# Verificar estructura del archivo generado
loaded = json.loads(OUTPUT.read_text(encoding="utf-8"))

assert loaded["nbformat"] == 4
assert len(loaded["cells"]) == len(cells)

for cell in loaded["cells"]:
    if cell["cell_type"] == "code":
        compile(cell["source"], "<notebook>", "exec")

print("\n========== NOTEBOOK FINAL ==========")
print("Archivo:", OUTPUT)
print("Total de celdas:", len(cells))

print(
    "Celdas de Markdown:",
    sum(c["cell_type"] == "markdown" for c in cells)
)

print(
    "Celdas de codigo:",
    sum(c["cell_type"] == "code" for c in cells)
)

print("Sintaxis de celdas: CORRECTA")
print("FASE 13 COMPLETADA")
