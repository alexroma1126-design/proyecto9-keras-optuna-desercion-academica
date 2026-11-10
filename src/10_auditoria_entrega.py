from pathlib import Path
import ast
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data/processed/students"
METRICS = ROOT / "results/metrics/students"
MODELS = ROOT / "models/students"
FIGURES = ROOT / "results/figures/students"

SCENARIOS = {
    "A_sin_semestres": 206,
    "B_primer_semestre": 212,
    "C_dos_semestres": 218,
    "D_completo": 221
}

scripts = [
    "00_descargar_datos_uci.py",
    "03_student_eda.py",
    "04_preprocesamiento.py",
    "05_modelos_base.py",
    "06_optuna.py",
    "07_balanceo_clases.py",
    "08_evaluacion_final.py",
    "09_importancia_permutacion.py"
]

errors = []
checks = []

def check(name, condition):
    status = "OK" if condition else "ERROR"
    print(f"[{status}] {name}")
    checks.append({
        "comprobacion": name,
        "correcto": bool(condition)
    })
    if not condition:
        errors.append(name)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

print("\n========== 1. SINTAXIS PYTHON ==========")

for name in scripts:
    path = SRC / name

    if not path.exists():
        check(name, False)
        continue

    try:
        source = path.read_text(encoding="utf-8-sig")
        ast.parse(source, filename=str(path))
        check(name, True)
    except (SyntaxError, UnicodeError) as exc:
        print("Detalle:", exc)
        check(name, False)

print("\n========== 2. DATOS PROCESADOS ==========")

reference_labels = None

for scenario, features in SCENARIOS.items():
    path = DATA / f"{scenario}.npz"

    if not path.exists():
        check(scenario, False)
        continue

    try:
        with np.load(path) as data:
            train = data["X_train"]
            val = data["X_val"]
            test = data["X_test"]

            y_train = data["y_train"]
            y_val = data["y_val"]
            y_test = data["y_test"]

            valid = (
                train.shape == (2654, features)
                and val.shape == (885, features)
                and test.shape == (885, features)
                and len(y_train) == 2654
                and len(y_val) == 885
                and len(y_test) == 885
                and np.isfinite(train).all()
                and np.isfinite(val).all()
                and np.isfinite(test).all()
            )

            labels = (y_train, y_val, y_test)

            if reference_labels is None:
                reference_labels = tuple(
                    arr.copy() for arr in labels
                )
            else:
                valid = valid and all(
                    np.array_equal(a, b)
                    for a, b in zip(labels, reference_labels)
                )

        check(scenario, valid)

    except Exception as exc:
        print("Detalle:", exc)
        check(scenario, False)

print("\n========== 3. OPTUNA ==========")

total_trials = 0

for scenario in SCENARIOS:
    path = METRICS / f"optuna_ensayos_{scenario}.csv"

    if path.exists():
        trials = pd.read_csv(path)
        completed = (
            int((trials["estado"] == "COMPLETE").sum())
            if "estado" in trials.columns else 0
        )
    else:
        completed = 0

    total_trials += completed
    check(
        f"{scenario}: {completed} ensayos",
        completed == 20
    )

check("Total: 80 ensayos completos", total_trials == 80)

print("\n========== 4. MODELOS ==========")

model_files = []

for scenario in SCENARIOS:
    model_files.extend([
        MODELS / f"baseline_{scenario}.keras",
        MODELS / f"optuna_{scenario}.keras",
        MODELS / f"preprocessor_{scenario}.joblib"
    ])

for strategy in ["Sin_pesos", "Moderado", "Balanceado"]:
    for seed in [42, 123, 2026]:
        model_files.append(
            MODELS / "robustez" /
            f"{strategy}_seed_{seed}.keras"
        )

existing = sum(
    p.is_file() and p.stat().st_size > 0
    for p in model_files
)

check(
    f"Modelos y transformadores: {existing}/{len(model_files)}",
    existing == len(model_files)
)

print("\n========== 5. EVALUACION FINAL ==========")

final_path = METRICS / "evaluacion_final_test.csv"

if final_path.exists():
    df = pd.read_csv(final_path)

    required = {
        "escenario", "modelo",
        "accuracy", "macro_f1",
        "recall_dropout",
        "recall_enrolled",
        "recall_graduate"
    }

    valid = (
        len(df) == 10
        and required.issubset(df.columns)
        and df["macro_f1"].between(0, 1).all()
        and df["accuracy"].between(0, 1).all()
    )

    check("Diez evaluaciones finales", valid)
else:
    check("Diez evaluaciones finales", False)

required_results = [
    "protocolo_final.json",
    "bootstrap_final.json",
    "evaluacion_final_por_clase.csv",
    "experimento_pesos_ejecuciones.csv",
    "experimento_pesos_resumen.csv",
    "importancia_permutacion.csv",
    "comparacion_baselines_validacion.csv"
]

for name in required_results:
    check(name, (METRICS / name).is_file())

print("\n========== 6. DOCUMENTACION ==========")

documentation = [
    ROOT / "README.md",
    ROOT / "requirements-lock.txt",
    ROOT / "docs/resumen_resultados.md",
    ROOT / "data/raw/student_dropout_success.csv"
]

for path in documentation:
    check(
        path.name,
        path.is_file() and path.stat().st_size > 0
    )

print("\n========== 7. VISUALIZACIONES ==========")

figures = [
    "09_importancia_permutacion.png",
    "test_comparacion_escenarios.png",
    "test_matriz_D_Moderado.png",
    "test_matriz_D_Optuna.png"
]

for name in figures:
    path = FIGURES / name
    check(name, path.is_file() and path.stat().st_size > 0)

print("\n========== 8. HUELLAS SHA-256 ==========")

hashes = {}

for path in [
    ROOT / "README.md",
    ROOT / "docs/resumen_resultados.md",
    ROOT / "data/raw/student_dropout_success.csv",
    ROOT / "results/metrics/students/evaluacion_final_test.csv"
]:
    if path.exists():
        hashes[str(path.relative_to(ROOT))] = sha256(path)
        print(path.name, hashes[str(path.relative_to(ROOT))])

report = {
    "proyecto": "Proyecto 9 - IA",
    "comprobaciones": checks,
    "total_comprobaciones": len(checks),
    "correctas": sum(c["correcto"] for c in checks),
    "errores": errors,
    "sha256": hashes
}

output = METRICS / "auditoria_entrega.json"

output.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\n========== RESUMEN AUDITORIA ==========")
print("Comprobaciones:", len(checks))
print("Correctas:", report["correctas"])
print("Errores:", len(errors))
print("Reporte:", output)

if errors:
    print("\nREQUIERE CORRECCIONES")
    for item in errors:
        print(" -", item)
    sys.exit(1)

print("\nAUDITORIA COMPLETADA SIN ERRORES")
