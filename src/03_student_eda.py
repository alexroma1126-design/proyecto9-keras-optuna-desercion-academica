from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd()
FILE = ROOT / "data/raw/student_dropout_success.csv"
FIG = ROOT / "results/figures/students"
REPORT = ROOT / "results/metrics/students"
DOCS = ROOT / "docs"

for folder in (FIG, REPORT, DOCS):
    folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FILE)
target = "Target"
classes = ["Dropout", "Enrolled", "Graduate"]

assert target in df.columns
assert set(df[target].unique()) == set(classes)

features = df.drop(columns=target)

sem1 = [c for c in features if "1st sem" in c]
sem2 = [c for c in features if "2nd sem" in c]
macro = ["Unemployment rate", "Inflation rate", "GDP"]

sin_semestres = [
    c for c in features
    if c not in sem1 + sem2
]

sin_semestres_ni_macro = [
    c for c in sin_semestres if c not in macro
]

print("\n========== AUDITORIA GENERAL ==========")
print("Filas:", len(df))
print("Columnas:", df.shape[1])
print("Variables predictoras:", features.shape[1])
print("Valores faltantes:", int(df.isna().sum().sum()))
print("Duplicados exactos:", int(df.duplicated().sum()))
print("Variables constantes:",
      int((features.nunique() <= 1).sum()))

print("\n========== DISTRIBUCION ==========")
distribution = (
    df[target].value_counts()
    .reindex(classes, fill_value=0)
)
for cls, count in distribution.items():
    print(f"{cls}: {count} ({count/len(df):.2%})")

print("\n========== GRUPOS TEMPORALES ==========")
print("Primer semestre:", len(sem1))
print("Segundo semestre:", len(sem2))
print("Macroeconomicas:", len(macro))
print("Sin notas semestrales:", len(sin_semestres))
print("Sin semestres ni macro:",
      len(sin_semestres_ni_macro))

# Catalogo de caracteristicas
catalog = []
for col in features.columns:
    series = features[col]
    group = (
        "primer_semestre" if col in sem1
        else "segundo_semestre" if col in sem2
        else "macroeconomica" if col in macro
        else "otra_variable"
    )
    catalog.append({
        "variable": col,
        "grupo": group,
        "tipo_python": str(series.dtype),
        "valores_unicos": int(series.nunique()),
        "faltantes": int(series.isna().sum()),
        "ceros": int((series == 0).sum())
            if pd.api.types.is_numeric_dtype(series)
            else None,
        "minimo": float(series.min())
            if pd.api.types.is_numeric_dtype(series)
            else None,
        "maximo": float(series.max())
            if pd.api.types.is_numeric_dtype(series)
            else None
    })

pd.DataFrame(catalog).to_csv(
    REPORT / "catalogo_variables.csv",
    index=False
)

distribution.rename_axis("clase").reset_index(
    name="estudiantes"
).to_csv(
    REPORT / "distribucion_clases.csv",
    index=False
)

sns.set_theme(style="whitegrid")

# Figura 1: clases
plt.figure(figsize=(8, 5))
ax = sns.countplot(
    data=df,
    x=target,
    order=classes,
    hue=target,
    hue_order=classes,
    palette="Set2",
    legend=False
)
for container in ax.containers:
    ax.bar_label(container)
plt.title("Distribucion del resultado academico")
plt.xlabel("Estado academico")
plt.ylabel("Estudiantes")
plt.tight_layout()
plt.savefig(FIG / "01_clases.png", dpi=200)
plt.close()

# Figura 2: edad de ingreso
plt.figure(figsize=(9, 5))
sns.histplot(
    data=df,
    x="Age at enrollment",
    hue=target,
    hue_order=classes,
    bins=30,
    element="step",
    stat="density",
    common_norm=False
)
plt.title("Edad de ingreso segun resultado academico")
plt.xlabel("Edad al matricularse")
plt.tight_layout()
plt.savefig(FIG / "02_edad_ingreso.png", dpi=200)
plt.close()

# Figura 3: nota del primer semestre
plt.figure(figsize=(9, 5))
sns.boxplot(
    data=df,
    x=target,
    y="Curricular units 1st sem (grade)",
    order=classes,
    hue=target,
    palette="Set2",
    legend=False
)
plt.title("Notas del primer semestre por clase")
plt.xlabel("Estado academico")
plt.ylabel("Nota media primer semestre")
plt.tight_layout()
plt.savefig(FIG / "03_notas_sem1.png", dpi=200)
plt.close()

# Figura 4: nota del segundo semestre
plt.figure(figsize=(9, 5))
sns.boxplot(
    data=df,
    x=target,
    y="Curricular units 2nd sem (grade)",
    order=classes,
    hue=target,
    palette="Set2",
    legend=False
)
plt.title("Notas del segundo semestre por clase")
plt.xlabel("Estado academico")
plt.ylabel("Nota media segundo semestre")
plt.tight_layout()
plt.savefig(FIG / "04_notas_sem2.png", dpi=200)
plt.close()

# Metadatos de reproducibilidad
summary = {
    "source": "UCI Machine Learning Repository",
    "dataset_id": 697,
    "rows": len(df),
    "features": features.shape[1],
    "classes": distribution.to_dict(),
    "missing": int(df.isna().sum().sum()),
    "duplicates": int(df.duplicated().sum()),
    "feature_groups": {
        "first_semester": sem1,
        "second_semester": sem2,
        "macroeconomic": macro,
        "without_semester_results": sin_semestres,
        "without_semesters_or_macro":
            sin_semestres_ni_macro
    },
    "methodological_note":
        "Temporal availability must be audited before "
        "claiming prediction at enrollment."
}

with open(
    REPORT / "auditoria.json", "w", encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n========== ARCHIVOS GENERADOS ==========")
for path in sorted(FIG.glob("*.png")):
    print(path.relative_to(ROOT))

print("results/metrics/students/catalogo_variables.csv")
print("results/metrics/students/distribucion_clases.csv")
print("results/metrics/students/auditoria.json")
print("\nFASE 3 - AUDITORIA COMPLETADA")
