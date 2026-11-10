# Prediccion del estado academico mediante redes neuronales y Optuna

**Yachay Tech | Maestria en Ciencia de Datos**  
**Asignatura:** Fundamentos de Inteligencia Artificial

## Descripcion

Proyecto de clasificacion multiclase para predecir tres estados
academicos: desercion (Dropout), matriculado (Enrolled) y
graduado (Graduate).

Se implementan redes neuronales con TensorFlow/Keras y
optimizacion de hiperparametros mediante Optuna.

Adicionalmente, se estudia el efecto de incorporar informacion
academica semestral, el desbalance de clases, la estabilidad
del entrenamiento y la importancia predictiva de las variables.

## 1. Dataset

**Nombre:** Predict Students' Dropout and Academic Success

**Fuente:** UCI Machine Learning Repository

**Identificador:** 697

**Enlace:** https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

| Caracteristica | Valor |
|---|---:|
| Registros | 4424 |
| Variables predictoras | 36 |
| Clases | 3 |
| Valores faltantes | 0 |
| Duplicados exactos | 0 |

Distribucion:

| Clase | Registros |
|---|---:|
| Dropout | 1421 |
| Enrolled | 794 |
| Graduate | 2209 |

Los datos proceden de una institucion de educacion superior de
Portugal. Los resultados no deben generalizarse automaticamente
a otras instituciones o paises.

## 2. Metodologia

Se utilizo una division estratificada y reproducible:

- Entrenamiento: 2654 registros
- Validacion: 885 registros
- Prueba: 885 registros

El preprocesamiento incluye:

- One-Hot Encoding para variables categoricas nominales
- StandardScaler para variables cuantitativas
- Ajuste de transformadores solo con entrenamiento
- Semilla inicial: 42

Los datos de prueba permanecieron reservados hasta cerrar
la seleccion de modelos.

### Escenarios

| Escenario | Variables originales | Entradas codificadas |
|---|---:|---:|
| A - Sin semestres | 21 | 206 |
| B - Primer semestre | 27 | 212 |
| C - Dos semestres | 33 | 218 |
| D - Completo | 36 | 221 |

La disponibilidad temporal de las variables administrativas
debe verificarse antes de interpretar el escenario A como una
prediccion realizada estrictamente al ingreso.

## 3. Modelos

### Modelo base

- Arquitectura: Dense(12) -> Dense(8) -> Dense(3)
- Activacion oculta: ReLU
- Activacion de salida: Softmax
- Funcion de perdida: Sparse Categorical Crossentropy
- Optimizador: Adam
- Early Stopping

### Optimizacion

Herramienta: **Optuna**

Metodo: **TPESampler**

Hiperparametros explorados:

- Numero de capas ocultas: 1 a 3
- Neuronas por capa: 16, 32, 64, 128
- Dropout: 0.0 a 0.4
- Regularizacion L2: 1e-6 a 1e-3
- Learning rate: 1e-4 a 5e-3
- Batch size: 16, 32, 64

Se realizaron 20 ensayos por escenario, sumando 80 ensayos.

La metrica principal fue el macro-F1 de validacion.

## 4. Resultados de validacion

| Escenario | Macro-F1 base | Macro-F1 Optuna |
|---|---:|---:|
| A | 0.4789 | 0.5627 |
| B | 0.6570 | 0.6857 |
| C | 0.6892 | 0.7142 |
| D | 0.7215 | 0.7219 |

Optuna mejoro el macro-F1 de validacion en los cuatro escenarios.
La mayor mejora se observo en A.

## 5. Experimento adicional: balanceo de clases

Sobre el escenario D se compararon tres estrategias:

1. Sin pesos de clase
2. Balanceo moderado
3. Balanceo completo

Cada estrategia se entreno con tres semillas:
42, 123 y 2026.

| Estrategia | Macro-F1 promedio | Recall Enrolled promedio |
|---|---:|---:|
| Sin pesos | 0.7099 | 0.3920 |
| Moderado | 0.7259 | 0.4864 |
| Balanceado | 0.7245 | 0.5765 |

Estos promedios se calcularon sobre una unica particion de
validacion, variando la semilla del entrenamiento.

## 6. Evaluacion final independiente

Se evaluaron diez modelos sobre los 885 registros reservados.

| Escenario | Modelo | Accuracy | Macro-F1 |
|---|---|---:|---:|
| A | Base | 0.6463 | 0.4647 |
| A | Optuna | 0.6362 | 0.5059 |
| B | Base | 0.7333 | 0.6335 |
| B | Optuna | 0.7333 | 0.6486 |
| C | Base | 0.7740 | 0.6968 |
| C | Optuna | 0.7627 | 0.6887 |
| D | Base | 0.7650 | 0.6896 |
| D | Optuna | 0.7740 | 0.7097 |
| D | Moderado | 0.7650 | 0.7107 |
| D | Balanceado | 0.7469 | 0.7064 |

El modelo principal seleccionado previamente fue
**D con balanceo moderado y semilla 42**.

Su macro-F1 en prueba fue 0.7107.

### Evaluacion de la clase Enrolled

| Modelo D | Recall Enrolled |
|---|---:|
| Base | 0.3774 |
| Optuna | 0.4403 |
| Moderado | 0.5220 |
| Balanceado | 0.6164 |

El balanceo moderado mejoro el recall de Enrolled frente
al modelo Optuna, con una reduccion en el recall de otras clases.

### Bootstrap pareado

Comparacion: D Moderado menos D Optuna.

Intervalos de confianza bootstrap del 95 %:

- Diferencia de macro-F1: [-0.0163, 0.0191]
- Diferencia de recall Enrolled: [0.0427, 0.1273]

El intervalo de macro-F1 incluye cero. Por tanto, no existe
evidencia concluyente de mejora global bajo este procedimiento.

## 7. Interpretabilidad

Se utilizo importancia por permutacion sobre la validacion.

Variables destacadas:

1. Curricular units 2nd sem (approved)
2. Curricular units 1st sem (approved)
3. Tuition fees up to date
4. Curricular units 2nd sem (grade)
5. Curricular units 2nd sem (enrolled)
6. Course

La mayor reduccion media del macro-F1 correspondio a
Curricular units 2nd sem (approved): 0.14346.

Estas importancias son predictivas, no causales.
Las variables correlacionadas pueden compartir informacion.

## 8. Tecnologias

- Python
- TensorFlow / Keras
- Optuna
- Pandas y NumPy
- Scikit-learn
- Matplotlib y Seaborn
- SQLite para almacenar ensayos

## 9. Ejecucion

Crear un entorno virtual:

```powershell
python -m venv .venv
```

Instalar las dependencias del entorno registrado:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Los scripts deben ejecutarse desde la raiz del repositorio.

Preparar y explorar los datos:

```powershell
.\.venv\Scripts\python.exe src\00_descargar_datos_uci.py
.\.venv\Scripts\python.exe src\03_student_eda.py
.\.venv\Scripts\python.exe src\04_preprocesamiento.py
```

Entrenar las redes base:

```powershell
.\.venv\Scripts\python.exe src\05_modelos_base.py
```

Optimizar un escenario:

```powershell
.\.venv\Scripts\python.exe src\06_optuna.py --escenario D_completo --ensayos 20
```

Repetir para los escenarios A, B y C.

Ejecutar el experimento de pesos:

```powershell
.\.venv\Scripts\python.exe src\07_balanceo_clases.py
```

Analizar la importancia de variables:

```powershell
.\.venv\Scripts\python.exe src\09_importancia_permutacion.py
```

La evaluacion final se ejecuta una vez cerradas todas
las decisiones de seleccion de modelos:

```powershell
.\.venv\Scripts\python.exe src\08_evaluacion_final.py
```

La reproducibilidad numerica exacta puede variar segun
hardware, bibliotecas y operaciones de TensorFlow.

## 10. Organizacion

- `src/`: scripts de preparacion y experimentacion
- `data/raw/`: datasets originales
- `data/processed/`: particiones transformadas
- `models/`: modelos y preprocesadores
- `results/figures/`: visualizaciones
- `results/metrics/`: resultados cuantitativos
- `docs/`: informe y documentacion
- `notebooks/`: notebooks de referencia

## 11. Limitaciones

- Un unico conjunto de datos institucional.
- Una unica particion estratificada de desarrollo y prueba.
- Desbalance entre las tres clases.
- Posible dependencia temporal de variables administrativas.
- No se han establecido relaciones causales.
- No se ha realizado validacion externa.
- El modelo no debe utilizarse para decisiones individuales
  de alto impacto sin una evaluacion adicional.

## 12. Documentacion adicional

Consultar [Resumen de resultados](docs/resumen_resultados.md).

El proyecto es una investigacion academica y metodologica.
No constituye un sistema de decisiones institucionales.

---

**Yachay Tech - Maestria en Ciencia de Datos**  
**Fundamentos de Inteligencia Artificial - 2026**
