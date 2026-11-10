# Proyecto 9 - Predicción del estado académico con redes neuronales y Optuna

**Autor:** Héctor Alexander Román Potosí  
**Materia:** Fundamentos de Inteligencia Artificial  
**Programa:** Maestría en Ciencia de Datos - Yachay Tech  

---

## 1. Objetivo

Desarrollar un modelo de clasificación multiclase con redes neuronales para predecir el estado académico final de estudiantes universitarios utilizando el dataset **Predict Students' Dropout and Academic Success** del repositorio UCI.

Las clases a predecir fueron:

- **Dropout**
- **Enrolled**
- **Graduate**

---

## 2. Dataset utilizado

- **Fuente:** UCI Machine Learning Repository
- **ID del dataset:** 697
- **Número de registros:** 4424
- **Variables predictoras:** 36
- **Valores faltantes:** 0
- **Duplicados exactos:** 0

Distribución de clases:

- **Graduate:** 2209
- **Dropout:** 1421
- **Enrolled:** 794

Este es un problema de **clasificación multiclase**, no binaria.

---

## 3. Escenarios evaluados

Se construyeron cuatro escenarios de modelado:

- **A_sin_semestres:** 21 variables
- **B_primer_semestre:** 27 variables
- **C_dos_semestres:** 33 variables
- **D_completo:** 36 variables

Los datos fueron divididos de forma estratificada en:

- **Entrenamiento:** 2654
- **Validación:** 885
- **Prueba:** 885

---

## 4. Preprocesamiento

Se aplicó un preprocesamiento reproducible con:

- **One-Hot Encoding** para variables categóricas
- **StandardScaler** para variables numéricas
- Ajuste de transformadores únicamente con el conjunto de entrenamiento
- Semilla fija = 42

---

## 5. Modelo base

Como línea base se utilizó una red neuronal con:

- Capa oculta 1: 12 neuronas, activación ReLU
- Capa oculta 2: 8 neuronas, activación ReLU
- Capa de salida: 3 neuronas, activación Softmax
- Función de pérdida: `sparse_categorical_crossentropy`
- Optimizador: Adam

---

## 6. Optimización con Optuna

Se usó **Optuna con TPESampler** para optimizar:

- Número de capas ocultas
- Número de neuronas por capa
- Dropout
- Regularización L2
- Learning rate
- Batch size

Se realizaron **20 ensayos por escenario**, para un total de **80 ensayos**.

### Comparación Base vs Optuna en validación

| Escenario | Base F1 (%) | Optuna F1 (%) | Mejora (pp) |
|----------|-------------|---------------|-------------|
| A_sin_semestres | 47.89 | 56.27 | 8.377 |
| B_primer_semestre | 65.70 | 68.57 | 2.870 |
| C_dos_semestres | 68.92 | 71.42 | 2.495 |
| D_completo | 72.15 | 72.19 | 0.031 |

---

## 7. Experimento adicional con pesos de clase

Para el escenario **D_completo** se evaluaron tres estrategias:

- **Sin pesos**
- **Moderado**
- **Balanceado**

Se entrenaron **3 semillas por estrategia**: 42, 123 y 2026.

### Promedios en validación

| Estrategia | Macro-F1 | Recall Enrolled | Recall Dropout | Accuracy |
|-----------|----------|-----------------|----------------|----------|
| Sin_pesos | 70.99 % | 39.20 % | 78.99 % | 78.12 % |
| Moderado | 72.59 % | 48.64 % | 78.87 % | 78.19 % |
| Balanceado | 72.45 % | 57.65 % | 77.82 % | 76.80 % |

---

## 8. Evaluación final en test

Se evaluaron 10 modelos sobre el conjunto de prueba reservado.

### Resultados principales

| Escenario | Modelo | Accuracy | Balanced Accuracy | Macro-F1 | Recall Dropout | Recall Enrolled | Recall Graduate |
|----------|--------|----------|-------------------|----------|----------------|-----------------|-----------------|
| A_sin_semestres | Base | 0.6463 | 0.5090 | 0.4647 | 0.6514 | 0.0000 | 0.8756 |
| A_sin_semestres | Optuna | 0.6362 | 0.5178 | 0.5059 | 0.6021 | 0.1006 | 0.8507 |
| B_primer_semestre | Base | 0.7333 | 0.6317 | 0.6335 | 0.7430 | 0.2516 | 0.9005 |
| B_primer_semestre | Optuna | 0.7333 | 0.6421 | 0.6486 | 0.7183 | 0.3145 | 0.8937 |
| C_dos_semestres | Base | 0.7740 | 0.6889 | 0.6968 | 0.7817 | 0.3711 | 0.9140 |
| C_dos_semestres | Optuna | 0.7627 | 0.6804 | 0.6887 | 0.7394 | 0.3899 | 0.9118 |
| D_completo | Base | 0.7650 | 0.6809 | 0.6896 | 0.7535 | 0.3774 | 0.9118 |
| D_completo | Optuna | 0.7740 | 0.6999 | 0.7097 | 0.7500 | 0.4403 | 0.9095 |
| D_completo | Moderado | 0.7650 | 0.7080 | 0.7107 | 0.7218 | 0.5220 | 0.8801 |
| D_completo | Balanceado | 0.7469 | 0.7148 | 0.7064 | 0.7113 | 0.6164 | 0.8167 |

---

## 9. Bootstrap final

Comparación entre **D_completo Moderado** y **D_completo Optuna**:

- **Diferencia macro-F1:** [-0.0163, 0.0191]
- **Diferencia recall Enrolled:** [0.0427, 0.1273]

### Interpretación

No se encontró evidencia concluyente de mejora global en macro-F1, pero sí evidencia de mejora en la identificación de la clase **Enrolled**.

---

## 10. Importancia de variables por permutación

El análisis se realizó con el modelo **D_completo Moderado** sobre el conjunto de validación.

### Top 12 variables más importantes

1. Curricular units 2nd sem (approved)
2. Curricular units 1st sem (approved)
3. Tuition fees up to date
4. Curricular units 2nd sem (grade)
5. Curricular units 2nd sem (enrolled)
6. Course
7. Curricular units 2nd sem (evaluations)
8. Mother's occupation
9. Curricular units 1st sem (enrolled)
10. Curricular units 1st sem (evaluations)
11. Debtor
12. Curricular units 1st sem (grade)

### Interpretación

Las variables más influyentes fueron principalmente académicas, especialmente del segundo semestre, seguidas por variables financieras y contextuales. Esto no implica causalidad, sino importancia predictiva para el modelo.

---

## 11. Respuestas a las preguntas del profesor

### 1. ¿Qué método de búsqueda se utilizó?

Se utilizó **Optuna con TPESampler (Tree-structured Parzen Estimator)** para optimizar hiperparámetros de la red neuronal.

### 2. ¿Qué hiperparámetros se optimizaron?

- Número de capas ocultas
- Número de neuronas por capa
- Dropout
- Regularización L2
- Learning rate
- Batch size

### 3. ¿Cómo mejoraron las métricas respecto al modelo base?

La mejora dependió del escenario. En validación, Optuna mejoró los escenarios A, B, C y D, con la mayor mejora en A. En test, la mejora también dependió del escenario y no fue universal. En D_completo, el experimento con pesos de clase mejoró el recall de la clase Enrolled, aunque no mostró una mejora concluyente en macro-F1 global.

### 4. ¿Qué otras mejoras se intentarían?

- Evaluar redes más profundas con regularización más fina
- Explorar funciones de pérdida con focal loss o costos diferenciados
- Aplicar validación cruzada estratificada
- Probar ensamblado con modelos clásicos
- Comparar con XGBoost, Random Forest y Logistic Regression
- Analizar calibración probabilística
- Incorporar explicabilidad adicional con SHAP

---

## 12. Conclusiones

1. El problema estudiado fue de clasificación multiclase.
2. El escenario completo (**D_completo**) produjo el mejor rendimiento global.
3. La optimización con Optuna mejoró varios escenarios, aunque no siempre garantizó una mejora en prueba.
4. El balanceo moderado permitió mejorar la detección de estudiantes matriculados.
5. El rendimiento académico semestral, especialmente del segundo semestre, fue la fuente de información más influyente para el modelo.
6. Los resultados deben interpretarse como predictivos y no causales.

---

## 13. Limitaciones

- Dataset de una institución específica, por lo que los resultados no se generalizan automáticamente a otros contextos.
- Solo se utilizó una partición fija de entrenamiento, validación y prueba.
- La interpretabilidad por permutación no implica causalidad.
- La clase Enrolled continuó siendo la más difícil de clasificar.

