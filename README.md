# Reconocimiento de Actividad Humana con Sensores Inerciales mediante Machine Learning
---

## Resumen
Proyecto para clasificación de actividades usando features extraídas de señales IMU. Incluye:
- Preprocesamiento y extracción de features
- Selección automática de features
- Entrenamiento y evaluación de modelos (SVM-RBF y k-NN)
- Comparación entre modelo con todas las features y con 8 features seleccionadas por relevancia (Orange top‑8)
- Guardado de datasets reducidos, reportes, figuras y modelos finales

---

## Contenido (README)
1. [Estructura del repositorio](#estructura-del-repositorio)  
2. [Requisitos e instalación](#requisitos-e-instalación)  
3. [Orden de ejecución de scripts y ejemplos](#orden-de-ejecución-de-scripts-y-ejemplos)  
4. [Descripción detallada de cada script](#descripción-detallada-de-cada-script)  
5. [Archivos de salida: dónde están y qué contienen](#archivos-de-salida-dónde-están-y-qué-contienen)  
6. [Interpretación de resultados (qué significan las métricas)](#interpretación-de-resultados-qué-significan-las-métricas)  
7. [Visualizaciones e imágenes (rutas)](#visualizaciones-e-imágenes-rutas)  
8. [Cómo reproducir todo y buenas prácticas](#cómo-reproducir-todo-y-buenas-prácticas)  
9. [Preguntas frecuentes y solución de errores comunes](#preguntas-frecuentes-y-solución-de-errores-comunes)

---

## Estructura del repositorio
(Estructura esperada)
```
AVD_project/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── final/
│       ├── All_features.csv
│       └── All_features_orange_top8.csv
├── models/                     # Modelos guardados (p. ej. SVM_RBF_full.joblib, KNN_k5_reduced.joblib)
├── reports/
│   ├── selection/
│   │   ├── features_ranking_raw.csv
│   │   └── features_ranking_final.csv
│   ├── train/
│   │   └── (matrices de confusión, métricas por CV)
│   └── final_models/
│       ├── confusion_SVM_full.png
│       ├── confusion_SVM_reduced.png
│       ├── confusion_KNN_full.png
│       ├── confusion_KNN_reduced.png
│       └── final_comparison.csv
├── src/
│   ├── 01_preprocessing.py
│   ├── 02_feature_extraction.py
│   ├── 03_feature_selection.py
│   ├── 04_ML_first_model.py
│   ├── 05_ML_using_top8_orange.py
│   ├── 06_feature_selection.py        # (si existe como helper)
│   └── 07_finalize_models_and_reports.py
├── images/                       # (opcional) imágenes empaquetadas para el README
└── README.md
```

---

## Requisitos e instalación

1. Clonar repositorio:
```powershell
git clone <URL_DEL_REPOSITORIO>
cd AVD_project
```

2. Crear y activar entorno virtual (Windows):
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

3. Instalar dependencias:
```powershell
pip install -r requirements.txt
```
Si no hay `requirements.txt`, instala mínimo:
```powershell
pip install pandas numpy scikit-learn matplotlib seaborn joblib tqdm
```

---

## Orden de ejecución de scripts y ejemplos

Se recomienda ejecutar en este orden:

1. Preprocesamiento (01)
```powershell
python src/01_preprocessing.py --input_raw data/raw --processed_dir data/processed
```

2. Extracción de features (02)
```powershell
python src/02_feature_extraction.py --processed_dir data/processed --features_dir data/features --final_dir data/final
```

3. Selección de features (03)
```powershell
python src/06_feature_selection.py --input_csv "data/final/All_features.csv"
```
Salida principal: `reports/selection/features_ranking_final.csv`

4. Entrenamiento y evaluación inicial (04)
```powershell
python src/05_ML_first_model.py --input_csv "data/final/All_features.csv" --target_col Clase --group_col Sujeto --cv stratified --n_splits 5
```
Genera reportes en `reports/train/` y modelos en `models/` (si la opción está activa).

5. Evaluación con top‑8 (Orange) y guardar CSV reducido (05/06)
```powershell
python src/06_ML_using_top8_orange.py --input_csv "data/final/All_features.csv" --n_splits 5 --cv stratified --final_dir "data/final" --reports_dir "reports" --verbose
```
Salida: `data/final/All_features_orange_top8.csv` y `reports/final_comparison.csv`

6. Modelos finales y reportes detallados (07)
```powershell
python src/07_finalize_models_and_reports.py --input_csv data/final/All_features.csv --reduced_csv data/final/All_features_orange_top8.csv --cv stratified --n_splits 5 --save_models --verbose
```
Salida: matrices de confusión, classification reports, modelos finales en `models/` y reportes en `reports/final_models/`.

---

## Descripción detallada de cada script

- src/01_preprocessing.py  
  Lee raw/, filtra, normaliza tiempos, segmenta ventanas y guarda en processed/.

- src/02_feature_extraction.py  
  Calcula features temporales y espectrales por ventana. Guarda en data/features/ y `data/final/All_features.csv`.

- src/06_feature_selection.py  
  Realiza ranking (varias heurísticas), elimina pares correlacionados y produce el ranking final. Guarda csv con rankings y summary en `reports/selection/`.

- src/05_ML_first_model.py  
  Entrena SVM-RBF y kNN con cross‑validation (stratified o LOSO). Guarda reportes en `reports/train/`.

- src/06_ML_using_top8_orange.py  
  Toma las 8 features definidas (Orange top8), guarda `data/final/All_features_orange_top8.csv`, entrena y compara baseline vs top8. Genera `reports/final_comparison.csv`.

- src/07_finalize_models_and_reports.py  
  Reevaluación final, guarda matrices de confusión PNG + CSV, classification reports, y entrena modelos finales sobre la matriz reducida; guarda modelos en `models/`.

---

## Archivos de salida: dónde están y qué contienen

- data/final/All_features.csv  
  Matriz completa con metadata (Sujeto, Clase, archivo, índices) y ~112 features numéricas.

- data/final/All_features_orange_top8.csv  
  Versión reducida con las 8 features seleccionadas + metadata.

- reports/selection/features_ranking_final.csv  
  Ranking final de features; contiene métricas de ranking y la lista final.

- reports/final_comparison.csv  
  Tabla comparativa (metrics) entre baseline (todas las features) y Orange top‑8.

Resultados:
```
=== COMPARACIÓN (Todas las características vs ORANGE-TOP8) ===
  model  before_acc_mean  before_acc_std  before_f1_mean  after_acc_mean  after_acc_std  after_f1_mean
SVM_RBF         0.984661        0.010454        0.984361        0.960455       0.012722       0.958876
 kNN_k5         0.980527        0.008864        0.980097        0.968122       0.008219       0.966723
```

- reports/final_models/  
  - confusion_SVM_full.png  
  - confusion_SVM_reduced.png  
  - confusion_KNN_full.png  
  - confusion_KNN_reduced.png  
  - final_comparison.csv (detallado)
  - classification_report_SVM_full.csv / _reduced.csv
  - classification_report_KNN_full.csv / _reduced.csv

- models/  
  Modelos finales guardados (ejemplo):
  - models/SVM_RBF_full.joblib
  - models/SVM_RBF_reduced.joblib
  - models/KNN_k5_full.joblib
  - models/KNN_k5_reduced.joblib

Nota: Los modelos nuevos se guardan con sufijos `_full` o `_reduced`. 

---

## Interpretación de resultados (qué significan las métricas)

- Accuracy (precisión): proporción de predicciones correctas. Valor entre 0 y 1.  
- f1 (F1-score): media armónica entre precisión y recall por clase, útil cuando hay desbalance.
- std: desviación estándar entre folds — indica estabilidad del modelo entre particiones.

Observación sobre los resultados:
- La caída de accuracy ~2% al reducir de ~98.4% a ~96.0% indica que las 8 features conservan la mayor parte de la información discriminativa.
- k-NN se mantiene competitivo con SVM en la versión reducida; varianza baja implica resultados estables.
- Las matrices de confusión muestran en qué clases ocurren errores.

---

## Visualizaciones e imágenes (rutas y cómo regenerarlas)

Imágenes generadas por los scripts se guardan en `reports/`:

- Comparación de métricas (tabla): `reports/final_comparison.csv`
- Matrices de confusión:
  - `reports/final_models/confusion_SVM_full.png`
  - `reports/final_models/confusion_SVM_reduced.png`
  - `reports/final_models/confusion_KNN_full.png`
  - `reports/final_models/confusion_KNN_reduced.png`
- Rankings / importancia de features:
  - `reports/selection/features_ranking_final.png` (si el script lo genera)
- Dataset reducido:
  - `data/final/All_features_orange_top8.csv` (CSV)

Si alguna imagen no existe, regenera ejecutando:
```powershell
python src/07_finalize_models_and_reports.py --input_csv data/final/All_features.csv --reduced_csv data/final/All_features_orange_top8.csv --cv stratified --n_splits 5 --save_models --verbose
```
Este script produce las matrices de confusión PNG y los CSVs de reportes.

---

## Cómo cargar y usar los modelos guardados

Ejemplo para cargar un modelo joblib:
```python
import joblib
m = joblib.load("models/SVM_RBF_reduced.joblib")
# X_new es un DataFrame con las mismas columnas que el entrenamiento reducido
y_pred = m.predict(X_new)
```

---

## Buenas prácticas y reproducibilidad

- Mantén versiones de los datos (raw/processed/features/final) para poder reproducir pasos exactos.
- Añade `requirements.txt` con versiones (p. ej. scikit-learn==1.2.2, pandas==2.x).
- Si cambias la semilla aleatoria, documenta el valor (los scripts usan random_state=42 por defecto).
- Guarda logs breves con parámetros de ejecución (CV, n_splits, features usadas) en `reports/`.

---

## Errores comunes y soluciones rápidas

- FileNotFoundError al leer CSV: verifica ruta relativa desde la raíz del proyecto (`data/final/All_features.csv`).
- SyntaxError por caracteres extra: abre el script y elimina caracteres extras (ej. `_` al final de líneas).
- ValueError: Input X contains NaN — solución:
  - Imputar valores (SimpleImputer) en pipeline, o
  - Eliminar filas con NaN antes de entrenar, o
  - Usar estimadores que acepten NaN (HistGradientBoosting).

---  
## Imágenes (incrustadas si están presentes)
Si las imágenes están en `reports/final_models/` y `reports/selection/`, aparecerán así en Markdown:

![Confusión SVM - Full](reports/final_models/confusion_SVM_full.png)
![Confusión SVM - Reduced](reports/final_models/confusion_SVM_reduced.png)

![Confusión KNN - Full](reports/final_models/confusion_KNN_full.png)
![Confusión KNN - Reduced](reports/final_models/confusion_KNN_reduced.png)

![Ranking de features](reports/selection/features_ranking_final.png)

---
