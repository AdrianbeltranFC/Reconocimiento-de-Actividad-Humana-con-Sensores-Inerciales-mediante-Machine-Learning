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
##  Tabla de Contenidos
1. [Descripción](#descripción)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Instalación](#instalación)
4. [Ejecución de Scripts](#ejecución-de-scripts)
5. [Resultados](#resultados)

##  Descripción
Sistema de clasificación de actividades usando datos de sensores IMU. Implementa modelos SVM-RBF y k-NN, con selección de características para reducir dimensionalidad manteniendo alto rendimiento.

## 📁 Estructura del Proyecto
```
AVD_project/
├── data/
│   └── final/
│       ├── All_features.csv              # Dataset completo
│       └── All_features_orange_top8.csv  # Dataset reducido (8 features)
├── models/
│   ├── SVM_todas_caracteristicas.joblib
│   └── kNN_8_caracteristicas.joblib
├── reports/
│   └── final_models/
│       ├── Métricas para SVM con todas las características.png
│       ├── Métricas para SVM con 8 características.png
│       ├── Métricas para k-NN con todas las características.png
│       ├── Métricas para k-NN con 8 características.png
│       └── Comparación_baseline_vs_8_caracteristicas.csv
└── src/
    ├── 01_preprocessing.py
    ├── 02_feature_extraction.py
    ├── 03_feature_selection.py
    ├── 04_ML_first_model.py
    ├── 05_ML_using_top8_orange.py
    ├── 06_feature_selection.py
    └── 07_finalize_models_and_reports.py
```

##  Instalación

1. **Clonar el repositorio**
```powershell
git clone <URL_DEL_REPOSITORIO>
cd AVD_project
```

2. **Crear entorno virtual**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

3. **Instalar dependencias**
```powershell
pip install pandas numpy scikit-learn matplotlib seaborn joblib tqdm
```

##  Ejecución de Scripts

### 1. Preprocesamiento y Extracción (opcional si ya tienes los CSVs)
```powershell
python src/01_preprocessing.py --input_raw data/raw --processed_dir data/processed
python src/02_feature_extraction.py --processed_dir data/processed --features_dir data/features
```

### 2. Generar Modelos y Reportes Finales
```powershell
python src/07_finalize_models_and_reports.py --input_csv data/final/All_features.csv --reduced_csv data/final/All_features_orange_top8.csv --n_splits 5 --save_models --verbose
```

## 📊 Resultados

### Comparación de Accuracy entre modelos 
| Modelo  | Con todas las features | Con 8 features |
|---------|----------------------|----------------|
| SVM-RBF | 98.47% ±1.05%       | 96.05% ±1.27% |
| k-NN    | 98.05% ±0.89%       | 96.81% ±0.82% |

### Matrices de Confusión
Las matrices de confusión se encuentran en:
- `reports/final_models/Métricas para SVM con todas las características.png`
- `reports/final_models/Métricas para SVM con 8 características.png`
- `reports/final_models/Métricas para k-NN con todas las características.png`
- `reports/final_models/Métricas para k-NN con 8 características.png`

### 8 Características Seleccionadas
1. Acceleration X(g)_mean
2. Acceleration X(g)_std
3. Acceleration X(g)_var
4. Acceleration X(g)_median
5. Acceleration X(g)_iqr
6. Acceleration X(g)_rms
7. Acceleration X(g)_ptp
8. Acceleration X(g)_sma

##  Interpretación
- La reducción a 8 características mantiene un rendimiento muy parecido al de considerar todas.  (~96% acc)
- k-NN muestra menor varianza en sus predicciones con features reducidas
- Las matrices de confusión muestran patrones de error específicos por actividad

##  Notas
- Las métricas completas están disponibles en los CSVs de reporte (reports)
