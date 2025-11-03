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
1. [Descripción](#Descripción)
2. [Estructura del Proyecto](#Organización-del-repositorio-y-descripción-de-las-carpetas)
3. [Instalación](#Instalación)
4. [Ejecución de Scripts](#Ejecución-de-scripts)
5. [Resultados](#Resultados)
6. [Interpretación](#Interpretación)
7. [Notas](#Notas)

##  Descripción
Este proyecto implementa un sistema de reconocimiento de actividad humana (HAR) basado en datos provenientes de sensores inerciales (Inertial Measurement Units, IMU). El objetivo es clasificar actividades como caminar, correr y permanecer quieto mediante el análisis de características estadísticas extraídas de señales de aceleración y velocidad angular.

El flujo completo del sistema abarca desde el preprocesamiento y segmentación de datos crudos, hasta la extracción y selección automática de características, seguido del entrenamiento y evaluación de modelos de aprendizaje automático (SVM-RBF y k-NN).

Asimismo, se incluye una comparación de desempeño entre modelos entrenados con todas las características y aquellos optimizados con las 8 variables más relevantes según el método de selección Orange (top-8 features), demostrando que la reducción de dimensionalidad mantiene una alta precisión con menor complejidad computacional.

## Organización del repositorio y descripción de las carpetas

- `/data/`: Contiene los conjuntos de datos generados en las diferentes etapas del flujo de trabajo.
    -`/final/`: Archivos CSV utilizados para el entrenamiento y evaluación de los modelos.
  
        - `All_features.csv`: Dataset completo con todas las características extraídas
        
        - `All_features_orange_top8.csv`: Dataset reducido con las 8 características seleccionadas por relevancia.

- `/models/`: Modelos de aprendizaje automático entrenados.
    - `SVM_todas_caracteristicas.joblib`: Modelo SVM con todas las características.
    - `kNN_8_caracteristicas.joblib`: Modelo k-NN entrenado con las 8 características seleccionadas.

- `/reports`: Contiene los reportes visuales y métricas finales de desempeño.
    - `/final_models/`: Incluye comparaciones de precisión, matrices de confusión y resultados de ambos modelos.

- `/src/`: Scripts principales que implementan el flujo completo del sistema: desde la adquisición y preprocesamiento de datos, hasta la generación de reportes y modelos finales.
    - `01_preprocessing.py`: Limpieza y segmentación de señales IMU.

    - `02_feature_extraction.py`: Extracción de características estadísticas.

    - `03_feature_selection.py`: Selección automática de características mediante Orange.

    - `04_ML_first_model.py`: Entrenamiento inicial con todas las características.

    - `05_ML_using_top8_orange.py`: Entrenamiento con el conjunto reducido (8 features).

    - `07_finalize_models_and_reports.py`: Cálculo de métricas, generación de reportes y guardado de modelos finale
 
- `/Caracteristicas/`
    -  `Mejores_Caracteristicas`: Pdf extraido de Orange Data Mining de todas las caracteristicas ordenadas dependiendo su rendimiento de clasificación en base a varios métodos de puntuación.

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

## Resultados

### 8 Características Seleccionadas
1. Acceleration X(g)_mean
2. Acceleration X(g)_std
3. Acceleration X(g)_var
4. Acceleration X(g)_median
5. Acceleration X(g)_iqr
6. Acceleration X(g)_rms
7. Acceleration X(g)_ptp
8. Acceleration X(g)_sma

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

### Ejemplo:
![Ejemplo: Métricas para SVM con 8 características](reports/final_models/M%C3%A9tricas%20para%20SVM%20con%208%20caracter%C3%ADsticas.png)


##  Interpretación
La reducción a ocho características mantiene un rendimiento cercano al 96 %, lo que evidencia que la selección automática de features puede preservar la capacidad predictiva con un modelo más ligero.
El modelo k-NN muestra menor varianza entre iteraciones al usar el conjunto reducido, mientras que SVM-RBF conserva la mayor precisión general.
Las matrices de confusión permiten identificar patrones específicos de error por clase, mostrando una alta consistencia en la clasificación de caminar y correr, con ligeras confusiones en las transiciones hacia el estado de quieto.

##  Notas
- Los reportes completos (CSV y figuras) se encuentran en `reports/final_models/`.
