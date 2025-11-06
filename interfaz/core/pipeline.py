# interfaz/core/pipeline.py
"""
Pipeline: tomar df_proc resampleado -> ventanas -> extraer features (completo o top8)
-> ajustar columnas al orden esperado por el modelo -> predecir y retornar:

Retorna: (df_feat, indices, y_pred, y_proba)

df_feat: DataFrame con columnas de features (siempre incluirá 'window_center_time')
indices: lista de (start_idx, end_idx)
y_pred: array de predicciones
y_proba: array o None (si model soporta predict_proba)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, List

from interfaz.core.features import sliding_windows, extract_top8_from_signal, extract_features_from_signal, detect_channel_columns
from interfaz.core.top8 import ORANGE_TOP8

DEFAULT_MODEL_PATH = Path("models/k-NN/kNN_k5_full.joblib")
MODEL_FS = 100.0
MODEL_WINDOW_SEC = 2.56
MODEL_OVERLAP = 0.5

def choose_window_params(fs: float):
    # Mantener siempre la ventana de entrenamiento para compatibilidad
    return MODEL_WINDOW_SEC, MODEL_OVERLAP

def run_pipeline_on_processed_df(df_proc: pd.DataFrame,
                                 acc_x_col_name: str = 'Acceleration X(g)',
                                 window_sec: float = None,
                                 overlap: float = None,
                                 model_path: str = None) -> Tuple[pd.DataFrame, List[tuple], np.ndarray, np.ndarray]:
    """
    Ejecuta pipeline sobre df_proc (resampleado a ~100Hz).
    - df_proc debe contener columna 'time' (segundos, empezando en 0) y columnas de señales (Acceleration X(g), etc.)
    - Devuelve: (df_features, indices, y_pred, y_proba)
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}")

    if 'time' not in df_proc.columns:
        raise ValueError("df_proc debe contener columna 'time' en segundos")

    # inferir fs desde time
    time_arr = df_proc['time'].to_numpy(dtype=float)
    dt = np.diff(time_arr)
    dt = dt[~np.isnan(dt) & (dt>0)]
    if len(dt) > 0:
        fs = float(np.round(1.0 / np.median(dt), 6))
    else:
        fs = MODEL_FS

    if window_sec is None or overlap is None:
        wsec, ov = choose_window_params(fs)
    else:
        wsec, ov = window_sec, overlap

    window_samples = int(round(wsec * fs))
    step = int(round(window_samples * (1.0 - ov)))
    if window_samples <= 0 or step <= 0:
        raise ValueError("Parámetros de ventana inválidos")

    # detectar canales disponibles en df_proc
    cols = list(df_proc.columns)
    colmap = detect_channel_columns(cols)
    # asegurarnos al menos acc_x
    if 'acc_x' not in colmap:
        # intentar detectar por nombre exacto 'Acceleration X(g)'
        if acc_x_col_name in df_proc.columns:
            colmap['acc_x'] = acc_x_col_name
        else:
            raise ValueError(f"No se detectó columna de aceleración X. Columnas: {cols}")

    feats_rows = []
    indices = []
    center_times = []

    # generar ventanas por índices (usar sliding_windows)
    for start, end, wdf in sliding_windows(df_proc, window_samples, step):
        # por cada ventana, construir fila de features
        row = {}
        # center time
        tcenter = float(np.nanmean(df_proc['time'].iloc[start:end].to_numpy(dtype=float)))
        # insertar features por canal detectado
        for key, colname in colmap.items():
            # obtener señal numérica (rellenar fwd/bwd)
            sig = pd.to_numeric(wdf[colname], errors='coerce').to_numpy(dtype=float)
            # si ventana corta, rellenar con nan y seguir
            if len(sig) == 0:
                # llenar con NaN features
                # para compatibilidad, dejar llaves pero con NaN
                tmp_top8 = {k: np.nan for k in ['mean','std','var','median','iqr','rms','ptp','sma']}
                for k,v in tmp_top8.items():
                    row[f"{colname}_{k}"] = v
                # y otros features del extractor completo
                for k in ['skew','kurtosis','zcr','dom_freq','spec_entropy','total_power']:
                    row[f"{colname}_{k}"] = np.nan
                continue

            # extraer features completas (tiempo+freq)
            feats = extract_features_from_signal(sig, fs=fs)
            # volcar con prefijo: "<colname>_<feat>"
            for k, v in feats.items():
                row[f"{colname}_{k}"] = v

        # añadir metadatos
        row['window_center_time'] = tcenter
        feats_rows.append(row)
        indices.append((start, end))
        center_times.append(tcenter)

    if len(feats_rows) == 0:
        raise RuntimeError("No se generaron ventanas. Revisa el tamaño del archivo / fs / duración.")

    df_feat = pd.DataFrame(feats_rows)

    # Cargar modelo y preparar X para predicción
    model = joblib.load(model_path)

    # Si el modelo guarda feature_names_in_ -> intentar alinear columnas a eso
    if hasattr(model, 'feature_names_in_'):
        expected = list(model.feature_names_in_)
        missing = [c for c in expected if c not in df_feat.columns]
        if len(missing) > 0:
            # Intentar mapeo heurístico para el caso top8: si modelo espera solo Acc X top8
            # buscar si expected contiene nombres como 'Acceleration X(g)_mean' etc.
            # si se encuentran, pero con diferencias de prefijo, intentar mapear por sufijo
            mapped = {}
            available = list(df_feat.columns)
            for exp in expected:
                if exp in df_feat.columns:
                    mapped[exp] = exp
                    continue
                # heurística: buscar una columna que endswith feat name (por ejemplo '_mean')
                parts = exp.split("_")
                if len(parts) >= 2:
                    suffix = "_".join(parts[1:])  # por ejemplo "X(g)_mean" ó "mean" según convención
                else:
                    suffix = exp
                # Buscar disponible que termine en suffix
                found = None
                for a in available:
                    if a.lower().endswith(suffix.lower()):
                        found = a
                        break
                if found:
                    mapped[exp] = found
            # si pudimos mapear todos los esperados, construir X aligned
            if set(mapped.keys()) == set(expected):
                Xdf = pd.DataFrame()
                for exp in expected:
                    Xdf[exp] = df_feat[mapped[exp]].astype(float)
            else:
                # no pudimos mapear todos: reportar error con detalle
                raise ValueError(
                    "El modelo espera columnas específicas que no se encontraron en las features generadas.\n"
                    f"Columnas esperadas por el modelo (feature_names_in_): {expected}\n"
                    f"Columnas disponibles en df_feat: {list(df_feat.columns)[:40]}...\n"
                    "Si tu modelo fue entrenado con un conjunto distinto de features, proporciona ese modelo o adapta el extractor."
                )
        else:
            Xdf = df_feat[expected].astype(float)
    else:
        # Si el modelo no guarda names, intentar usar columna ORANGE_TOP8 si existe (caso KNN_8)
        # buscar si ORANGE_TOP8 están en df_feat
        if all([c in df_feat.columns for c in ORANGE_TOP8]):
            Xdf = df_feat[ORANGE_TOP8].astype(float)
        else:
            # tomar todas las columnas numéricas excepto 'window_center_time'
            numeric_cols = [c for c in df_feat.columns if c != 'window_center_time']
            Xdf = df_feat[numeric_cols].astype(float)

    # Predicción (usar numpy array para modelos que no aceptan DataFrame)
    try:
        y_pred = model.predict(Xdf)
    except Exception:
        # fallback: pasar numpy array
        y_pred = model.predict(Xdf.values)

    y_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(Xdf)
        except Exception:
            try:
                # intentar acceder al estimador final si es pipeline
                if hasattr(model, 'named_steps'):
                    final = list(model.named_steps.values())[-1]
                    y_proba = final.predict_proba(Xdf)
            except Exception:
                y_proba = None

    # asegurar formato de salida
    return df_feat, indices, np.asarray(y_pred), (np.asarray(y_proba) if y_proba is not None else None)
