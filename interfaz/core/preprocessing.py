"""
interfaz/core/preprocessing.py
Preprocesado ligero y utilidades para la GUI
Incluye filtros Hampel y Butterworth solicitados.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal

# Mapas para normalizar nombres de columnas a nombres canónicos
COLUMN_CANDIDATES = {
    'Acceleration X(g)': ['ax', 'accx', 'acc_x', 'acceleration x(g)', 'acceleration_x', 'acceleration_x_g'],
    'Acceleration Y(g)': ['ay', 'accy', 'acc_y', 'acceleration y(g)', 'acceleration_y'],
    'Acceleration Z(g)': ['az', 'accz', 'acc_z', 'acceleration z(g)', 'acceleration_z'],
    'Gyroscope X(deg/s)': ['gx', 'gyrx', 'gyro_x', 'gyroscope x', 'gyro_x_deg', 'gyr_x'],
    'Gyroscope Y(deg/s)': ['gy', 'gyry', 'gyro_y', 'gyroscope y', 'gyro_y_deg', 'gyr_y'],
    'Gyroscope Z(deg/s)': ['gz', 'gyrz', 'gyro_z', 'gyroscope z', 'gyro_z_deg', 'gyr_z'],
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mapea columnas del DataFrame original a nombres canónicos.
    Devuelve un DataFrame nuevo con columnas normalizadas y (si existe) 'time'.
    """
    cols = {c.lower(): c for c in df.columns}
    out = pd.DataFrame()
    for canonical, candidates in COLUMN_CANDIDATES.items():
        found = None
        for cand in candidates:
            if cand.lower() in cols:
                found = cols[cand.lower()]
                break
        if found is not None:
            out[canonical] = pd.to_numeric(df[found], errors='coerce')
    # detectar columna de tiempo si existe
    for tname in ['time', 'timestamp', 'ts']:
        if tname in df.columns:
            out['time'] = pd.to_numeric(df[tname], errors='coerce')
            break
    return out

def estimate_fs_from_time(time_arr: np.ndarray) -> float:
    """
    Estima la frecuencia de muestreo en Hz a partir de un array de tiempos en segundos.
    Usa la mediana de 1/dt para robustez.
    """
    dt = np.diff(time_arr)
    dt = dt[~np.isnan(dt) & (dt > 0)]
    if len(dt) == 0:
        return None
    return float(np.round(1.0 / np.median(dt), 6))

def hampel_filter_vector(v, window_size=10, n_sigmas=3):
    """
    Aplica filtro Hampel para eliminar outliers.
    v: array-like
    window_size: radio de ventana en muestras (entero)
    n_sigmas: umbral en múltiplos de MAD
    Retorna: nuevo array donde los outliers se reemplazan por la mediana local.
    """
    x = np.asarray(v, dtype=float)
    n = len(x)
    new_x = x.copy()
    k = 1.4826
    for i in range(n):
        start = max(i - window_size, 0)
        end = min(i + window_size + 1, n)
        window = x[start:end]
        median = np.nanmedian(window)
        mad = k * np.nanmedian(np.abs(window - median))
        if mad == 0:
            continue
        if np.abs(x[i] - median) > n_sigmas * mad:
            new_x[i] = median
    return new_x

def butterworth_lowpass(data, cutoff, fs, order=4):
    """
    Aplica filtro Butterworth paso bajo (filtfilt para no fase).
    data: array-like
    cutoff: frecuencia de corte (Hz)
    fs: frecuencia de muestreo (Hz)
    order: orden del filtro
    Retorna: señal filtrada (numpy array)
    """
    data = np.asarray(data, dtype=float)
    nyq = 0.5 * fs
    if nyq <= 0:
        return data
    normal_cutoff = cutoff / nyq
    # proteger contra valores >=1
    if normal_cutoff >= 1.0:
        return data
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def preprocess_csv(path: str, output_path: str = None, target_fs: float = None, apply_filters: bool = True):
    """
    Lee CSV crudo, normaliza nombres, asegura columna 'time' en segundos, aplica filtros simples
    y devuelve (df_procesado, meta).
    - path: ruta al CSV de entrada
    - output_path: si se indica, guarda CSV procesado
    - target_fs: frecuencia objetivo si no hay timestamps
    - apply_filters: aplica Hampel y Butterworth simple
    """
    path = Path(path)
    df_raw = pd.read_csv(path)
    df_norm = _normalize_columns(df_raw)
    meta = {'input_path': str(path)}

    # asegurar columna time
    if 'time' in df_norm.columns and df_norm['time'].notna().any():
        t = df_norm['time'].to_numpy(dtype=float)
        if np.median(t) > 1e6:
            t = t / 1000.0
        t = t - t[0]
        df_norm['time'] = t
        fs_est = estimate_fs_from_time(t)
    else:
        if target_fs is None:
            target_fs = 50.0
        n = len(df_norm.index)
        df_norm['time'] = np.arange(n) / float(target_fs)
        fs_est = target_fs
    meta['fs_est'] = fs_est

    # aplicar filtros básicos
    if apply_filters:
        for col in list(df_norm.columns):
            if col == 'time':
                continue
            try:
                x = df_norm[col].to_numpy(dtype=float)
                # Hampel
                x = hampel_filter_vector(x, window_size=10, n_sigmas=3)
                # Butterworth lowpass con cutoff razonable (p.ej. 20 Hz)
                fs_local = fs_est if fs_est is not None else 50.0
                cutoff = min(20.0, 0.45 * 0.5 * fs_local)
                if cutoff > 0:
                    x = butterworth_lowpass(x, cutoff=cutoff, fs=fs_local, order=3)
                df_norm[col] = x
            except Exception:
                pass

    if output_path:
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        df_norm.to_csv(outp, index=False)
        meta['output_path'] = str(outp)
    return df_norm, meta
