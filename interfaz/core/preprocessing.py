# interfaz/core/preprocessing.py
"""
Preprocesado para la GUI — wrapper que intenta reutilizar el script original
src/01_preprocessing.py (si existe) o utiliza un fallback que aplica:
 - reconstrucción de tiempos robusta
 - remuestreo a target_fs (por defecto 100 Hz)
 - Hampel, Butterworth (fc=15 Hz) y Savitzky-Golay
Exporta:
 - preprocess_csv_for_model(input_path, output_dir=None, target_fs=100, ...)
 - resample_window_to_fs(times, values, target_fs)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util
import sys
import os
from scipy.signal import butter, filtfilt, savgol_filter
import logging

# parámetros por defecto que coinciden con tu script 01_preprocessing.py
DEFAULT_TARGET_FS = 100.0
DEFAULT_FC_BUTTER = 15.0
DEFAULT_HAMPEL_MS = 200.0
DEFAULT_SG_WINDOW = 11
DEFAULT_SG_POLY = 3

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def _try_import_repo_preprocess():
    """
    Si existe src/01_preprocessing.py intenta importarlo dinámicamente y
    devuelve la función process_file si está definida.
    """
    candidates = [
        Path("src") / "01_preprocessing.py",
        Path("01_preprocessing.py"),
    ]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("repo_preprocess", str(p))
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
                if hasattr(mod, "process_file"):
                    return mod.process_file
            except Exception:
                return None
    return None

def _hampel_filter_vector(v, window_size=10, n_sigmas=3):
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

def _butterworth_lowpass(data, cutoff, fs, order=4):
    data = np.asarray(data, dtype=float)
    nyq = 0.5 * fs
    if nyq <= 0:
        return data
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        return data
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def _reconstruct_time_if_needed(df, expected_duration=30.0, target_fs=DEFAULT_TARGET_FS):
    """
    Intento simple de reconstrucción de columna 'time' si no existe.
    (Esta función es menos completa que tu 01_preprocessing.py pero sirve como fallback).
    """
    if 'time' in df.columns:
        t = pd.to_numeric(df['time'], errors='coerce').to_numpy(dtype=float)
        if np.nanmax(t) - np.nanmin(t) > 0:
            # normalizar para empezar en 0
            t = t - t[0]
            return t
    n = len(df)
    return np.arange(n) / float(target_fs)

def _resample_df_to_target_fs(df, time_col='time', target_fs=DEFAULT_TARGET_FS, columns_to_resample=None, duration=None):
    """
    Interpola cada columna en 'columns_to_resample' en una grid uniforme a target_fs.
    Devuelve DataFrame con columna 'time' (desde 0 a duration) y columnas resampleadas.
    """
    if columns_to_resample is None:
        columns_to_resample = [c for c in df.columns if c != time_col]
    times = df[time_col].to_numpy(dtype=float)
    if duration is None:
        duration = float(times[-1] - times[0]) if len(times) >= 2 else (len(times)/target_fs)
    n_samples = int(round(duration * target_fs)) + 1
    new_times = np.linspace(0.0, duration, n_samples)
    out = pd.DataFrame({'time': new_times})
    for col in columns_to_resample:
        vals = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
        # if length mismatch or monotonic issues: fallback linspace for source times
        if len(times) != len(vals) or not np.all(np.diff(times) >= 0):
            src_t = np.linspace(0.0, duration, len(vals))
        else:
            src_t = times - times[0]
        try:
            res = np.interp(new_times, src_t, vals)
        except Exception:
            # fallback simple
            res = np.resize(vals, new_times.shape)
        out[col] = res
    return out

def resample_window_to_fs(times, values, target_fs=DEFAULT_TARGET_FS):
    """
    Dado un vector times (segundos) y valores (1D), devuelve valores resampleados
    uniformemente a target_fs usando interpolación lineal.
    Retorna: (new_times, new_values)
    """
    if len(times) < 2:
        # replicar value
        n = int(round(target_fs * (1.0)))  # 1s default fallback
        new_t = np.linspace(0.0, 1.0, n)
        new_v = np.full_like(new_t, values[-1] if len(values) else 0.0)
        return new_t, new_v
    start = times[0]
    end = times[-1]
    duration = end - start
    if duration <= 0:
        duration = 1.0
    n = int(round(duration * target_fs)) + 1
    new_times = np.linspace(0.0, duration, n)
    # shift original times to 0..duration
    shifted = np.array(times) - start
    new_values = np.interp(new_times, shifted, values)
    return new_times, new_values

def preprocess_csv_for_model(input_path: str,
                             output_dir: str = None,
                             target_fs: float = DEFAULT_TARGET_FS,
                             fc_butter: float = DEFAULT_FC_BUTTER,
                             hampel_ms: float = DEFAULT_HAMPEL_MS,
                             sg_window: int = DEFAULT_SG_WINDOW,
                             sg_poly: int = DEFAULT_SG_POLY,
                             apply_savgol: bool = True,
                             expected_duration: float = 30.0,
                             verbose: bool = False):
    """
    Preprocesa un CSV para que el modelo lo reciba como en entrenamiento.
    Intentará usar src/01_preprocessing.py::process_file si está disponible.
    Si no, usa un pipeline de fallback que:
      - lee CSV, reconstruye tiempos si es necesario,
      - resamplea a target_fs (100 Hz) con interpolación,
      - aplica Hampel, Butterworth (fc=15Hz) y Savitzky-Golay.
    Retorna: (df_processed, meta) donde df_processed tiene columna 'time' y columnas con señales.
    Si se usó el script repo y generó un CSV en disco, df_processed es la lectura de ese CSV de salida.
    """
    logger.info(f"Preprocessing file: {input_path}")
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = Path("interfaz_tmp_processed")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # intentar importar process_file del repo
    repo_proc = _try_import_repo_preprocess()
    if repo_proc is not None:
        # el process_file del repo escribe un CSV en output_dir; invocarlo
        try:
            # El process_file de tu repo acepta (filepath, output_dir, target_fs=100, ...)
            res = repo_proc(str(input_path), str(output_dir), target_fs=target_fs,
                            fc_butter=fc_butter, hampel_window_ms=hampel_ms,
                            sg_window=sg_window, sg_poly=sg_poly,
                            apply_savgol=apply_savgol,
                            expected_duration=expected_duration,
                            verbose=verbose)
            # res es dict con clave 'output' apuntando al CSV procesado
            out_path = res.get("output", "")
            if out_path and Path(out_path).exists():
                df = pd.read_csv(out_path)
                # normalizar nombre columna de tiempo a 'time' si necesario
                # algunos scripts usan 'Time (s)'
                if 'Time (s)' in df.columns:
                    df = df.rename(columns={'Time (s)': 'time'})
                if 'time' not in df.columns and 'Time' in df.columns:
                    df = df.rename(columns={'Time': 'time'})
                meta = {'method': 'repo_process_file', 'output_path': out_path, 'res': res}
                return df, meta
        except Exception as e:
            if verbose:
                print("Error ejecutando repo process_file:", e)
            # fallback al pipeline interno

    # Fallback: pipeline interno similar al 01_preprocessing.py
    df_raw = pd.read_csv(input_path, low_memory=False)
    cols = [c.strip() for c in df_raw.columns]
    # intento heurístico simple para identificar columnas de aceleración/gyro
    acc_cols = [c for c in cols if 'acc' in c.lower() or 'acceleration' in c.lower()]
    gyr_cols = [c for c in cols if 'gyro' in c.lower() or 'angular' in c.lower()]
    # priorizar nombres ax, ay, az, gx, gy, gz si existen
    colmap = {}
    for c in cols:
        lc = c.lower()
        if 'accel' in lc and 'x' in lc: colmap['Acceleration X(g)'] = c
        if 'accel' in lc and 'y' in lc: colmap['Acceleration Y(g)'] = c
        if 'accel' in lc and 'z' in lc: colmap['Acceleration Z(g)'] = c
        if 'gyro' in lc and 'x' in lc: colmap['Gyroscope X(deg/s)'] = c
        if 'gyro' in lc and 'y' in lc: colmap['Gyroscope Y(deg/s)'] = c
        if 'gyro' in lc and 'z' in lc: colmap['Gyroscope Z(deg/s)'] = c
    # fallback picks
    if 'Acceleration X(g)' not in colmap and len(acc_cols) >= 1:
        colmap['Acceleration X(g)'] = acc_cols[0]
    if 'Acceleration Y(g)' not in colmap and len(acc_cols) >= 2:
        colmap['Acceleration Y(g)'] = acc_cols[1]
    if 'Acceleration Z(g)' not in colmap and len(acc_cols) >= 3:
        colmap['Acceleration Z(g)'] = acc_cols[2]
    if 'Gyroscope X(deg/s)' not in colmap and len(gyr_cols) >= 1:
        colmap['Gyroscope X(deg/s)'] = gyr_cols[0]
    if 'Gyroscope Y(deg/s)' not in colmap and len(gyr_cols) >= 2:
        colmap['Gyroscope Y(deg/s)'] = gyr_cols[1]
    if 'Gyroscope Z(deg/s)' not in colmap and len(gyr_cols) >= 3:
        colmap['Gyroscope Z(deg/s)'] = gyr_cols[2]

    # crear df norm con time
    df_norm = pd.DataFrame()
    # buscar columna de tiempo
    time_candidates = [c for c in cols if c.lower().startswith('time') or 'chip' in c.lower()]
    if time_candidates:
        # intentar convertir
        tcol = time_candidates[0]
        try:
            times = pd.to_numeric(df_raw[tcol], errors='coerce').to_numpy(dtype=float)
            # heurística: si los números son grandes -> ms o ns
            if np.nanmedian(times) > 1e6:
                times = times / 1000.0
            # shift
            times = times - times[0]
            df_norm['time'] = times
        except Exception:
            df_norm['time'] = _reconstruct_time_if_needed(df_raw, expected_duration, target_fs)
    else:
        df_norm['time'] = _reconstruct_time_if_needed(df_raw, expected_duration, target_fs)

    # extraer señales encontradas
    for canonical, candidate in colmap.items():
        df_norm[canonical] = pd.to_numeric(df_raw[candidate], errors='coerce')

    # determinar duration y resamplear a target_fs
    duration = float(np.nanmax(df_norm['time']) - np.nanmin(df_norm['time'])) if len(df_norm['time'])>1 else max(1.0, len(df_norm)/target_fs)
    df_resampled = _resample_df_to_target_fs(df_norm, time_col='time', target_fs=target_fs,
                                            columns_to_resample=[c for c in df_norm.columns if c!='time'],
                                            duration=duration)

    # filtros: hampel, butterworth, savgol
    hampel_window = max(1, int((hampel_ms / 1000.0) * target_fs))
    for col in [c for c in df_resampled.columns if c != 'time']:
        arr = df_resampled[col].to_numpy(dtype=float)
        arr = _hampel_filter_vector(arr, window_size=hampel_window, n_sigmas=3)
        try:
            arr = _butterworth_lowpass(arr, cutoff=fc_butter, fs=target_fs, order=4)
        except Exception:
            pass
        if apply_savgol:
            wl = sg_window
            if wl >= len(arr):
                wl = (len(arr)//2)*2 + 1
                if wl < 3: wl = 3
            try:
                arr = savgol_filter(arr, window_length=wl, polyorder=sg_poly)
            except Exception:
                pass
        df_resampled[col] = arr

    # guardar si se desea
    out_path = None
    try:
        fname = Path(input_path).stem + "_processed_for_model.csv"
        out_path = output_dir / fname
        df_resampled.to_csv(out_path, index=False)
    except Exception:
        out_path = None

    meta = {'method': 'fallback_internal', 'output_path': str(out_path) if out_path else None}
    return df_resampled, meta

    # Validar columnas requeridas
    required_cols = ['time', 'Acceleration X(g)', 'Acceleration Y(g)', 'Acceleration Z(g)',
                    'Gyroscope X(deg/s)', 'Gyroscope Y(deg/s)', 'Gyroscope Z(deg/s)']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validar valores
    for col in df.columns:
        if df[col].isnull().any():
            logger.warning(f"NaN values found in column {col}")
        if df[col].dtype in [np.float64, np.float32]:
            outliers = np.abs(df[col] - df[col].mean()) > (df[col].std() * 3)
            if outliers.any():
                logger.warning(f"Possible outliers in {col}: {sum(outliers)} values")
