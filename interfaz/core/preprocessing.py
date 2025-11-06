# interfaz/core/preprocessing.py
"""
Preprocesamiento robusto para los CSV del proyecto.

Funciones públicas:
- preprocess_csv_for_model(filepath, output_dir=..., target_fs=100.0, ...)
    -> devuelve (df_proc, meta) donde df_proc tiene al menos columnas:
       ['time', 'Acceleration X(g)'] (time en segundos, comienza en 0, fs ~ target_fs)
- resample_window_to_fs(time_vec, signal_vec, target_fs=100.0)
    -> remuestrea vectores a target_fs, devuelve (t_rel, y_uniform)
"""

import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter

import logging
logger = logging.getLogger(__name__)


def parse_time_column_to_seconds(time_series):
    s = time_series.dropna().astype(str).str.strip()
    try:
        arr = pd.to_numeric(s, errors='coerce').to_numpy(dtype=float)
        if np.all(~np.isnan(arr)):
            arr = arr - arr[0]
            return arr
    except Exception:
        pass

    times = []
    for t in time_series.astype(str).str.strip():
        if t == "" or t.lower() == "nan":
            times.append(np.nan)
            continue
        try:
            dt = datetime.strptime(t, "%H:%M:%S.%f")
            sec = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
            times.append(sec)
            continue
        except Exception:
            pass
        try:
            dt = datetime.strptime(t, "%H:%M:%S")
            sec = dt.hour * 3600 + dt.minute * 60 + dt.second
            times.append(sec)
            continue
        except Exception:
            pass
        try:
            sec = float(t)
            times.append(sec)
        except Exception:
            raise ValueError(f"No se pudo parsear tiempo: '{t[:40]}'")
    times = np.array(times, dtype=float)
    notnan = ~np.isnan(times)
    if not np.any(notnan):
        raise ValueError("Columna de tiempo vacía o no numérica.")
    first = times[notnan][0]
    times = times - first
    nans = np.isnan(times)
    if np.any(nans):
        good_idx = np.where(~nans)[0]
        times[nans] = np.interp(np.where(nans)[0], good_idx, times[good_idx])
    return times


def infer_time_from_chiptime_column(series, expected_duration=30.0, target_fs=100.0):
    vals_full = pd.to_numeric(series, errors='coerce').to_numpy()
    vals = vals_full[~np.isnan(vals_full)]
    if len(vals) < 2:
        raise ValueError("ChipTime no numérico o escaso")
    vmin = float(np.nanmin(vals)); vmax = float(np.nanmax(vals))
    vrange = vmax - vmin
    diffs = np.diff(vals)
    median_diff = float(np.median(np.abs(diffs))) if len(diffs) > 0 else 0.0

    if vrange >= 1.0:
        if median_diff > 1e6:
            times = (vals - vals[0]) / 1e9
        elif median_diff > 1e3:
            times = (vals - vals[0]) / 1e6
        elif median_diff > 1:
            times = (vals - vals[0]) / 1e3
        else:
            times = vals - vals[0]
        return times

    n = len(vals)
    expected_samples = int(expected_duration * target_fs)
    if vrange < 1.0 and n >= 0.8 * expected_samples:
        times = np.linspace(0.0, expected_duration, n)
        return times

    if median_diff > 1e6:
        times = (vals - vals[0]) / 1e9
    elif median_diff > 1e3:
        times = (vals - vals[0]) / 1e6
    elif median_diff > 1:
        times = (vals - vals[0]) / 1e3
    else:
        times = vals - vals[0]
    return times


def hampel_filter_vector(v, window_size=10, n_sigmas=3):
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
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def resample_window_to_fs(time_vec, signal_vec, target_fs=100.0):
    time_vec = np.asarray(time_vec, dtype=float)
    signal_vec = np.asarray(signal_vec, dtype=float)
    mask = ~(np.isnan(time_vec) | np.isnan(signal_vec))
    time_vec = time_vec[mask]
    signal_vec = signal_vec[mask]
    if len(time_vec) < 2:
        raise ValueError("Demasiados pocos puntos para remuestreo")
    if not np.all(np.diff(time_vec) >= 0):
        idx = np.argsort(time_vec)
        time_vec = time_vec[idx]
        signal_vec = signal_vec[idx]
    t0 = float(time_vec[0])
    tf = float(time_vec[-1])
    duration = tf - t0
    if duration <= 0:
        duration = 1.0
    n_samples = max(2, int(np.round(duration * target_fs)))
    t_uniform = np.linspace(t0, tf, n_samples)
    y_uniform = np.interp(t_uniform, time_vec, signal_vec)
    return t_uniform - t_uniform[0], y_uniform


def preprocess_csv_for_model(filepath,
                             output_dir=None,
                             target_fs=100.0,
                             fc_butter=15.0,
                             hampel_ms=200.0,
                             hampel_nsigmas=3,
                             sg_window=11,
                             sg_poly=3,
                             apply_savgol=True,
                             expected_duration=30.0,
                             tol_lower=5.0,
                             tol_upper=60.0,
                             verbose=True):
    p = Path(filepath)
    if verbose:
        logger.info(f"Preprocessing file: {p}")

    df_raw = pd.read_csv(p, low_memory=False)
    df_raw.columns = [c.strip() for c in df_raw.columns]

    lower_cols = [c.lower() for c in df_raw.columns]
    time_col = None
    chip_col = None
    for c, lc in zip(df_raw.columns, lower_cols):
        if lc == 'time' or lc.startswith('time') or 'time (' in lc or 'timestamp' in lc:
            time_col = c
            break
        if 'chip' in lc and 'time' in lc:
            chip_col = c
    if time_col is None and chip_col is None:
        for c, lc in zip(df_raw.columns, lower_cols):
            if 'chip' in lc:
                chip_col = c
                break

    time_seconds = None
    time_method = "none"
    if time_col is not None:
        try:
            time_seconds = parse_time_column_to_seconds(df_raw[time_col])
            time_method = f"time_col_parsed:{time_col}"
        except Exception:
            time_seconds = None

    if time_seconds is None and chip_col is not None:
        try:
            time_seconds = infer_time_from_chiptime_column(df_raw[chip_col], expected_duration=expected_duration, target_fs=target_fs)
            time_method = f"chip_time_parsed_or_reconstructed:{chip_col}"
        except Exception:
            time_seconds = None

    n_rows = len(df_raw)
    expected_samples = int(expected_duration * target_fs)
    if time_seconds is None:
        if n_rows >= 0.8 * expected_samples:
            time_seconds = np.linspace(0.0, expected_duration, n_rows)
            time_method = "reconstructed_from_nrows"
        else:
            time_seconds = np.arange(n_rows) / target_fs
            time_method = "index_uniform_fallback"

    duration_raw = float(np.nanmax(time_seconds) - np.nanmin(time_seconds)) if len(time_seconds) > 1 else n_rows / target_fs
    if duration_raw < tol_lower and n_rows < 0.8 * expected_samples:
        raise ValueError("Archivo demasiado corto o tiempos incorrectos (duration_raw muy pequeño)")

    colmap = {}
    cols = list(df_raw.columns)
    for c in cols:
        lc = c.lower()
        if ("acceleration x" in lc) or re.search(r"accel.*x", lc) or re.search(r"\bax\b", lc):
            colmap['acc_x'] = c
        if ("acceleration y" in lc) or re.search(r"accel.*y", lc) or re.search(r"\bay\b", lc):
            colmap['acc_y'] = c
        if ("acceleration z" in lc) or re.search(r"accel.*z", lc) or re.search(r"\baz\b", lc):
            colmap['acc_z'] = c

    if not any(k in colmap for k in ('acc_x', 'acc_y', 'acc_z')):
        acc_candidates = [c for c in cols if ('acc' in c.lower() and 'x' in c.lower()) or re.search(r'\bax\b', c.lower())]
        if len(acc_candidates) >= 1:
            colmap['acc_x'] = acc_candidates[0]
        acc_y_cand = [c for c in cols if ('acc' in c.lower() and 'y' in c.lower()) or re.search(r'\bay\b', c.lower())]
        if len(acc_y_cand) >= 1:
            colmap['acc_y'] = acc_y_cand[0]
        acc_z_cand = [c for c in cols if ('acc' in c.lower() and 'z' in c.lower()) or re.search(r'\baz\b', c.lower())]
        if len(acc_z_cand) >= 1:
            colmap['acc_z'] = acc_z_cand[0]

    if 'acc_x' not in colmap:
        raise ValueError(f"No se detectó columna de aceleración X en {p}. Columnas: {cols}")

    duration = duration_raw
    if duration > max(expected_duration, tol_upper):
        duration = expected_duration

    new_n = int(np.floor(duration * target_fs)) + 1
    new_times = np.linspace(0.0, duration, new_n)

    df_out = pd.DataFrame({'time': new_times})

    orig_time = np.array(time_seconds, dtype=float)
    for key, col in [('acc_x', colmap.get('acc_x')), ('acc_y', colmap.get('acc_y')), ('acc_z', colmap.get('acc_z'))]:
        if col is None:
            continue
        series = pd.to_numeric(df_raw[col], errors='coerce').to_numpy(dtype=float)
        if np.all(np.isnan(series)):
            interp = np.zeros_like(new_times)
        else:
            sser = pd.Series(series).ffill().bfill().to_numpy(dtype=float)
            if len(orig_time) != len(sser) or not np.all(np.diff(orig_time) >= 0):
                orig_time_local = np.linspace(0.0, duration_raw if duration_raw > 0 else expected_duration, len(sser))
            else:
                orig_time_local = orig_time
            try:
                interp = np.interp(new_times, orig_time_local, sser)
            except Exception:
                interp = np.resize(sser, new_times.shape)
        out_name = 'Acceleration X(g)' if key == 'acc_x' else ('Acceleration Y(g)' if key == 'acc_y' else 'Acceleration Z(g)')
        df_out[out_name] = interp

    hampel_window = max(1, int((hampel_ms / 1000.0) * target_fs))
    for cname in [c for c in df_out.columns if c != 'time']:
        arr = df_out[cname].to_numpy(dtype=float)
        try:
            arr = hampel_filter_vector(arr, window_size=hampel_window, n_sigmas=hampel_nsigmas)
        except Exception:
            pass
        if len(arr) > 8:
            try:
                arr = butterworth_lowpass(arr, cutoff=fc_butter, fs=target_fs, order=4)
            except Exception:
                pass
        if apply_savgol:
            wl = sg_window
            if wl >= len(arr):
                wl = (len(arr) // 2) * 2 + 1
                if wl < 3:
                    wl = 3
            try:
                arr = savgol_filter(arr, window_length=wl, polyorder=sg_poly)
            except Exception:
                pass
        df_out[cname] = arr

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (p.stem + "_pipeline.csv")
        df_out.to_csv(out_path, index=False)
        if verbose:
            logger.info(f"Procesado guardado en {out_path}")

    meta = {
        'input': str(p),
        'n_raw_rows': n_rows,
        'duration_raw': duration_raw,
        'time_method': time_method,
        'n_processed': len(df_out),
        'target_fs': target_fs
    }

    return df_out, meta
