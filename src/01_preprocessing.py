"""
01_preprocessing.py 
- Reconstrucción de tiempos robusta, incluyendo cuando las columnas Time/ChipTime no representan la
  duración real (p. ej. ChipTime rango << 1s pero hay ~3000 muestras).
- Añade campos en resumen: raw_nrows, duration_raw, time_method, status.
- Política: recortar si > expected_duration; rechazar si < tol_lower y pocas muestras. La razón es para
  evitar que archivos mal grabados con tiempos erróneos afecten la calidad del dataset final.
- Filtros: Hampel + Butterworth + Savitzky-Golay.

Como usar:
python src/01_preprocessing.py --input_dir data/raw --output_dir data/processed --target_fs 100

Nota: No olvides activar tu entorno virtual que tiene todas las librerias necesarias para
compilar los códigos (: 
) --fc_butter 15 --hampel_ms 200 --expected_duration 30 --tol_lower 28 --tol_upper 35 --verbose
"""

import os, re, argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter
from tqdm import tqdm


def parse_time_column_to_seconds(time_series):
    times = []
    for t in time_series.astype(str).str.strip():
        try:
            dt = datetime.strptime(t, "%H:%M:%S.%f")
        except ValueError:
            try:
                dt = datetime.strptime(t, "%H:%M:%S")
            except ValueError:
                raise ValueError(f"No se pudo parsear tiempo: '{t[:40]}'")
        sec = dt.hour*3600 + dt.minute*60 + dt.second + dt.microsecond/1e6
        times.append(sec)
    times = np.array(times, dtype=float)
    times = times - times[0]
    return times

def infer_time_from_chiptime_column(series, expected_duration=30.0, target_fs=100):
    vals_full = pd.to_numeric(series, errors='coerce').to_numpy()
    vals = vals_full[~np.isnan(vals_full)]
    if len(vals) < 2:
        raise ValueError("ChipTime no numérico o escaso")
    vmin = float(np.nanmin(vals)); vmax = float(np.nanmax(vals))
    vrange = vmax - vmin
    diffs = np.diff(vals)
    median_diff = float(np.median(np.abs(diffs))) if len(diffs)>0 else 0.0
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
    # Caso: rango pequeño pero muchas muestras => reconstruir uniformemente
    n = len(vals)
    expected_samples = int(expected_duration * target_fs)
    if vrange < 1.0 and n >= 0.8 * expected_samples:
        times = np.linspace(0.0, expected_duration, n)
        return times
    # Fallback heurístico por median_diff
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
    x = np.asarray(v, dtype=float); n = len(x); new_x = x.copy(); k = 1.4826
    for i in range(n):
        start = max(i - window_size, 0); end = min(i + window_size + 1, n)
        window = x[start:end]; median = np.nanmedian(window)
        mad = k * np.nanmedian(np.abs(window - median))
        if mad == 0: continue
        if np.abs(x[i] - median) > n_sigmas * mad:
            new_x[i] = median
    return new_x

def butterworth_lowpass(data, cutoff, fs, order=4):
    nyq = 0.5 * fs; normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# -------------------------
# process_file
# -------------------------
def process_file(filepath, output_dir, target_fs=100, fc_butter=15.0,
                 hampel_window_ms=200, hampel_nsigmas=3, sg_window=11, sg_poly=3,
                 apply_savgol=True, expected_duration=30.0, tol_lower=28.0, tol_upper=35.0,
                 min_samples_ratio=0.8, verbose=False):
    df_raw = pd.read_csv(filepath, low_memory=False)
    original_cols = list(df_raw.columns)
    n_rows = len(df_raw)
    # detectar columnas
    colmap = {}; cols = [c.strip() for c in df_raw.columns]
    for c in cols:
        lc = c.lower()
        if "acceleration x" in lc or re.search(r"accel.*x", lc): colmap['acc_x'] = c
        if "acceleration y" in lc or re.search(r"accel.*y", lc): colmap['acc_y'] = c
        if "acceleration z" in lc or re.search(r"accel.*z", lc): colmap['acc_z'] = c
        if "angular velocity x" in lc or re.search(r"angular.*x", lc) or re.search(r"gyro.*x", lc): colmap['gyr_x'] = c
        if "angular velocity y" in lc or re.search(r"angular.*y", lc) or re.search(r"gyro.*y", lc): colmap['gyr_y'] = c
        if "angular velocity z" in lc or re.search(r"angular.*z", lc) or re.search(r"gyro.*z", lc): colmap['gyr_z'] = c
        if "angle x" in lc: colmap['ang_x'] = c
        if "angle y" in lc: colmap['ang_y'] = c
        if "angle z" in lc: colmap['ang_z'] = c
        if lc == "time" or lc.startswith("time") or "time (" in lc: colmap.setdefault('time_col', c)
        if 'chip time' in lc or ('chip' in lc and 'time' in lc): colmap.setdefault('chip_time', c)

    acc_candidates = [c for c in cols if 'accel' in c.lower() or 'acceleration' in c.lower() or re.search(r'\bax\b', c.lower())]
    if not any(k in colmap for k in ('acc_x','acc_y','acc_z')) and len(acc_candidates) >= 3:
        colmap['acc_x'], colmap['acc_y'], colmap['acc_z'] = acc_candidates[:3]

    # determinar tiempo y registrar metodo
    time_seconds = None; time_method = "none"
    if 'time_col' in colmap:
        try:
            time_seconds = parse_time_column_to_seconds(df_raw[colmap['time_col']])
            time_method = "time_col_parsed"
            if verbose: print("Parsed Time col:", colmap['time_col'])
        except Exception:
            time_seconds = None

    if time_seconds is None and 'chip_time' in colmap:
        try:
            time_seconds = infer_time_from_chiptime_column(df_raw[colmap['chip_time']], expected_duration=expected_duration, target_fs=target_fs)
            time_method = "chip_time_parsed_or_reconstructed"
            if verbose: print("Inferido desde Chip Time:", colmap['chip_time'])
        except Exception:
            time_seconds = None

    if time_seconds is None:
        # Si la columna time existe pero no parseó, o no hay chip_time,
        # y si n_rows ~ expected -> reconstruimos; si no, usamos índice y expected_duration
        expected_samples = int(expected_duration * target_fs)
        if n_rows >= 0.8 * expected_samples:
            time_seconds = np.linspace(0.0, expected_duration, n_rows)
            time_method = "reconstructed_from_nrows"
            if verbose: print("Reconstruido tiempo por n_rows ~ expected")
        else:
            time_seconds = np.linspace(0.0, expected_duration, n_rows)
            time_method = "index_uniform_fallback"
            if verbose: print("Fallback index uniform")

    # calcular duración raw y aplicar política
    duration_raw = float(np.nanmax(time_seconds) - np.nanmin(time_seconds))
    if duration_raw <= 0:
        duration_raw = n_rows / target_fs

    # Si duration_raw muy pequeña pero tenemos muchas muestras -> reconstruir
    expected_samples = int(expected_duration * target_fs)
    if duration_raw < tol_lower and n_rows >= 0.8 * expected_samples:
        time_seconds = np.linspace(0.0, expected_duration, n_rows)
        duration_raw = expected_duration
        time_method = "reconstructed_due_small_duration"
        if verbose: print("Reconstruido porque duration_raw pequeño pero n_rows alto")

    # si sigue siendo muy corto -> rechazar
    if duration_raw < tol_lower and n_rows < 0.8 * expected_samples:
        return {"input": filepath, "output": "", "raw_nrows": n_rows, "duration_raw": duration_raw, "time_method": time_method, "status": "TOO_SHORT"}

    # decidir duration final (recortar si > expected)
    status_note = "OK"
    duration = duration_raw
    if duration > tol_upper or duration > expected_duration:
        duration = expected_duration
        status_note = "TRIMMED"

    new_n = int(np.floor(duration * target_fs)) + 1
    new_times = np.linspace(0.0, duration, new_n)

    # columnas a procesar
    target_cols = []
    for k in ['acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z','ang_x','ang_y','ang_z']:
        if k in colmap: target_cols.append(colmap[k])
    if len(target_cols) == 0:
        raise ValueError(f"No se detectaron columnas de interés en {filepath}. Columnas: {original_cols}")

    orig_time = pd.Series(time_seconds).to_numpy(dtype=float)
    df_out = pd.DataFrame({'Time (s)': new_times})

    for col in target_cols:
        series = pd.to_numeric(df_raw[col], errors='coerce').to_numpy(dtype=float)
        if np.all(np.isnan(series)):
            interp = np.zeros_like(new_times)
        else:
            series = pd.Series(series).ffill().bfill().to_numpy(dtype=float)
            if not np.all(np.diff(orig_time) >= 0) or len(orig_time) != len(series):
                orig_time_local = np.linspace(0.0, duration_raw if duration_raw>0 else expected_duration, len(series))
            else:
                orig_time_local = orig_time
            try:
                interp = np.interp(new_times, orig_time_local, series)
            except Exception:
                interp = np.resize(series, new_times.shape)
        df_out[col] = interp

    # filtros
    hampel_window = max(1, int((hampel_window_ms/1000.0) * target_fs))
    if hampel_window < 1: hampel_window = 1
    for col in target_cols:
        arr = df_out[col].to_numpy(dtype=float)
        arr = hampel_filter_vector(arr, window_size=hampel_window, n_sigmas=hampel_nsigmas)
        if len(arr) > (4*3):
            try:
                arr = butterworth_lowpass(arr, cutoff=fc_butter, fs=target_fs, order=4)
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
        df_out[col] = arr

    # sujeto y clase
    fname = os.path.basename(filepath)
    sujeto = "Unknown"; clase = "Desconocida"
    m_s = re.search(r"(S\d{1,2})", fname, re.IGNORECASE)
    if m_s: sujeto = m_s.group(1)
    if re.search(r'quieto', fname, re.IGNORECASE): clase = "Quieto"
    elif re.search(r'caminar', fname, re.IGNORECASE): clase = "Caminar"
    elif re.search(r'correr', fname, re.IGNORECASE): clase = "Correr"

    out_folder = Path(output_dir) / sujeto / clase
    out_folder.mkdir(parents=True, exist_ok=True)
    out_name = fname.replace(".csv", "_pipeline.csv")
    out_path = out_folder / out_name
    df_out.to_csv(out_path, index=False)

    # check final sample count
    if new_n < min_samples_ratio * expected_samples:
        return {"input": filepath, "output": str(out_path), "raw_nrows": n_rows, "duration_raw": duration_raw, "time_method": time_method, "status": "FEW_SAMPLES"}
    return {"input": filepath, "output": str(out_path), "raw_nrows": n_rows, "duration_raw": duration_raw, "time_method": time_method, "status": status_note, "n_processed": new_n}

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Limpieza y filtrado CSV IMU (v2)")
    parser.add_argument("--input_dir", type=str, default="data/raw", help="Carpeta con CSV crudos")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Carpeta salida")
    parser.add_argument("--target_fs", type=float, default=100.0, help="Frecuencia final (Hz)")
    parser.add_argument("--fc_butter", type=float, default=15.0, help="Cutoff Butterworth (Hz)")
    parser.add_argument("--hampel_ms", type=float, default=200.0, help="Ventana Hampel en ms")
    parser.add_argument("--expected_duration", type=float, default=30.0, help="Duración esperada (s)")
    parser.add_argument("--tol_lower", type=float, default=28.0, help="Min duración aceptable (s)")
    parser.add_argument("--tol_upper", type=float, default=35.0, help="Max duración para recortar (s)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir); output_dir = Path(args.output_dir)
    csv_files = list(input_dir.rglob("*.csv"))
    if len(csv_files) == 0:
        print("No se encontraron CSV en", input_dir); return

    resumen = []
    for f in tqdm(csv_files, desc="Archivos"):
        try:
            res = process_file(str(f), str(output_dir), target_fs=args.target_fs,
                               fc_butter=args.fc_butter, hampel_window_ms=args.hampel_ms,
                               expected_duration=args.expected_duration, tol_lower=args.tol_lower,
                               tol_upper=args.tol_upper, verbose=args.verbose)
            # res es dict
            resumen.append(res)
        except Exception as e:
            resumen.append({"input": str(f), "output": "", "raw_nrows": 0, "duration_raw": 0.0, "time_method": "err", "status": f"ERR: {e}"})
            if args.verbose: print("Error procesando", f, ":", e)

    resumen_df = pd.DataFrame(resumen)
    out_summary_path = output_dir / "resumen_limpieza.csv"
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)
    resumen_df.to_csv(out_summary_path, index=False)
    print("Procesamiento finalizado. Resumen guardado en", out_summary_path)

if __name__ == "__main__":
    main()
