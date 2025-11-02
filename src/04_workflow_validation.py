#!/usr/bin/env python3
"""
plot_validation.py
Comparar señales raw vs processed y visualizar ventanas. Guarda figuras en reports/validation/.
Uso para carpetas:
python -u src/04_workflow_validation.py --raw_dir data/raw --processed_dir data/processed --n_examples 5 --fs 100
"""
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_raw(path):
    df = pd.read_csv(path, low_memory=False)
    return df

def load_processed(path):
    df = pd.read_csv(path, low_memory=False)
    return df

def find_matching_processed(raw_path, processed_root):
    # raw filename sin extension -> buscar *_pipeline.csv que contenga ese nombre
    name = Path(raw_path).stem
    matches = list(Path(processed_root).rglob(f"*{name}*_pipeline.csv"))
    return matches[0] if matches else None

def plot_signals(raw_df, proc_df, out_prefix, cols_to_plot=None, fs=100.0):
    # intentar detectar columnas de aceleración
    if cols_to_plot is None:
        possible = [c for c in raw_df.columns if 'accel' in c.lower() or 'acceleration' in c.lower() or c.lower().startswith('acc')]
        # si no en raw, buscar en proc
        if not possible:
            possible = [c for c in proc_df.columns if 'accel' in c.lower() or 'acceleration' in c.lower()]
    cols_to_plot = possible[:3]  # máximo 3 por defecto
    t_raw = None
    if 'Time' in raw_df.columns:
        try:
            # tratar de parsear como HH:MM:SS
            t_raw = pd.to_datetime(raw_df['Time'].astype(str).str.strip(), errors='coerce')
            if t_raw.isna().all():
                t_raw = np.arange(len(raw_df))/fs
            else:
                t_raw = (t_raw - t_raw.iloc[0]).total_seconds()
        except Exception:
            t_raw = np.arange(len(raw_df))/fs
    else:
        t_raw = np.arange(len(raw_df))/fs

    t_proc = np.arange(len(proc_df))/fs

    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    for c in cols_to_plot:
        if c not in raw_df.columns and c not in proc_df.columns:
            continue
        plt.figure(figsize=(10,4))
        if c in raw_df.columns:
            plt.plot(t_raw, pd.to_numeric(raw_df[c], errors='coerce').fillna(0), label='raw', alpha=0.5)
        if c in proc_df.columns:
            plt.plot(t_proc, pd.to_numeric(proc_df[c], errors='coerce').fillna(0), label='processed', linewidth=1)
        plt.title(f"{Path(out_prefix).name} - {c}")
        plt.xlabel("Tiempo [s]")
        plt.ylabel(c)
        plt.legend()
        plt.grid(True)
        out_file = f"{out_prefix}_{c.replace(' ','_')}.png"
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()
        print("Saved", out_file)

def summary_stats_for_file(proc_df, fs=100.0):
    n = len(proc_df)
    duration = n / fs
    return {"n_rows": n, "duration_s": duration}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, help="Archivo raw específico")
    parser.add_argument("--processed", type=str, help="Archivo processed específico")
    parser.add_argument("--raw_dir", type=str, help="Carpeta raw", default="data/raw")
    parser.add_argument("--processed_dir", type=str, help="Carpeta processed", default="data/processed")
    parser.add_argument("--n_examples", type=int, default=3)
    parser.add_argument("--fs", type=float, default=100.0)
    args = parser.parse_args()

    reports_dir = Path("reports/validation")
    reports_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    if args.raw and args.processed:
        targets = [(Path(args.raw), Path(args.processed))]
    elif args.raw:
        # buscar matches procesados
        p = find_matching_processed(args.raw, args.processed_dir)
        if p:
            targets = [(Path(args.raw), p)]
        else:
            print("No match processed found for", args.raw)
            return
    else:
        # listar algunos raw y encontrar el processed correspondiente
        raw_files = list(Path(args.raw_dir).glob("*.csv"))
        raw_files = raw_files[:args.n_examples]
        for r in raw_files:
            p = find_matching_processed(r, args.processed_dir)
            if p:
                targets.append((r, p))

    for raw_p, proc_p in targets:
        print("Validating:", raw_p, "->", proc_p)
        raw_df = load_raw(raw_p)
        proc_df = load_processed(proc_p)
        prefix = reports_dir / (raw_p.stem + "_validation")
        # guardamos stats
        stats = summary_stats_for_file(proc_df, fs=args.fs)
        print("processed rows:", stats)
        plot_signals(raw_df, proc_df, str(prefix), cols_to_plot=None, fs=args.fs)

if __name__ == "__main__":
    main()
