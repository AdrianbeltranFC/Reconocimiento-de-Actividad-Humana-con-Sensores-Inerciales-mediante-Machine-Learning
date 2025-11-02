#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python src/06_ML_using_top8_orange.py --input_csv "data/final/All_features.csv" --n_splits 5 --cv stratified --final_dir "data/final" --reports_dir "reports" --verbose

Script que:
1) Carga data/final/All_features.csv (o la ruta que indiques)
2) Selecciona automáticamente las 8 features que definiste (no hace falta pegarlas)
3) Guarda la matriz reducida en data/final/All_features_orange_top8.csv
4) Evalúa modelos (SVM-RBF y kNN) con la matriz ORIGINAL (baseline) y con la REDUCIDA (top8)
5) Guarda la tabla comparativa en reports/final_comparison.csv y la muestra en consola

Comentarios:
- Validación por defecto: estratificada (StratifiedKFold n_splits=5). Cambia --cv a "loso" si prefieres LOSO.
- Si faltan columnas entre las 8, el script te lo notificará y usará las que sí existan.
- Todo en español y con mensajes claros.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import time

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.impute import SimpleImputer

# -----------------------
# UTILIDADES
# -----------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def vprint(msg, verbose=True):
    if verbose:
        print(msg)

# -----------------------
# EVALUACIÓN
# -----------------------
def evaluate_stratified(X: pd.DataFrame, y: pd.Series, n_splits=5):
    """
    Evalúa SVM_RBF y kNN con StratifiedKFold.
    Ahora incluye SimpleImputer(strategy='median') en la pipeline para manejar NaN
    de forma segura (imputación por mediana realizada dentro de cada fold).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = {
        'SVM_RBF': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=1.0, gamma='scale'))
        ]),
        'kNN_k5': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=5))
        ])
    }
    results = {}
    for name, model in models.items():
        accs = []; f1s = []
        for train_idx, test_idx in cv.split(X, y):
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(Xtr, ytr)           # el imputer se ajusta solo con Xtr -> sin data leakage
            yp = model.predict(Xte)
            accs.append(accuracy_score(yte, yp))
            _, _, f1, _ = precision_recall_fscore_support(yte, yp, average='macro', zero_division=0)
            f1s.append(f1)
        results[name] = {
            'accs': accs,
            'f1s': f1s,
            'acc_mean': float(np.mean(accs)),
            'acc_std': float(np.std(accs)),
            'f1_mean': float(np.mean(f1s))
        }
    return results

def evaluate_loso(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    """
    Evalúa con Leave-One-Subject-Out (LOSO).
    Incluye imputación por mediana en la pipeline.
    """
    logo = LeaveOneGroupOut()
    models = {
        'SVM_RBF': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=1.0, gamma='scale'))
        ]),
        'kNN_k5': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=5))
        ])
    }
    results = {}
    for name, model in models.items():
        accs = []; f1s = []
        for train_idx, test_idx in logo.split(X, y, groups):
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(Xtr, ytr)
            yp = model.predict(Xte)
            accs.append(accuracy_score(yte, yp))
            _, _, f1, _ = precision_recall_fscore_support(yte, yp, average='macro', zero_division=0)
            f1s.append(f1)
        results[name] = {
            'accs': accs,
            'f1s': f1s,
            'acc_mean': float(np.mean(accs)),
            'acc_std': float(np.std(accs)),
            'f1_mean': float(np.mean(f1s))
        }
    return results

# -----------------------
# MAIN
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Entrena y compara modelos usando las 8 features definidas por Orange (predefinidas en el script).")
    parser.add_argument("--input_csv", type=str, default="data/final/All_features.csv", help="CSV original con todas las features.")
    parser.add_argument("--n_splits", type=int, default=5, help="n_splits para StratifiedKFold.")
    parser.add_argument("--cv", choices=['stratified','loso'], default='stratified', help="Tipo de validación (stratified o loso).")
    parser.add_argument("--final_dir", type=str, default="data/final", help="Carpeta para guardar la matriz reducida.")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Carpeta para guardar la comparación.")
    parser.add_argument("--verbose", action="store_true", help="Mostrar mensajes detallados.")
    args = parser.parse_args()

    t0 = time.time()

    inp = Path(args.input_csv)
    final_dir = Path(args.final_dir)
    reports_dir = Path(args.reports_dir)
    ensure_dir(final_dir); ensure_dir(reports_dir)

    if not inp.exists():
        print("ERROR: No se encontró el CSV de entrada:", inp)
        sys.exit(1)

    # 1) Cargar CSV completo
    df = pd.read_csv(inp, low_memory=False)
    vprint(f"CSV cargado: {inp} — filas: {len(df)} columnas: {len(df.columns)}", args.verbose)

    # metadata que conservamos si existen
    metadata = [c for c in ['Sujeto','Clase','archivo','start_idx','end_idx'] if c in df.columns]
    if 'Clase' not in df.columns:
        print("ERROR: no se encontró la columna 'Clase' en el CSV. Abortando.")
        sys.exit(1)

    # 2) Lista fija de 8 características (las que pediste)
    orange_top8 = [
        "Acceleration X(g)_mean",
        "Acceleration X(g)_std",
        "Acceleration X(g)_var",
        "Acceleration X(g)_median",
        "Acceleration X(g)_iqr",
        "Acceleration X(g)_rms",
        "Acceleration X(g)_ptp",
        "Acceleration X(g)_sma"
    ]

    vprint("Características definidas (Orange top 8):", args.verbose)
    for f in orange_top8:
        vprint(" - " + f, args.verbose)

    # comprobar existencia en el CSV original
    missing = [f for f in orange_top8 if f not in df.columns]
    if missing:
        print("ADVERTENCIA: las siguientes features NO se encontraron en el CSV original y serán omitidas:")
        for m in missing:
            print("  -", m)
    # mantener solo las que existen
    present = [f for f in orange_top8 if f in df.columns]
    if len(present) == 0:
        print("ERROR: Ninguna de las 8 features está en el CSV original. Abortando.")
        sys.exit(1)

    vprint(f"Features presentes que se usarán ({len(present)}): {present}", args.verbose)

    # 3) Crear nueva matriz (metadata + present) y guardarla
    cols_final = metadata + present
    df_reduced = df[cols_final].copy()
    out_csv = final_dir / "All_features_orange_top8.csv"
    df_reduced.to_csv(out_csv, index=False)
    print("CSV reducido guardado en:", out_csv)

    # 4) Evaluar modelos: baseline (todas las columnas numéricas menos metadata) y reducido (present)
    # Baseline
    print("\nEvaluando modelos — BASELINE (todas las características numéricas disponibles)...")
    X_full = df.drop(columns=[c for c in metadata if c in df.columns], errors='ignore')
    X_full = X_full.select_dtypes(include=[np.number])
    y = df['Clase'].astype(str)

    if X_full.shape[1] == 0:
        print("ERROR: no hay columnas numéricas en la matriz original después de eliminar metadata. Abortando.")
        sys.exit(1)

    if args.cv == 'stratified':
        res_full = evaluate_stratified(X_full, y, n_splits=args.n_splits)
    else:
        groups = df[args.group_col].astype(str) if args.group_col in df.columns else None
        if groups is None:
            print("ERROR: pediste LOSO pero no hay columna 'Sujeto' (groups). Cambia --cv a stratified o añade 'Sujeto'.")
            sys.exit(1)
        res_full = evaluate_loso(X_full, y, groups)

    # Reducida
    print("\nEvaluando modelos — REDUCIDA (Orange top8)...")
    X_red = df_reduced.drop(columns=[c for c in metadata if c in df_reduced.columns], errors='ignore')
    X_red = X_red.select_dtypes(include=[np.number])
    if X_red.shape[1] == 0:
        print("ERROR: la matriz reducida no contiene columnas numéricas. Abortando.")
        sys.exit(1)

    if args.cv == 'stratified':
        res_red = evaluate_stratified(X_red, y, n_splits=args.n_splits)
    else:
        groups = df_reduced[args.group_col].astype(str) if args.group_col in df_reduced.columns else None
        if groups is None:
            print("ERROR: pediste LOSO pero no hay columna 'Sujeto' en la matriz reducida. Abortando.")
            sys.exit(1)
        res_red = evaluate_loso(X_red, y, groups)

    # 5) Crear tabla comparativa y guardarla
    rows = []
    for model_name in res_full.keys():
        before = res_full[model_name]
        after = res_red[model_name]
        row = {
            'model': model_name,
            'before_acc_mean': before.get('acc_mean', None),
            'before_acc_std': before.get('acc_std', None),
            'before_f1_mean': before.get('f1_mean', None),
            'after_acc_mean': after.get('acc_mean', None),
            'after_acc_std': after.get('acc_std', None),
            'after_f1_mean': after.get('f1_mean', None)
        }
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    comp_csv = reports_dir / "final_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print("\nTabla comparativa guardada en:", comp_csv)

    # Mostrar resultados en consola con formato
    print("\n=== FEATURES USADAS (Orange top8 presentes) ===")
    for f in present:
        print(" -", f)

    print("\n=== COMPARACIÓN (BASELINE vs ORANGE-TOP8) ===")
    # formateo numérico a 6 decimales
    print(comp_df.to_string(index=False, float_format='{:0.6f}'.format))

    t1 = time.time()
    print("\nCompletado en {:.1f} s.".format(t1 - t0))
    print("CSV reducido en:", out_csv)
    print("Comparación guardada en:", comp_csv)

if __name__ == "__main__":
    main()
