#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_finalize_models_and_reports.py

Genera:
- PNG (matrices de confusión) con títulos en español (por ejemplo:
  "Métricas para SVM con todas las características",
  "Métricas para SVM con 8 características").
- CSV: matriz de confusión y classification report (para reporte).
- Modelos finales serializados con nombres claros:
  SVM_todas_caracteristicas.joblib
  SVM_8_caracteristicas.joblib
- Un CSV resumen comparativo.

Uso ejemplo:
python src/07_finalize_models_and_reports.py --input_csv data/final/All_features.csv --reduced_csv data/final/All_features_orange_top8.csv --n_splits 5 --save_models --verbose
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import re
import os

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

# ---------------------------
# Utilidades
# ---------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def printv(msg, verbose):
    if verbose:
        print(msg)

def sanitize_filename(s: str) -> str:
    """
    Devuelve una versión segura para filename.
    Mantiene espacios y caracteres legibles, pero elimina los
    caracteres que causan problemas en Windows/Unix.
    """
    # quitar / \ : * ? " < > | y controlar dobles espacios
    s2 = re.sub(r'[\/\\\:\*\?\"\<\>\|]', '_', s)
    s2 = re.sub(r'\s+', ' ', s2).strip()
    return s2

# ---------------------------
# Plot + CSV helpers (en español)
# ---------------------------
def plot_confusion_matrix_and_save(cm, labels, out_png: Path, title: str):
    """Guarda matriz de confusión como imagen con título en español."""
    ensure_dir(out_png.parent)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Etiqueta real')
    plt.xlabel('Etiqueta predicha')
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def save_confusion_csv(cm, labels, out_csv: Path):
    ensure_dir(out_csv.parent)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_csv)

def save_classification_report_csv(y_true, y_pred, labels, out_csv: Path):
    ensure_dir(out_csv.parent)
    cr = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    df_cr = pd.DataFrame(cr).T
    df_cr.to_csv(out_csv)
    return df_cr

# ---------------------------
# CV predict and save (con nombres y títulos en español)
# ---------------------------
def cv_predict_and_save(X, y, pipeline, cv, labels, out_dir: Path, display_title: str, filename_base: str, verbose=False):
    """
    Ejecuta cross_val_predict, guarda:
      - PNG con título = display_title
      - CSV: filename_base + "_confusion_matrix.csv"
      - CSV: filename_base + "_classification_report.csv"
    filename_base se sanea internamente.
    """
    ensure_dir(out_dir)
    # predecir por CV
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, method='predict', n_jobs=-1)
    # métricas
    acc = accuracy_score(y, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=labels)

    safe_base = sanitize_filename(filename_base)
    png_path = out_dir / f"{safe_base}.png"
    csv_cm_path = out_dir / f"{safe_base}_confusion_matrix.csv"
    csv_cr_path = out_dir / f"{safe_base}_classification_report.csv"

    # guardar PNG con título en español
    plot_confusion_matrix_and_save(cm, labels, png_path, display_title)
    # guardar CSVs
    save_confusion_csv(cm, labels, csv_cm_path)
    save_classification_report_csv(y, y_pred, labels, csv_cr_path)

    if verbose:
        print(f"Guardados: {png_path}  {csv_cm_path}  {csv_cr_path}  (acc={acc:.6f}, f1_macro={f1:.6f})")

    return {'accuracy': acc, 'f1_macro': f1, 'png': png_path, 'csv_conf': csv_cm_path, 'csv_cr': csv_cr_path, 'confusion_matrix': cm}

# ---------------------------
# MAIN
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Genera PNG/CSV con títulos en español y guarda modelos con nombres claros.")
    parser.add_argument("--input_csv", type=str, default="data/final/All_features.csv", help="CSV completo (baseline).")
    parser.add_argument("--reduced_csv", type=str, default="data/final/All_features_orange_top8.csv", help="CSV reducido (Orange top8).")
    parser.add_argument("--target_col", type=str, default="Clase")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--save_models", action="store_true", help="Si se pasa, guarda modelos finales entrenados sobre la matriz reducida.")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--reports_dir", type=str, default="reports/final_models")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    inp = Path(args.input_csv)
    red = Path(args.reduced_csv)
    reports_dir = Path(args.reports_dir)
    models_dir = Path(args.models_dir)
    ensure_dir(reports_dir); ensure_dir(models_dir)

    if not inp.exists():
        raise FileNotFoundError(f"No existe {inp}")
    if not red.exists():
        raise FileNotFoundError(f"No existe {red}")

    # Cargar CSVs
    df_full = pd.read_csv(inp, low_memory=False)
    df_red = pd.read_csv(red, low_memory=False)

    if args.target_col not in df_full.columns:
        raise ValueError(f"No existe columna target '{args.target_col}' en {inp}")

    y_full = df_full[args.target_col].astype(str)
    labels = sorted(y_full.unique())

    # Baseline X (solo numéricas)
    metadata = [c for c in ['Sujeto','Clase','archivo','start_idx','end_idx'] if c in df_full.columns]
    X_full = df_full.drop(columns=[c for c in metadata if c in df_full.columns], errors='ignore').select_dtypes(include=[np.number])
    X_red = df_red.drop(columns=[c for c in metadata if c in df_red.columns], errors='ignore').select_dtypes(include=[np.number])

    if X_full.shape[1] == 0 or X_red.shape[1] == 0:
        raise ValueError("X_full o X_red no contienen features numéricas.")

    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    # Pipelines
    svm_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', C=1.0, gamma='scale'))])
    knn_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))])

     # NOMBRES/TITULOS solicitados por ti (en español)
    title_svm_full = "Métricas para SVM con todas las características"
    title_svm_red  = "Métricas para SVM con 8 características"
    title_knn_full = "Métricas para k-NN con todas las características"
    title_knn_red  = "Métricas para k-NN con 8 características"

    # filenames (exactamente igual que los títulos)
    base_svm_full = "Métricas para SVM con todas las características"
    base_svm_red  = "Métricas para SVM con 8 características"
    base_knn_full = "Métricas para k-NN con todas las características"
    base_knn_red  = "Métricas para k-NN con 8 características"

    # Ejecutar CV y guardar archivos (PNG + CSVs)
    printv("Evaluando BASELINE para SVM y k-NN y guardando PNG/CSV...", args.verbose)
    res_svm_full = cv_predict_and_save(X_full, y_full, svm_pipe, cv, labels, reports_dir, title_svm_full, base_svm_full, verbose=args.verbose)
    res_knn_full = cv_predict_and_save(X_full, y_full, knn_pipe, cv, labels, reports_dir, title_knn_full, base_knn_full, verbose=args.verbose)

    printv("Evaluando REDUCIDA (8 features) para SVM y k-NN y guardando PNG/CSV...", args.verbose)
    y_red = df_red[args.target_col].astype(str)
    res_svm_red = cv_predict_and_save(X_red, y_red, svm_pipe, cv, labels, reports_dir, title_svm_red, base_svm_red, verbose=args.verbose)
    res_knn_red = cv_predict_and_save(X_red, y_red, knn_pipe, cv, labels, reports_dir, title_knn_red, base_knn_red, verbose=args.verbose)

    # Guardar resumen comparativo (CSV)
    comp_rows = [
        {'model': 'SVM_RBF', 'baseline_acc_mean': res_svm_full['accuracy'], 'baseline_f1_macro': res_svm_full['f1_macro'],
         'reduced_acc_mean': res_svm_red['accuracy'], 'reduced_f1_macro': res_svm_red['f1_macro']},
        {'model': 'kNN_k5', 'baseline_acc_mean': res_knn_full['accuracy'], 'baseline_f1_macro': res_knn_full['f1_macro'],
         'reduced_acc_mean': res_knn_red['accuracy'], 'reduced_f1_macro': res_knn_red['f1_macro']}
    ]
    comp_df = pd.DataFrame(comp_rows)
    comp_csv_name = sanitize_filename("Comparación_baseline_vs_8_caracteristicas.csv")
    comp_csv = reports_dir / comp_csv_name
    comp_df.to_csv(comp_csv, index=False)
    print("Resumen comparativo guardado en:", comp_csv)

    # Entrenar y guardar modelos finales (si el usuario pidió guardar)
    if args.save_models:
        print("Entrenando modelos finales sobre la matriz reducida y guardando modelos con nombres claros...")
        svm_pipe.fit(X_red, y_red)
        knn_pipe.fit(X_red, y_red)
        # nombres de modelos (saneados para filename)
        model_svm_name = sanitize_filename("SVM_todas_caracteristicas.joblib")
        model_knn_name = sanitize_filename("kNN_8_caracteristicas.joblib")
        joblib.dump(svm_pipe, Path(models_dir) / model_svm_name)
        joblib.dump(knn_pipe, Path(models_dir) / model_knn_name)
        print("Modelos guardados en:", models_dir, "->", model_svm_name, model_knn_name)

    t1 = time.time()
    print("\nProceso completado en {:.1f} s.".format(t1 - t0))
    print("Revisa la carpeta:", reports_dir)
    if args.save_models:
        print("Modelos guardados en:", models_dir)

if __name__ == "__main__":
    main()
