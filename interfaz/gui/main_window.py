# interfaz/gui/main_window.py
"""
Interfaz principal para AVDs con sensores inerciales.

- Modo A: cargar CSV -> preprocesado coherente con el entrenamiento (100 Hz) ->
          segmentación en ventanas (2.56 s) -> extracción top8 -> predicción por ventana ->
          graficar y guardar diagnósticos.
- Modo B: adquisición BLE (o simulación) -> buffer -> guardar CSV crudo.
- Modo C: realtime -> toma ventana cruda -> remuestrea a 100 Hz -> extrae top8 -> predecir.

Uso:
    python -m interfaz.gui.main_window
"""

import sys
import time
import csv
import random
from collections import Counter, deque
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QTableWidget, QTableWidgetItem, QLineEdit, QGroupBox,
    QStackedWidget, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QCursor

import pyqtgraph as pg
from pyqtgraph import ScatterPlotItem

# Tema oscuro opcional
try:
    import qdarkstyle
    HAS_QDARK = True
except Exception:
    HAS_QDARK = False

# BLE opcional (Bleak)
try:
    from bleak import BleakClient
    HAS_BLEAK = True
except Exception:
    HAS_BLEAK = False

# Importar funciones/core del repo - asegúrate que existan en interfaz/core/
from interfaz.core.preprocessing import preprocess_csv_for_model, resample_window_to_fs
from interfaz.core.pipeline import run_pipeline_on_processed_df
from interfaz.core.top8 import window_df_to_top8, ORANGE_TOP8

import joblib
import numpy as np
import pandas as pd

# Configuración por defecto (ruta del modelo proporcionada por ti)
DEFAULT_BLE_MAC = "eb:33:2e:f9:57:b5"
READ_CHAR_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"
DEFAULT_MODEL_PATH = Path("models/k-NN/kNN_8_caracteristicas.joblib")


# ---------------------------
# Helper: predicción segura usando DataFrame con nombres esperados
# ---------------------------
def predict_with_model_df(model, df_features: pd.DataFrame):
    """
    model: sklearn Pipeline (imputer, scaler, knn)
    df_features: DataFrame que contiene ORANGE_TOP8 y opcionalmente window_center_time
    Retorna: (y_pred, y_proba)
    Asegura pasar DataFrame con los nombres en el orden esperado por model.feature_names_in_ si existe.
    """
    Xdf = df_features.copy()
    # quitar columna de tiempo si existe
    if 'window_center_time' in Xdf.columns:
        Xdf = Xdf.drop(columns=['window_center_time'])

    # comprobar columnas necesarias
    for c in ORANGE_TOP8:
        if c not in Xdf.columns:
            raise ValueError(f"Falta la columna de feature esperada: '{c}' en df_features")

    # reordenar según lo que el modelo espera (si es que lo guarda)
    if hasattr(model, 'feature_names_in_'):
        expected = list(model.feature_names_in_)
        missing = [c for c in expected if c not in Xdf.columns]
        if missing:
            raise ValueError(f"Faltan columnas esperadas por el modelo: {missing}")
        Xdf = Xdf[expected]

    # predecir
    y_pred = model.predict(Xdf)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(Xdf)
        except Exception:
            y_proba = None
    return y_pred, y_proba


# ---------------------------
# BLEWorker: hilo que lee notificaciones BLE y emite muestras (dict)
# ---------------------------
class BLEWorker(QThread):
    sample_received = pyqtSignal(dict)
    status = pyqtSignal(str)

    def __init__(self, address: str, notify_uuid: str, parent=None):
        super().__init__(parent)
        self.address = address
        self.notify_uuid = notify_uuid
        self._running = False
        # intentar usar parser del repo si existe
        try:
            from src.device_model import parse_notify as _repo_parse
            self._repo_parser = _repo_parse
        except Exception:
            self._repo_parser = None

    def run(self):
        if not HAS_BLEAK:
            self.status.emit("bleak no disponible")
            return
        self._running = True
        import asyncio
        asyncio.run(self._run_async())

    async def _run_async(self):
        client = BleakClient(self.address)
        try:
            await client.connect()
            self.status.emit("Conectado (BLE)")
            def callback(sender, data: bytearray):
                sample = None
                if self._repo_parser is not None:
                    try:
                        sample = self._repo_parser(data)
                    except Exception:
                        sample = None
                if sample is None:
                    # intentar parse ASCII "ax,ay,az,gx,gy,gz"
                    try:
                        s = data.decode("utf-8").strip()
                        parts = s.split(",")
                        if len(parts) >= 6:
                            ax, ay, az, gx, gy, gz = map(float, parts[:6])
                            sample = {
                                "Acceleration X(g)": ax,
                                "Acceleration Y(g)": ay,
                                "Acceleration Z(g)": az,
                                "Gyroscope X(deg/s)": gx,
                                "Gyroscope Y(deg/s)": gy,
                                "Gyroscope Z(deg/s)": gz,
                                "time": time.time()
                            }
                    except Exception:
                        sample = None
                if sample:
                    self.sample_received.emit(sample)
            await client.start_notify(self.notify_uuid, callback)
            while self._running:
                await asyncio.sleep(0.1)
            await client.stop_notify(self.notify_uuid)
            await client.disconnect()
            self.status.emit("Desconectado (BLE)")
        except Exception as e:
            self.status.emit(f"Error BLE: {e}")

    def stop(self):
        self._running = False


# ---------------------------
# Modo A: cargar y procesar archivo
# ---------------------------
class ModeAWidget(QWidget):
    def __init__(self, go_home_callback):
        super().__init__()
        self.go_home_callback = go_home_callback

        # Botones
        self.load_btn = QPushButton("📂 Importar archivo (.csv)")
        self.proc_btn = QPushButton("⚙️ Procesar archivo (para modelo)")
        self.back_btn = QPushButton("⬅️ Volver")

        # Tabla donde mostramos las 8 features (valores)
        self.table = QTableWidget(len(ORANGE_TOP8), 3)
        self.table.setHorizontalHeaderLabels(["Característica", "Valor", "Unidad"])
        unidades = ["g", "g", "g^2", "g", "g", "g", "g", "g·muestras"]
        for i, name in enumerate(ORANGE_TOP8):
            it = QTableWidgetItem(name)
            it.setFlags(it.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(i, 0, it)
            self.table.setItem(i, 2, QTableWidgetItem(unidades[i]))

        # Gráficas
        self.plot_raw = pg.PlotWidget(title="Acc X (resampleado a 100 Hz)")
        self.plot_pred = pg.PlotWidget(title="Predicciones por ventana (tiempo)")
        self.raw_curve = self.plot_raw.plot(pen=pg.mkPen('#6C63FF', width=1.5))
        self.plot_raw.setLabel('left', 'Acc X (g)')
        self.plot_raw.setLabel('bottom', 'Tiempo (s)')

        self.pred_label = QLabel("Última ventana: —")
        self.pred_label.setAlignment(Qt.AlignCenter)

        # Layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.load_btn)
        top_layout.addWidget(self.proc_btn)
        top_layout.addWidget(self.back_btn)

        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addWidget(self.plot_raw)
        layout.addWidget(self.plot_pred)
        layout.addWidget(self.table)
        layout.addWidget(self.pred_label)
        self.setLayout(layout)

        # Estado
        self.loaded_path = None
        self.df_processed = None
        self.df_features = None
        self.y_pred = None
        self.y_proba = None

        # Conexiones
        self.load_btn.clicked.connect(self.select_file)
        self.proc_btn.clicked.connect(self.process_file)
        self.back_btn.clicked.connect(self.go_home_callback)

    def select_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Seleccionar CSV", filter="CSV Files (*.csv)")
        if not fname:
            return
        self.loaded_path = fname
        QMessageBox.information(self, "Archivo", f"Archivo seleccionado:\n{fname}")

    def process_file(self):
        """Proceso completo: preprocesado para modelo (100Hz) + pipeline ventanas 2.56s + predicciones."""
        if not self.loaded_path:
            QMessageBox.warning(self, "Sin archivo", "Importa un archivo CSV primero.")
            return
        try:
            # ---- 1) Preprocesado compatible con entrenamiento ----
            tmp_out = Path("interfaz_tmp_processed")
            tmp_out.mkdir(parents=True, exist_ok=True)
            # preprocess_csv_for_model -> devuelve df resampleado a 100Hz y filtrado
            df_proc, meta = preprocess_csv_for_model(
                self.loaded_path,
                output_dir=str(tmp_out),
                target_fs=100.0,
                fc_butter=15.0,
                hampel_ms=200.0,
                sg_window=11,
                sg_poly=3,
                apply_savgol=True,
                expected_duration=30.0,
                verbose=False
            )
            self.df_processed = df_proc

            # ---- 2) Pipeline: ventanas y predicción (usa ventana de 2.56 s en pipeline) ----
            # IMPORTANTE: run_pipeline_on_processed_df debe aceptar solo el df procesado
            df_feat, indices, y_pred, y_proba = run_pipeline_on_processed_df(df_proc)

            self.df_features = df_feat
            self.y_pred = y_pred
            self.y_proba = y_proba

            # ---- 2b) Validar que df_feat contenga las 8 features; intentar mapear si faltan ----
            expected_columns = [
                'Acceleration X(g)_mean', 'Acceleration X(g)_std', 'Acceleration X(g)_var',
                'Acceleration X(g)_median', 'Acceleration X(g)_iqr',
                'Acceleration X(g)_rms', 'Acceleration X(g)_ptp', 'Acceleration X(g)_sma'
            ]
            missing = [c for c in expected_columns if c not in df_feat.columns]
            if missing:
                # intentar heurística de mapeo
                mapped = {}
                available = list(df_feat.columns)
                for exp in expected_columns:
                    base = exp.split("_", 1)[0].lower().strip()
                    found = None
                    for col in available:
                        if base in col.lower():
                            found = col
                            break
                    if found:
                        mapped[exp] = found
                if set(mapped.keys()) == set(expected_columns):
                    df_reordered = pd.DataFrame()
                    for exp in expected_columns:
                        df_reordered[exp] = df_feat[mapped[exp]].astype(float)
                    if 'window_center_time' in df_feat.columns:
                        df_reordered['window_center_time'] = df_feat['window_center_time']
                    df_feat = df_reordered
                    self.df_features = df_feat
                else:
                    raise ValueError(
                        "Las columnas generadas por el pipeline no coinciden con las esperadas por el modelo.\n"
                        f"Esperadas: {expected_columns}\n"
                        f"Disponibles en df_feat: {available}\n"
                        "Ajusta el preprocesado/pipeline o añade un mapeo."
                    )

            # ---- Guardar diagnósticos: features y predicciones por ventana ----
            diag_dir = Path("interfaz_diagnostics")
            diag_dir.mkdir(parents=True, exist_ok=True)
            try:
                df_feat.to_csv(diag_dir / f"{Path(self.loaded_path).stem}_features_windows.csv", index=False)
            except Exception:
                try:
                    pd.DataFrame(df_feat).to_csv(diag_dir / f"{Path(self.loaded_path).stem}_features_windows.csv", index=False)
                except Exception:
                    pass

            pred_df = pd.DataFrame({
                'window_center_time': df_feat['window_center_time'] if 'window_center_time' in df_feat.columns else np.arange(len(y_pred)),
                'pred': list(y_pred)
            })
            if y_proba is not None:
                pred_df['conf_max'] = [float(np.max(p)) for p in y_proba]
            pred_df.to_csv(diag_dir / f"{Path(self.loaded_path).stem}_pred_windows.csv", index=False)

            # ---- 3) Graficar señal resampleada (Acc X) ----
            t = df_proc['time'].to_numpy(dtype=float)
            y = df_proc.get('Acceleration X(g)', np.zeros_like(t)).to_numpy(dtype=float)
            self.plot_raw.clear()
            self.plot_raw.plot(t - t[0], y, pen=pg.mkPen('#6C63FF', width=1.5))
            self.plot_raw.setLabel('bottom', 'Tiempo (s)')

            # ---- 4) Graficar predicciones por ventana como puntos coloreados ----
            times = df_feat['window_center_time'].to_numpy(dtype=float)
            unique_labels = list(dict.fromkeys(list(y_pred)))
            class_map = {lbl: i for i, lbl in enumerate(unique_labels)}
            palette = ['#6C63FF', '#48C9B0', '#F39C12', '#E74C3C', '#9B59B6']
            color_map = {lbl: palette[i % len(palette)] for i, lbl in enumerate(unique_labels)}
            yvals = np.array([class_map[lbl] for lbl in y_pred])

            self.plot_pred.clear()
            scatter = ScatterPlotItem(size=10)
            spots = []
            for xi, yi, lbl in zip(times - t[0], yvals, y_pred):
                spots.append({'pos': (float(xi), float(yi)), 'brush': color_map[lbl], 'symbol': 'o'})
            scatter.addPoints(spots)
            self.plot_pred.addItem(scatter)
            ticks = [(v, str(k)) for k, v in class_map.items()]
            self.plot_pred.getPlotItem().getAxis('left').setTicks([ticks])
            self.plot_pred.setLabel('bottom', 'Tiempo (s)')
            self.plot_pred.setLabel('left', 'Clase (ventana)')

            # ---- 5) Tabla: medias de features (top8) ----
            avg = df_feat[ORANGE_TOP8].mean(axis=0)
            for i, col in enumerate(ORANGE_TOP8):
                val = avg.get(col, np.nan)
                self.table.setItem(i, 1, QTableWidgetItem(f"{val:.6g}"))

            # ---- 6) Última predicción y confianza ----
            last_pred = y_pred[-1]
            conf = None
            if y_proba is not None:
                conf = float(np.max(y_proba[-1]))
            self.pred_label.setText(f"Última ventana: {last_pred}  •  Confianza: {conf:.3f}" if conf is not None else f"Última ventana: {last_pred}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error procesando archivo:\n{e}")


# ---------------------------
# Modo B: adquisición BLE (o simulación)
# ---------------------------
class ModeBWidget(QWidget):
    def __init__(self, go_home_callback):
        super().__init__()
        self.go_home_callback = go_home_callback

        # Inputs y botones
        self.mac_input = QLineEdit(DEFAULT_BLE_MAC)
        self.uuid_input = QLineEdit(READ_CHAR_UUID)
        self.start_btn = QPushButton("▶️ Iniciar adquisición")
        self.stop_btn = QPushButton("⏹ Detener")
        self.save_btn = QPushButton("💾 Guardar datos")
        self.back_btn = QPushButton("⬅️ Volver")

        # Plot
        self.plot = pg.PlotWidget(title="Señales en tiempo real (AccX / GyrX)")
        self.acc_curve = self.plot.plot(pen=pg.mkPen('#6C63FF', width=2))
        self.gyr_curve = self.plot.plot(pen=pg.mkPen('#F39C12', width=2))

        conn_group = QGroupBox("🔗 Conexión BLE")
        conn_layout = QHBoxLayout()
        conn_layout.addWidget(QLabel("MAC:"))
        conn_layout.addWidget(self.mac_input)
        conn_layout.addWidget(QLabel("UUID:"))
        conn_layout.addWidget(self.uuid_input)
        conn_group.setLayout(conn_layout)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.back_btn)

        layout = QVBoxLayout()
        layout.addWidget(conn_group)
        layout.addLayout(btn_layout)
        layout.addWidget(self.plot)
        self.setLayout(layout)

        # Estado
        self.worker = None
        self.buffer = []  # lista de dicts con muestras
        self.simulate = not HAS_BLEAK

        self.timer = QTimer()
        self.timer.setInterval(100)  # 10 Hz render
        self.timer.timeout.connect(self.update_plot)

        # Conexiones
        self.start_btn.clicked.connect(self.start_stream)
        self.stop_btn.clicked.connect(self.stop_stream)
        self.save_btn.clicked.connect(self.save_raw)
        self.back_btn.clicked.connect(self.go_home_callback)

        # buffers para plot
        self.data_acc = []
        self.data_gyr = []
        self.time_buff = []

    def start_stream(self):
        self.buffer.clear()
        self.data_acc.clear()
        self.data_gyr.clear()
        self.time_buff.clear()
        if self.simulate:
            self.timer.start()
            QMessageBox.information(self, "Simulación", "BLE no disponible: usando simulación.")
            return
        mac = self.mac_input.text().strip()
        uuid = self.uuid_input.text().strip()
        if not mac or not uuid:
            QMessageBox.warning(self, "Parámetros", "Proporciona MAC y UUID.")
            return
        self.worker = BLEWorker(mac, uuid)
        self.worker.sample_received.connect(self.on_sample)
        self.worker.status.connect(lambda s: QMessageBox.information(self, "BLE", s) if "Error" in s else None)
        self.worker.start()
        self.timer.start()

    def stop_stream(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
        self.timer.stop()
        QMessageBox.information(self, "Stop", "Adquisición detenida.")

    def on_sample(self, sample: dict):
        """Recibe muestra y la añade al buffer"""
        if "time" not in sample:
            sample["time"] = time.time()
        self.buffer.append(sample)
        t = sample["time"]
        ax = sample.get("Acceleration X(g)", np.nan)
        gx = sample.get("Gyroscope X(deg/s)", np.nan)
        self.time_buff.append(t)
        self.data_acc.append(ax)
        self.data_gyr.append(gx)
        maxlen = 2000
        if len(self.time_buff) > maxlen:
            self.time_buff.pop(0); self.data_acc.pop(0); self.data_gyr.pop(0)

    def update_plot(self):
        """Si no hay BLE, simula datos; en cualquiera de los casos actualiza curvas."""
        if self.simulate:
            # generar muestra simulada
            t = time.time()
            ax = random.uniform(-1.2, 1.2)
            gx = random.uniform(-200, 200)
            self.on_sample({
                "Acceleration X(g)": ax,
                "Acceleration Y(g)": random.uniform(-1.2, 1.2),
                "Acceleration Z(g)": random.uniform(-1.2, 1.2),
                "Gyroscope X(deg/s)": gx,
                "Gyroscope Y(deg/s)": random.uniform(-200, 200),
                "Gyroscope Z(deg/s)": random.uniform(-200, 200),
                "time": t
            })

        if not self.time_buff:
            return
        self.acc_curve.setData(self.data_acc)
        self.gyr_curve.setData(self.data_gyr)

    def save_raw(self):
        if not self.buffer:
            QMessageBox.warning(self, "Guardar", "No hay datos para guardar.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Guardar CSV crudo", filter="CSV Files (*.csv)")
        if not fname:
            return
        keys = list(self.buffer[0].keys())
        with open(fname, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.buffer:
                writer.writerow(row)
        QMessageBox.information(self, "Guardado", f"Archivo guardado:\n{fname}")


# ---------------------------
# Modo C: realtime - tomar ventana cruda -> remuestrear a 100Hz -> extraer top8 -> predecir
# ---------------------------
class ModeCWidget(QWidget):
    def __init__(self, go_home_callback, source_buffer_callable):
        """
        source_buffer_callable: callable sin args que retorna la lista buffer (ModeB.buffer)
        """
        super().__init__()
        self.go_home_callback = go_home_callback
        self.get_source_buffer = source_buffer_callable

        self.plot = pg.PlotWidget(title="Acc X (ventana remuestreada a 100Hz)")
        self.curve = self.plot.plot(pen=pg.mkPen('#48C9B0', width=2))

        self.pred_label = QLabel("Predicción actual: —")
        self.conf_label = QLabel("Confianza: —")
        for lbl in [self.pred_label, self.conf_label]:
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-weight:bold;")

        self.start_btn = QPushButton("Start realtime")
        self.stop_btn = QPushButton("Stop realtime")
        self.back_btn = QPushButton("⬅️ Volver")
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.back_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.plot)
        layout.addWidget(self.pred_label)
        layout.addWidget(self.conf_label)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        # estado
        self.timer = QTimer()
        self.timer.setInterval(200)  # cada 200 ms intenta predecir (latencia adicional pequeña)
        self.timer.timeout.connect(self._on_tick)
        self.model = None
        self.window_sec = 2.56  # ventana usada en entrenamiento
        self.overlap = 0.5
        self.window_len = None
        self.fs_assumed = 15.0  # si no hay timestamps, asumimos 15Hz crudo

        # historial de predicciones para suavizar (majority vote)
        self.pred_history = deque(maxlen=5)

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.back_btn.clicked.connect(self.go_home_callback)

    def load_model(self):
        if not DEFAULT_MODEL_PATH.exists():
            QMessageBox.critical(self, "Modelo", f"Modelo no encontrado en {DEFAULT_MODEL_PATH}")
            return False
        try:
            self.model = joblib.load(DEFAULT_MODEL_PATH)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Modelo", f"No se pudo cargar el modelo:\n{e}")
            return False

    def start(self):
        ok = self.load_model()
        if not ok:
            return
        self.timer.start()
        QMessageBox.information(self, "Realtime", f"Realtime started (ventana={self.window_sec}s).")

    def stop(self):
        self.timer.stop()
        QMessageBox.information(self, "Realtime", "Realtime detenido")

    def _on_tick(self):
        """
        Flujo:
         - leer buffer (lista de dicts)
         - seleccionar muestras que cubran los últimos self.window_sec segundos
         - extraer acc_x y tiempos
         - remuestrear a 100 Hz con resample_window_to_fs
         - construir df_res con columnas 'time' y 'Acceleration X(g)'
         - calcular top8 y predecir (usando DataFrame con nombres)
         - actualizar GUI (graf, etiqueta)
        """
        buf = self.get_source_buffer()
        if not buf or len(buf) < 3:
            return

        # estimar fs crudo si hay timestamps
        times_all = np.array([s.get('time', np.nan) for s in buf], dtype=float)
        times_valid = times_all[~np.isnan(times_all)]
        fs_raw = None
        if len(times_valid) >= 3:
            dt = np.diff(times_valid)
            dt = dt[~np.isnan(dt) & (dt > 0)]
            if len(dt) > 0:
                fs_raw = float(np.round(1.0 / np.median(dt), 6))

        # seleccionar ventana por tiempo (preferible)
        last_time = buf[-1].get('time', time.time())
        if fs_raw is not None:
            window_raw = [s for s in buf if s.get('time', 0) >= last_time - self.window_sec]
        else:
            # fallback: tomar última N muestras aproximadas
            approx_n = int(round(self.window_sec * self.fs_assumed))
            window_raw = list(buf[-approx_n:])

        if len(window_raw) < 3:
            return

        # construir arrays de tiempo y acc_x a partir de window_raw
        raw_times = np.array([s.get('time', np.nan) for s in window_raw], dtype=float)
        acc_x = np.array([s.get('Acceleration X(g)', np.nan) for s in window_raw], dtype=float)
        # si todos NaN, salir
        if np.all(np.isnan(acc_x)):
            return

        # remuestrear a 100Hz
        try:
            new_t_rel, new_acc = resample_window_to_fs(raw_times, acc_x, target_fs=100.0)
        except Exception:
            # si falla interpolación, intentar simple rellenado
            try:
                duration = raw_times[-1] - raw_times[0] if not np.isnan(raw_times[-1]) and not np.isnan(raw_times[0]) else self.window_sec
                if duration <= 0:
                    duration = self.window_sec
                n = int(round(duration * 100.0)) + 1
                new_t_rel = np.linspace(0.0, duration, n)
                new_acc = np.interp(new_t_rel, np.linspace(0.0, duration, len(acc_x)), acc_x)
            except Exception:
                return

        # construir df_res compatible con pipeline (time en segundos desde 0)
        df_res = pd.DataFrame({'time': new_t_rel})
        df_res['Acceleration X(g)'] = new_acc

        # actualizar gráfico (señal remuestreada)
        try:
            self.curve.setData(new_t_rel, new_acc)
        except Exception:
            self.curve.setData(new_acc)

        # extraer features (top8) y predecir
        try:
            feats = window_df_to_top8(df_res, acc_x_col_name='Acceleration X(g)')
            # convertir a DataFrame con nombres ORANGE_TOP8
            df_single = pd.DataFrame([feats.values], columns=ORANGE_TOP8)
            # predecir usando DataFrame (evita warnings)
            y_pred, y_proba = predict_with_model_df(self.model, df_single)
            pred_label = y_pred[0]
            pred_conf = float(np.max(y_proba[0])) if y_proba is not None else None

            # suavizado: majority vote en pred_history
            self.pred_history.append(pred_label)
            hist = Counter(list(self.pred_history))
            smooth_pred = hist.most_common(1)[0][0]

            self.pred_label.setText(f"Predicción actual (suavizada): {smooth_pred}")
            self.conf_label.setText(f"Confianza: {pred_conf:.3f}" if pred_conf is not None else "Confianza: -")
        except Exception as e:
            # no bloquear GUI: imprimir error en consola
            print("Error realtime predict:", e)


# ---------------------------
# Home y MainWindow
# ---------------------------
class HomeScreen(QWidget):
    def __init__(self, switch_callback):
        super().__init__()
        self.switch_callback = switch_callback

        self.logo = QLabel("🦶")
        self.logo.setAlignment(Qt.AlignCenter)
        self.logo.setStyleSheet("font-size: 80px;")

        title = QLabel("Sistema de Análisis de Marcha Humana")
        subtitle = QLabel("Basado en sensores IMU WT901DCL")
        for lbl in [title, subtitle]:
            lbl.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #6C63FF;")
        subtitle.setStyleSheet("font-size: 14px; color: #AAAAAA;")

        btnA = QPushButton("🅰️ Modo A - Procesar Archivo")
        btnB = QPushButton("🅱️ Modo B - Adquisición BLE")
        btnC = QPushButton("🧠 Modo C - Procesamiento en Tiempo Real")

        for btn, color, idx in zip([btnA, btnB, btnC], ["#6C63FF", "#48C9B0", "#F39C12"], [1, 2, 3]):
            btn.setStyleSheet(
                f"background-color: {color}; color: white; padding: 12px; font-weight: bold; border-radius: 10px;"
            )
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.clicked.connect(lambda checked, x=idx: self.switch_callback(x))

        footer = QLabel("Desarrollado por Adrián Beltrán, Lalo Varela y Paola Guzmán")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #666666; font-size: 12px;")

        layout = QVBoxLayout()
        layout.addWidget(self.logo)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(20)
        layout.addWidget(btnA)
        layout.addWidget(btnB)
        layout.addWidget(btnC)
        layout.addSpacing(20)
        layout.addWidget(footer)
        layout.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Análisis de Marcha - WT901DCL")
        self.resize(1100, 700)

        self.stack = QStackedWidget()
        self.home = HomeScreen(self.switch_to_mode)
        # Crear ModeB antes porque ModeC lee su buffer
        self.modeB = ModeBWidget(self.go_home)
        self.modeC = ModeCWidget(self.go_home, source_buffer_callable=lambda: self.modeB.buffer)
        self.modeA = ModeAWidget(self.go_home)

        # Orden de indices: 0-home, 1-modeA, 2-modeB, 3-modeC
        self.stack.addWidget(self.home)   # 0
        self.stack.addWidget(self.modeA)  # 1
        self.stack.addWidget(self.modeB)  # 2
        self.stack.addWidget(self.modeC)  # 3

        self.setCentralWidget(self.stack)

    def switch_to_mode(self, mode_index):
        self.stack.setCurrentIndex(mode_index)

    def go_home(self):
        self.stack.setCurrentIndex(0)


# ---------------------------
# Launch
# ---------------------------
def main():
    app = QApplication(sys.argv)
    if HAS_QDARK:
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
