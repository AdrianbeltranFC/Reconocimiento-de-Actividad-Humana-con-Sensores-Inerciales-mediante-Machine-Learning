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
import threading
import asyncio
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QTableWidget, QTableWidgetItem, QLineEdit, QGroupBox,
    QStackedWidget, QFileDialog, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QCursor

import pyqtgraph as pg
from pyqtgraph import ScatterPlotItem

# optional dark theme
try:
    import qdarkstyle
    HAS_QDARK = True
except Exception:
    HAS_QDARK = False

# Bleak
try:
    from bleak import BleakClient, BleakScanner
    HAS_BLEAK = True
except Exception:
    HAS_BLEAK = False

# project core imports (must exist in repo)
from interfaz.core.preprocessing import preprocess_csv_for_model, resample_window_to_fs
from interfaz.core.pipeline import run_pipeline_on_processed_df
from interfaz.core.top8 import window_df_to_top8, ORANGE_TOP8

import joblib
import numpy as np
import pandas as pd

# Defaults
DEFAULT_BLE_MAC = "eb:33:2e:f9:57:b5"
READ_CHAR_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"
DEFAULT_MODEL_PATH = Path("models/k-NN/kNN_8_caracteristicas.joblib")

# CSV headers requested
CSV_HEADERS = [
    "ime", "Device name", "Chip Time()", "Acceleration X(g)", "Acceleration Y(g)", "Acceleration Z(g)",
    "Angular velocity X(°/s)", "Angular velocity Y(°/s)", "Angular velocity Z(°/s)",
    "ShiftX(mm)", "ShiftY(mm)", "ShiftZ(mm)", "SpeedX(mm/s)", "SpeedY(mm/s)", "SpeedZ(mm/s)",
    "Angle X(°)", "Angle Y(°)", "Angle Z(°)"
]


def try_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def format_time(t: float) -> str:
    try:
        dt = datetime.fromtimestamp(t)
        return dt.strftime("%H:%M:%S.") + f"{int((t % 1) * 1000):03d}"
    except Exception:
        return str(t)


# WT901 binary parser (20-byte frame)
def parse_wt901_frame(frame_bytes: bytes) -> Optional[Dict[str, Any]]:
    if len(frame_bytes) < 20:
        return None
    b = frame_bytes
    try:
        if b[1] != 0x61:
            return None
    except Exception:
        return None

    def get_signed_int16(v):
        if v >= 0x8000:
            v -= 0x10000
        return v

    try:
        ax = get_signed_int16((b[3] << 8) | b[2]) / 32768.0 * 16.0
        ay = get_signed_int16((b[5] << 8) | b[4]) / 32768.0 * 16.0
        az = get_signed_int16((b[7] << 8) | b[6]) / 32768.0 * 16.0
        gx = get_signed_int16((b[9] << 8) | b[8]) / 32768.0 * 2000.0
        gy = get_signed_int16((b[11] << 8) | b[10]) / 32768.0 * 2000.0
        gz = get_signed_int16((b[13] << 8) | b[12]) / 32768.0 * 2000.0
        ang_x = get_signed_int16((b[15] << 8) | b[14]) / 32768.0 * 180.0
        ang_y = get_signed_int16((b[17] << 8) | b[16]) / 32768.0 * 180.0
        ang_z = get_signed_int16((b[19] << 8) | b[18]) / 32768.0 * 180.0

        return {
            "Ax": round(ax, 6), "Ay": round(ay, 6), "Az": round(az, 6),
            "Gx": round(gx, 6), "Gy": round(gy, 6), "Gz": round(gz, 6),
            "AngX": round(ang_x, 6), "AngY": round(ang_y, 6), "AngZ": round(ang_z, 6)
        }
    except Exception:
        return None


# Normalize parser output to CSV_HEADERS + timestamp numeric key
def normalize_parsed(parsed: Dict[str, Any], state: Dict[str, Any], device_name: str = "") -> Dict[str, Any]:
    out = {h: "" for h in CSV_HEADERS}
    # timestamp handling
    t_abs = None
    if "time" in parsed:
        t_abs = try_float(parsed.get("time"))
    elif "timestamp" in parsed:
        t_abs = try_float(parsed.get("timestamp"))
    elif "Chip Time()" in parsed and parsed.get("Chip Time()"):
        out["Chip Time()"] = parsed.get("Chip Time()")
    elif "DeltaTime" in parsed:
        dt = try_float(parsed.get("DeltaTime"))
        if dt is not None:
            if state.get("last_time") is None:
                state["last_time"] = time.time() - dt
            state["last_time"] = state["last_time"] + dt
            t_abs = state["last_time"]

    if t_abs is None:
        t_abs = time.time()
    state["last_time"] = t_abs
    out["ime"] = format_time(t_abs)
    out["Device name"] = device_name or parsed.get("Device name", "")

    # Map acceleration / gyro / angles
    ax = parsed.get("Acceleration X(g)") or parsed.get("Ax") or parsed.get("ax") or parsed.get("AccX")
    ay = parsed.get("Acceleration Y(g)") or parsed.get("Ay") or parsed.get("ay") or parsed.get("AccY")
    az = parsed.get("Acceleration Z(g)") or parsed.get("Az") or parsed.get("az") or parsed.get("AccZ")
    gx = parsed.get("Gyroscope X(deg/s)") or parsed.get("Gx") or parsed.get("gx")
    gy = parsed.get("Gyroscope Y(deg/s)") or parsed.get("Gy") or parsed.get("gy")
    gz = parsed.get("Gyroscope Z(deg/s)") or parsed.get("Gz") or parsed.get("gz")
    angx = parsed.get("AngX") or parsed.get("Angle X(°)") or parsed.get("Angle X")
    angy = parsed.get("AngY") or parsed.get("Angle Y(°)") or parsed.get("Angle Y")
    angz = parsed.get("AngZ") or parsed.get("Angle Z(°)") or parsed.get("Angle Z")

    def fmt_val(v):
        vv = try_float(v)
        return "" if vv is None else f"{vv:.6g}"

    out["Acceleration X(g)"] = fmt_val(ax)
    out["Acceleration Y(g)"] = fmt_val(ay)
    out["Acceleration Z(g)"] = fmt_val(az)
    out["Angular velocity X(°/s)"] = fmt_val(gx)
    out["Angular velocity Y(°/s)"] = fmt_val(gy)
    out["Angular velocity Z(°/s)"] = fmt_val(gz)
    out["Angle X(°)"] = fmt_val(angx)
    out["Angle Y(°)"] = fmt_val(angy)
    out["Angle Z(°)"] = fmt_val(angz)

    # Shift/Speed optional
    for k in ["ShiftX(mm)", "ShiftY(mm)", "ShiftZ(mm)", "SpeedX(mm/s)", "SpeedY(mm/s)", "SpeedZ(mm/s)"]:
        if parsed.get(k) is not None:
            out[k] = fmt_val(parsed.get(k))

    # keep numeric timestamp for plotting
    out["timestamp"] = t_abs
    return out


# Async BLE client like previous version
class AsyncBLEClient(QObject):
    sample_received = pyqtSignal(dict)
    status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._loop = None
        self._thread = None
        self._client = None
        try:
            from src.device_model import parse_notify as _repo_parse
            self._repo_parser = _repo_parse
        except Exception:
            self._repo_parser = None

    def _start_loop_thread(self):
        if self._loop is not None:
            return
        self._loop = asyncio.new_event_loop()
        def _run(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()
        self._thread = threading.Thread(target=_run, args=(self._loop,), daemon=True)
        self._thread.start()

    def start(self, address: str, notify_uuid: str):
        if not HAS_BLEAK:
            self.status.emit("bleak no disponible")
            return
        self._start_loop_thread()
        asyncio.run_coroutine_threadsafe(self._connect_and_listen(address, notify_uuid), self._loop)

    def stop(self):
        if not self._loop:
            return
        fut = asyncio.run_coroutine_threadsafe(self._stop_and_cleanup(), self._loop)
        try:
            fut.result(timeout=5.0)
        except Exception:
            pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=1.0)
        self._loop = None
        self._thread = None
        self._client = None
        self.status.emit("BLE loop detenido")

    async def _stop_and_cleanup(self):
        try:
            if self._client:
                await self._client.disconnect()
                self._client = None
        except Exception:
            pass

    async def _connect_and_listen(self, address: str, notify_uuid: str):
        try:
            self.status.emit("Haciendo discovery BLE (5s)...")
            try:
                devices = await BleakScanner.discover(timeout=5.0)
            except Exception as e:
                devices = []
                self.status.emit(f"Discovery falló: {e}")

            addr_norm = address.lower().replace('-', ':')
            chosen = address
            treat_as_name = (':' not in address and any(c.isalpha() for c in address))
            for d in devices:
                if d.address and d.address.lower().replace('-', ':') == addr_norm:
                    chosen = d.address
                    break
            if chosen == address and treat_as_name:
                for d in devices:
                    if d.name and address.lower() in d.name.lower():
                        chosen = d.address
                        break

            self.status.emit(f"Intentando conectar a {chosen} ...")
            client = BleakClient(chosen)
            self._client = client
            await client.connect()
            self.status.emit("Conectado (BLE)")

            services = None
            try:
                services = await client.get_services()
            except AttributeError:
                try:
                    services = client.services
                except Exception:
                    services = None
            except Exception:
                try:
                    services = client.services
                except Exception:
                    services = None

            if services is None:
                self.status.emit("No pude obtener servicios GATT del dispositivo (services=None).")
                await client.disconnect()
                self._client = None
                return

            notify_char = None
            if notify_uuid:
                for svc in services:
                    for ch in getattr(svc, "characteristics", []):
                        try:
                            if str(ch.uuid).lower() == notify_uuid.lower():
                                notify_char = ch.uuid
                                break
                        except Exception:
                            continue
                    if notify_char:
                        break
            if notify_char is None:
                for svc in services:
                    for ch in getattr(svc, "characteristics", []):
                        props = getattr(ch, "properties", None)
                        if isinstance(props, (list, tuple)):
                            if any('notify' in str(p).lower() or 'indicate' in str(p).lower() for p in props):
                                notify_char = ch.uuid
                                break
                        else:
                            if props and ('notify' in str(props).lower() or 'indicate' in str(props).lower()):
                                notify_char = ch.uuid
                                break
                    if notify_char:
                        break

            if notify_char is None:
                all_chars = []
                for svc in services:
                    for ch in getattr(svc, "characteristics", []):
                        all_chars.append(str(getattr(ch, "uuid", ch)))
                self.status.emit(f"No encontré characteristic NOTIFY. Disponibles: {all_chars}")
                await client.disconnect()
                self._client = None
                return

            def _notify_cb(sender, data: bytearray):
                sample = None
                if self._repo_parser is not None:
                    try:
                        sample = self._repo_parser(data)
                    except Exception:
                        sample = None
                if sample is not None and isinstance(sample, dict):
                    try:
                        self.sample_received.emit(sample)
                    except Exception:
                        pass
                    return
                try:
                    s = data.decode("utf-8").strip()
                    parts = s.split(",")
                    if len(parts) >= 6:
                        ax, ay, az, gx, gy, gz = map(float, parts[:6])
                        self.sample_received.emit({
                            "Acceleration X(g)": ax,
                            "Acceleration Y(g)": ay,
                            "Acceleration Z(g)": az,
                            "Gyroscope X(deg/s)": gx,
                            "Gyroscope Y(deg/s)": gy,
                            "Gyroscope Z(deg/s)": gz,
                            "time": time.time()
                        })
                        return
                except Exception:
                    pass
                try:
                    self.sample_received.emit({"raw_bytes": bytes(data)})
                except Exception:
                    pass

            await client.start_notify(str(notify_char), _notify_cb)
            self.status.emit(f"Notify iniciado en {notify_char}")

            while True:
                await asyncio.sleep(0.5)
                try:
                    if not client.is_connected:
                        break
                except Exception:
                    break

        except Exception as e:
            self.status.emit(f"Error BLE (connect/listen): {e}")
            try:
                if self._client:
                    await self._client.disconnect()
            except Exception:
                pass
            self._client = None
            return


# Mode A (unchanged)
class ModeAWidget(QWidget):
    def __init__(self, go_home_callback):
        super().__init__()
        self.go_home_callback = go_home_callback
        self.load_btn = QPushButton("📂 Importar archivo (.csv)")
        self.proc_btn = QPushButton("⚙️ Procesar archivo (para modelo)")
        self.back_btn = QPushButton("⬅️ Volver")

        self.table = QTableWidget(len(ORANGE_TOP8), 3)
        self.table.setHorizontalHeaderLabels(["Característica", "Valor", "Unidad"])
        unidades = ["g", "g", "g^2", "g", "g", "g", "g", "g·muestras"]
        for i, name in enumerate(ORANGE_TOP8):
            it = QTableWidgetItem(name)
            it.setFlags(it.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(i, 0, it)
            self.table.setItem(i, 2, QTableWidgetItem(unidades[i]))

        self.plot_raw = pg.PlotWidget(title="Acc X (resampleado a 100 Hz)")
        self.plot_pred = pg.PlotWidget(title="Predicciones por ventana (tiempo)")
        self.raw_curve = self.plot_raw.plot(pen=pg.mkPen('#6C63FF', width=1.5))
        self.plot_raw.setLabel('left', 'Acc X (g)')
        self.plot_raw.setLabel('bottom', 'Tiempo (s)')

        self.pred_label = QLabel("Última ventana: —")
        self.pred_label.setAlignment(Qt.AlignCenter)

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

        self.loaded_path = None
        self.df_processed = None
        self.df_features = None
        self.y_pred = None
        self.y_proba = None

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
        if not self.loaded_path:
            QMessageBox.warning(self, "Sin archivo", "Importa un archivo CSV primero.")
            return
        try:
            tmp_out = Path("interfaz_tmp_processed")
            tmp_out.mkdir(parents=True, exist_ok=True)
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
            df_feat, indices, y_pred, y_proba = run_pipeline_on_processed_df(df_proc)
            self.df_features = df_feat
            self.y_pred = y_pred
            self.y_proba = y_proba

            expected_columns = [
                'Acceleration X(g)_mean', 'Acceleration X(g)_std', 'Acceleration X(g)_var',
                'Acceleration X(g)_median', 'Acceleration X(g)_iqr',
                'Acceleration X(g)_rms', 'Acceleration X(g)_ptp', 'Acceleration X(g)_sma'
            ]
            missing = [c for c in expected_columns if c not in df_feat.columns]
            if missing:
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

            t = df_proc['time'].to_numpy(dtype=float)
            y = df_proc.get('Acceleration X(g)', np.zeros_like(t)).to_numpy(dtype=float)
            self.plot_raw.clear()
            self.plot_raw.plot(t - t[0], y, pen=pg.mkPen('#6C63FF', width=1.5))
            self.plot_raw.setLabel('bottom', 'Tiempo (s)')

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
            try:
                self.plot_pred.getPlotItem().getAxis('left').setTicks([ticks])
            except Exception:
                pass
            self.plot_pred.setLabel('bottom', 'Tiempo (s)')
            self.plot_pred.setLabel('left', 'Clase (ventana)')

            avg = df_feat[ORANGE_TOP8].mean(axis=0)
            for i, col in enumerate(ORANGE_TOP8):
                val = avg.get(col, np.nan)
                self.table.setItem(i, 1, QTableWidgetItem(f"{val:.6g}"))

            last_pred = y_pred[-1]
            conf = None
            if y_proba is not None:
                conf = float(np.max(y_proba[-1]))
            self.pred_label.setText(f"Última ventana: {last_pred}  •  Confianza: {conf:.3f}" if conf is not None else f"Última ventana: {last_pred}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error procesando archivo:\n{e}")


# Mode B with 3 plots
class ModeBWidget(QWidget):
    def __init__(self, go_home_callback):
        super().__init__()
        self.go_home_callback = go_home_callback

        # inputs & buttons
        self.mac_input = QLineEdit(DEFAULT_BLE_MAC)
        self.uuid_input = QLineEdit(READ_CHAR_UUID)
        self.start_btn = QPushButton("▶️ Iniciar adquisición")
        self.stop_btn = QPushButton("⏹ Detener")
        self.save_btn = QPushButton("💾 Guardar datos")
        self.back_btn = QPushButton("⬅️ Volver")

        # left: three stacked plots
        self.acc_plot = pg.PlotWidget(title="Aceleraciones (X, Y, Z) [g]")
        self.gyro_plot = pg.PlotWidget(title="Velocidad Angular (X, Y, Z) [°/s]")
        self.ang_plot = pg.PlotWidget(title="Ángulos (X, Y, Z) [°]")

        # create 3 curves each
        self.acc_curves = {
            'x': self.acc_plot.plot(pen=pg.mkPen('#6C63FF', width=1.5)),
            'y': self.acc_plot.plot(pen=pg.mkPen('#48C9B0', width=1.5)),
            'z': self.acc_plot.plot(pen=pg.mkPen('#F39C12', width=1.5))
        }
        self.gyro_curves = {
            'x': self.gyro_plot.plot(pen=pg.mkPen('#6C63FF', width=1.5)),
            'y': self.gyro_plot.plot(pen=pg.mkPen('#48C9B0', width=1.5)),
            'z': self.gyro_plot.plot(pen=pg.mkPen('#F39C12', width=1.5))
        }
        self.ang_curves = {
            'x': self.ang_plot.plot(pen=pg.mkPen('#6C63FF', width=1.5)),
            'y': self.ang_plot.plot(pen=pg.mkPen('#48C9B0', width=1.5)),
            'z': self.ang_plot.plot(pen=pg.mkPen('#F39C12', width=1.5))
        }

        for p in (self.acc_plot, self.gyro_plot, self.ang_plot):
            p.showGrid(x=True, y=True)
            p.setLabel('bottom', 'Tiempo (s)')

        # table on right
        self.table = QTableWidget()
        self.ui_columns = ["ime", "Device name", "Chip Time()", "Acceleration X(g)", "Acceleration Y(g)", "Acceleration Z(g)",
                           "Angular velocity X(°/s)", "Angular velocity Y(°/s)", "Angular velocity Z(°/s)",
                           "Angle X(°)", "Angle Y(°)", "Angle Z(°)"]
        self.table.setColumnCount(len(self.ui_columns))
        self.table.setHorizontalHeaderLabels(self.ui_columns)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.max_table_rows = 500

        # layout
        conn_group = QGroupBox("🔗 Conexión BLE")
        conn_layout = QHBoxLayout()
        conn_layout.addWidget(QLabel("MAC / Name:"))
        conn_layout.addWidget(self.mac_input)
        conn_layout.addWidget(QLabel("UUID:"))
        conn_layout.addWidget(self.uuid_input)
        conn_group.setLayout(conn_layout)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.back_btn)

        left_plots_layout = QVBoxLayout()
        left_plots_layout.addWidget(self.acc_plot, 1)
        left_plots_layout.addWidget(self.gyro_plot, 1)
        left_plots_layout.addWidget(self.ang_plot, 1)

        center_layout = QHBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left_plots_layout)
        center_layout.addWidget(left_widget, 2)
        center_layout.addWidget(self.table, 1)

        main_layout = QVBoxLayout()
        main_layout.addWidget(conn_group)
        main_layout.addLayout(btn_layout)
        main_layout.addLayout(center_layout)
        self.setLayout(main_layout)

        # state
        self.ble_client = None
        self.normalized_buffer: List[Dict[str, Any]] = []
        self.simulate = not HAS_BLEAK

        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_plot)

        self.start_btn.clicked.connect(self.start_stream)
        self.stop_btn.clicked.connect(self.stop_stream)
        self.save_btn.clicked.connect(self.save_raw)
        self.back_btn.clicked.connect(self.go_home_callback)

        # plotting data lists
        self.time_buff: List[float] = []
        self.acc_x: List[float] = []
        self.acc_y: List[float] = []
        self.acc_z: List[float] = []
        self.gyro_x: List[float] = []
        self.gyro_y: List[float] = []
        self.gyro_z: List[float] = []
        self.ang_x: List[float] = []
        self.ang_y: List[float] = []
        self.ang_z: List[float] = []

        self._norm_state = {"last_time": None}

    def _on_status(self, text: str):
        print("[BLE status]", text)

    def _on_sample_emitted(self, sample: Dict[str, Any]):
        try:
            if "raw_bytes" in sample:
                b = sample["raw_bytes"]
                parsed_frame = None
                for i in range(0, len(b) - 19, 1):
                    chunk = b[i:i+20]
                    pf = parse_wt901_frame(chunk)
                    if pf is not None:
                        parsed_frame = pf
                        break
                if parsed_frame is None:
                    return
                parsed = {
                    "Ax": parsed_frame["Ax"], "Ay": parsed_frame["Ay"], "Az": parsed_frame["Az"],
                    "Gx": parsed_frame["Gx"], "Gy": parsed_frame["Gy"], "Gz": parsed_frame["Gz"],
                    "AngX": parsed_frame["AngX"], "AngY": parsed_frame["AngY"], "AngZ": parsed_frame["AngZ"]
                }
                normalized = normalize_parsed(parsed, self._norm_state, device_name=self.mac_input.text().strip())
                self._append_normalized(normalized)
                return

            normalized = normalize_parsed(sample, self._norm_state, device_name=self.mac_input.text().strip())
            self._append_normalized(normalized)
        except Exception as e:
            print("Error handling sample:", e)

    def _append_normalized(self, normalized_row: Dict[str, Any]):
        # ensure numeric timestamp exists
        tnum = normalized_row.get("timestamp")
        if tnum is None:
            tnum = self._norm_state.get("last_time", time.time())
            normalized_row["timestamp"] = tnum

        self.normalized_buffer.append(normalized_row)
        if len(self.normalized_buffer) > 20000:
            self.normalized_buffer.pop(0)

        # append to plotting arrays converting strings -> floats
        try:
            t = float(normalized_row.get("timestamp", time.time()))
            ax = try_float(normalized_row.get("Acceleration X(g)")) if normalized_row.get("Acceleration X(g)") != "" else np.nan
            ay = try_float(normalized_row.get("Acceleration Y(g)")) if normalized_row.get("Acceleration Y(g)") != "" else np.nan
            az = try_float(normalized_row.get("Acceleration Z(g)")) if normalized_row.get("Acceleration Z(g)") != "" else np.nan
            gx = try_float(normalized_row.get("Angular velocity X(°/s)")) if normalized_row.get("Angular velocity X(°/s)") != "" else np.nan
            gy = try_float(normalized_row.get("Angular velocity Y(°/s)")) if normalized_row.get("Angular velocity Y(°/s)") != "" else np.nan
            gz = try_float(normalized_row.get("Angular velocity Z(°/s)")) if normalized_row.get("Angular velocity Z(°/s)") != "" else np.nan
            angx = try_float(normalized_row.get("Angle X(°)")) if normalized_row.get("Angle X(°)") != "" else np.nan
            angy = try_float(normalized_row.get("Angle Y(°)")) if normalized_row.get("Angle Y(°)") != "" else np.nan
            angz = try_float(normalized_row.get("Angle Z(°)")) if normalized_row.get("Angle Z(°)") != "" else np.nan

            self.time_buff.append(t)
            self.acc_x.append(ax); self.acc_y.append(ay); self.acc_z.append(az)
            self.gyro_x.append(gx); self.gyro_y.append(gy); self.gyro_z.append(gz)
            self.ang_x.append(angx); self.ang_y.append(angy); self.ang_z.append(angz)

            maxlen = 5000
            if len(self.time_buff) > maxlen:
                for lst in (self.time_buff, self.acc_x, self.acc_y, self.acc_z,
                            self.gyro_x, self.gyro_y, self.gyro_z,
                            self.ang_x, self.ang_y, self.ang_z):
                    lst.pop(0)

            # plot with relative time
            try:
                t0 = self.time_buff[0]
                x = [tt - t0 for tt in self.time_buff]
                self.acc_curves['x'].setData(x, self.acc_x)
                self.acc_curves['y'].setData(x, self.acc_y)
                self.acc_curves['z'].setData(x, self.acc_z)
                self.gyro_curves['x'].setData(x, self.gyro_x)
                self.gyro_curves['y'].setData(x, self.gyro_y)
                self.gyro_curves['z'].setData(x, self.gyro_z)
                self.ang_curves['x'].setData(x, self.ang_x)
                self.ang_curves['y'].setData(x, self.ang_y)
                self.ang_curves['z'].setData(x, self.ang_z)
                if x:
                    self.acc_plot.setXRange(max(0, x[-1] - 5), x[-1])
                    self.gyro_plot.setXRange(max(0, x[-1] - 5), x[-1])
                    self.ang_plot.setXRange(max(0, x[-1] - 5), x[-1])
            except Exception:
                pass
        except Exception as e:
            print("Plot append error:", e)

        # update table
        try:
            display_rows = self.normalized_buffer[-self.max_table_rows:]
            self.table.setRowCount(len(display_rows))
            for i, row in enumerate(display_rows):
                for j, col in enumerate(self.ui_columns):
                    self.table.setItem(i, j, QTableWidgetItem(str(row.get(col, ""))))
            if self.table.rowCount() > 0:
                self.table.scrollToItem(self.table.item(self.table.rowCount()-1, 0))
        except Exception as e:
            print("Table update error:", e)

    def start_stream(self):
        self.normalized_buffer.clear()
        for lst in (self.time_buff, self.acc_x, self.acc_y, self.acc_z,
                    self.gyro_x, self.gyro_y, self.gyro_z,
                    self.ang_x, self.ang_y, self.ang_z):
            lst.clear()
        self._norm_state = {"last_time": None}

        if self.simulate:
            self.timer.start()
            QMessageBox.information(self, "Simulación", "BLE no disponible: usando simulación.")
            return

        mac = self.mac_input.text().strip()
        uuid = self.uuid_input.text().strip()
        if not mac or not uuid:
            QMessageBox.warning(self, "Parámetros", "Proporciona MAC o nombre del dispositivo y UUID.")
            return

        self.ble_client = AsyncBLEClient()
        self.ble_client.sample_received.connect(self._on_sample_emitted)
        self.ble_client.status.connect(self._on_status)
        self.ble_client.start(mac, uuid)
        self.timer.start()

    def stop_stream(self):
        if self.ble_client:
            self.ble_client.stop()
            self.ble_client = None
        self.timer.stop()
        QMessageBox.information(self, "Stop", "Adquisición detenida.")

    def update_plot(self):
        if self.simulate:
            t = time.time()
            samp = {
                "Acceleration X(g)": np.random.uniform(-1.2, 1.2),
                "Acceleration Y(g)": np.random.uniform(-1.2, 1.2),
                "Acceleration Z(g)": np.random.uniform(-1.2, 1.2),
                "Gyroscope X(deg/s)": np.random.uniform(-200, 200),
                "Gyroscope Y(deg/s)": np.random.uniform(-200, 200),
                "Gyroscope Z(deg/s)": np.random.uniform(-200, 200),
                "time": t
            }
            normalized = normalize_parsed(samp, self._norm_state, device_name=self.mac_input.text().strip())
            self._append_normalized(normalized)
            return

        # For real BLE, callback already updates UI. keep plots in sync if needed
        if not self.time_buff:
            return
        try:
            t0 = self.time_buff[0]
            x = [tt - t0 for tt in self.time_buff]
            self.acc_curves['x'].setData(x, self.acc_x)
            self.acc_curves['y'].setData(x, self.acc_y)
            self.acc_curves['z'].setData(x, self.acc_z)
            self.gyro_curves['x'].setData(x, self.gyro_x)
            self.gyro_curves['y'].setData(x, self.gyro_y)
            self.gyro_curves['z'].setData(x, self.gyro_z)
            self.ang_curves['x'].setData(x, self.ang_x)
            self.ang_curves['y'].setData(x, self.ang_y)
            self.ang_curves['z'].setData(x, self.ang_z)
        except Exception:
            pass

    def save_raw(self):
        if not self.normalized_buffer:
            QMessageBox.warning(self, "Guardar", "No hay datos para guardar.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"raw_ble_{ts}.csv"
        fname, _ = QFileDialog.getSaveFileName(self, "Guardar CSV crudo", default_name, "CSV Files (*.csv)")
        if not fname:
            return
        try:
            with open(fname, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writeheader()
                for row in self.normalized_buffer:
                    out = {k: row.get(k, "") for k in CSV_HEADERS}
                    writer.writerow(out)
            QMessageBox.information(self, "Guardado", f"Archivo guardado:\n{fname}")
        except Exception as e:
            QMessageBox.critical(self, "Error guardar", f"No se pudo guardar CSV:\n{e}")


# Mode C (uses ModeB.normalized_buffer)
class ModeCWidget(QWidget):
    def __init__(self, go_home_callback, source_buffer_callable):
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

        self.timer = QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._on_tick)
        self.model = None
        self.window_sec = 2.56
        self.overlap = 0.5
        self.fs_assumed = 15.0
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
        buf = self.get_source_buffer()
        if not buf or len(buf) < 3:
            return
        acc_vals = []
        time_vals = []
        for i, r in enumerate(buf):
            ax = try_float(r.get("Acceleration X(g)")) if r.get("Acceleration X(g)") != "" else np.nan
            acc_vals.append(ax)
            # time fallback: use index / fs_assumed
            time_vals.append(i / self.fs_assumed)
        acc_vals = np.array(acc_vals, dtype=float)
        if np.all(np.isnan(acc_vals)):
            return
        try:
            sampled = acc_vals[-int(self.window_sec * self.fs_assumed):]
            t_rel = np.linspace(0, len(sampled)/self.fs_assumed, num=len(sampled))
            new_t_rel, new_acc = resample_window_to_fs(np.array(t_rel), sampled, target_fs=100.0)
            self.curve.setData(new_t_rel, new_acc)
        except Exception:
            try:
                self.curve.setData(acc_vals)
            except Exception:
                pass


class HomeScreen(QWidget):
    def __init__(self, switch_callback):
        super().__init__()
        self.switch_callback = switch_callback
        self.logo = QLabel("🦶"); self.logo.setAlignment(Qt.AlignCenter); self.logo.setStyleSheet("font-size: 80px;")
        title = QLabel("Sistema de Análisis de Marcha Humana"); subtitle = QLabel("Basado en sensores IMU WT901DCL")
        for lbl in [title, subtitle]:
            lbl.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #6C63FF;")
        subtitle.setStyleSheet("font-size: 14px; color: #AAAAAA;")
        btnA = QPushButton("🅰️ Modo A - Procesar Archivo")
        btnB = QPushButton("🅱️ Modo B - Adquisición BLE")
        btnC = QPushButton("🧠 Modo C - Procesamiento en Tiempo Real")
        for btn, color, idx in zip([btnA, btnB, btnC], ["#6C63FF", "#48C9B0", "#F39C12"], [1, 2, 3]):
            btn.setStyleSheet(f"background-color: {color}; color: white; padding: 12px; font-weight: bold; border-radius: 10px;")
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.clicked.connect(lambda checked, x=idx: self.switch_callback(x))
        footer = QLabel("Desarrollado por Adrián Beltrán, Lalo Varela y Paola Montaño")
        footer.setAlignment(Qt.AlignCenter); footer.setStyleSheet("color: #666666; font-size: 12px;")
        layout = QVBoxLayout()
        layout.addWidget(self.logo); layout.addWidget(title); layout.addWidget(subtitle)
        layout.addSpacing(20); layout.addWidget(btnA); layout.addWidget(btnB); layout.addWidget(btnC)
        layout.addSpacing(20); layout.addWidget(footer); layout.setAlignment(Qt.AlignCenter); self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Análisis de Marcha - WT901DCL")
        self.resize(1280, 800)
        self.stack = QStackedWidget()
        self.home = HomeScreen(self.switch_to_mode)
        self.modeB = ModeBWidget(self.go_home)
        self.modeC = ModeCWidget(self.go_home, source_buffer_callable=lambda: self.modeB.normalized_buffer)
        self.modeA = ModeAWidget(self.go_home)
        self.stack.addWidget(self.home)   # 0
        self.stack.addWidget(self.modeA)  # 1
        self.stack.addWidget(self.modeB)  # 2
        self.stack.addWidget(self.modeC)  # 3
        self.setCentralWidget(self.stack)

    def switch_to_mode(self, mode_index):
        self.stack.setCurrentIndex(mode_index)

    def go_home(self):
        self.stack.setCurrentIndex(0)


def main():
    app = QApplication(sys.argv)
    if HAS_QDARK:
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
