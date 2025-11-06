# -------------------------------
# ARCHIVO: interfaz/gui/main_window.py
# -------------------------------
"""
Aplicación PyQt5 con tres modos:
 - Modo A: cargar CSV y procesar (offline)
 - Modo B: adquisición BLE (start/stop), visualización y guardado de crudo
 - Modo C: procesamiento en tiempo real con ventanas deslizantes (usa el modelo k-NN)

Instrucciones rápidas:
 - Colocar los archivos de este paquete en: interfaz/core/... e interfaz/gui/main_window.py
 - Asegúrate de tener instalada la librería bleak para BLE y pyqt5/pyqtgraph para la GUI
 (están en requirements.txt)
 - Ejecutar: python -m interfaz.gui.main_window 
"""
import sys
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pandas as pd
import numpy as np
from pathlib import Path
import csv
import time
from collections import deque

from interfaz.core.preprocessing import preprocess_csv
from interfaz.core.pipeline import run_pipeline_on_processed_df

# Intento de importar bleak (si no está, el modo BLE mostrará error al iniciar)
try:
    from bleak import BleakClient
    HAS_BLEAK = True
except Exception:
    HAS_BLEAK = False

# Valores BLE por defecto que proporcionaste
DEFAULT_BLE_MAC = 'eb:33:2e:f9:57:b5'
SERVICE_UUID = '0000ffe5-0000-1000-8000-00805f9a34fb'
READ_CHAR_UUID = '0000ffe4-0000-1000-8000-00805f9a34fb'
WRITE_CHAR_UUID = '0000ffe9-0000-1000-8000-00805f9a34fb'


class ModeAWidget(QtWidgets.QWidget):
    """Widget para Modo A: cargar CSV, ejecutar pipeline y mostrar features/predicciones."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.load_btn = QtWidgets.QPushButton('Cargar CSV')
        self.load_btn.clicked.connect(self.on_load)
        self.save_feat_btn = QtWidgets.QPushButton('Guardar features (.csv)')
        self.save_feat_btn.clicked.connect(self.on_save_features)
        self.table = QtWidgets.QTableWidget()
        self.pred_label = QtWidgets.QLabel('Predicciones: -')
        layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.load_btn)
        row.addWidget(self.save_feat_btn)
        layout.addLayout(row)
        layout.addWidget(self.table)
        layout.addWidget(self.pred_label)
        self.df_features = None
        self.y_pred = None
        self.y_proba = None

    def on_load(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Selecciona CSV', filter='CSV Files (*.csv)')
        if not fname:
            return
        df_proc, meta = preprocess_csv(fname, output_path=None, target_fs=None, apply_filters=True)
        try:
            df_feat, indices, y_pred, y_proba = run_pipeline_on_processed_df(df_proc)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Error ejecutando pipeline: {e}')
            return
        self.df_features = df_feat
        self.y_pred = y_pred
        self.y_proba = y_proba
        self._show_table(df_feat)
        self.pred_label.setText('Predicciones ventanas: ' + str(list(y_pred)))

    def _show_table(self, df: pd.DataFrame):
        self.table.clear()
        self.table.setColumnCount(len(df.columns))
        self.table.setRowCount(len(df))
        self.table.setHorizontalHeaderLabels(list(df.columns))
        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                val = df.iloc[i, j]
                it = QtWidgets.QTableWidgetItem(str(np.round(val, 6) if pd.notna(val) else 'nan'))
                self.table.setItem(i, j, it)
        self.table.resizeColumnsToContents()

    def on_save_features(self):
        if self.df_features is None:
            QtWidgets.QMessageBox.warning(self, 'Aviso', 'No hay features para guardar')
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Guardar features', filter='CSV Files (*.csv)')
        if not fname:
            return
        self.df_features.to_csv(fname, index=False)
        QtWidgets.QMessageBox.information(self, 'Guardado', f'Features guardadas en {fname}')


class BLEWorker(QtCore.QThread):
    """Hilo que gestiona la conexión BLE y emite muestras parseadas.

    - Emite `sample_signal` con un diccionario que debe contener al menos:
      {'Acceleration X(g)', 'Acceleration Y(g)', 'Acceleration Z(g)',
       'Gyroscope X(deg/s)', 'Gyroscope Y(deg/s)', 'Gyroscope Z(deg/s)', 'time'}
    - Si en el repo existe una función parse_notify la intentará usar.
    """
    sample_signal = QtCore.pyqtSignal(dict)
    status_signal = QtCore.pyqtSignal(str)

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
            self.status_signal.emit('Bleak no disponible (instala bleak)')
            return
        self._running = True
        import asyncio
        asyncio.run(self._run_async())

    async def _run_async(self):
        client = BleakClient(self.address)
        try:
            await client.connect()
            self.status_signal.emit('Conectado')

            def callback(sender, data: bytearray):
                sample = None
                # intentar parser del repo
                if self._repo_parser is not None:
                    try:
                        sample = self._repo_parser(data)
                    except Exception:
                        sample = None
                if sample is None:
                    # intentar parse ASCII CSV "ax,ay,az,gx,gy,gz"
                    try:
                        s = data.decode('utf-8').strip()
                        parts = s.split(',')
                        if len(parts) >= 6:
                            ax, ay, az, gx, gy, gz = map(float, parts[:6])
                            sample = {
                                'Acceleration X(g)': ax, 'Acceleration Y(g)': ay, 'Acceleration Z(g)': az,
                                'Gyroscope X(deg/s)': gx, 'Gyroscope Y(deg/s)': gy, 'Gyroscope Z(deg/s)': gz,
                                'time': time.time()
                            }
                    except Exception:
                        sample = None
                if sample is not None:
                    self.sample_signal.emit(sample)

            await client.start_notify(self.notify_uuid, callback)
            while self._running:
                await asyncio.sleep(0.1)
            await client.stop_notify(self.notify_uuid)
            await client.disconnect()
            self.status_signal.emit('Desconectado')
        except Exception as e:
            self.status_signal.emit(f'Error BLE: {e}')

    def stop(self):
        self._running = False


class ModeBWidget(QtWidgets.QWidget):
    """Widget para adquisición BLE: start/stop, visualización y guardado del crudo."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.addr_edit = QtWidgets.QLineEdit()
        self.addr_edit.setPlaceholderText('Dirección MAC (p.ej. eb:33:2e:f9:57:b5)')
        self.addr_edit.setText(DEFAULT_BLE_MAC)
        self.uuid_edit = QtWidgets.QLineEdit()
        self.uuid_edit.setPlaceholderText('Notify UUID (p.ej. 0000ffe4-...)')
        self.uuid_edit.setText(READ_CHAR_UUID)
        self.start_btn = QtWidgets.QPushButton('Start')
        self.stop_btn = QtWidgets.QPushButton('Stop')
        self.save_btn = QtWidgets.QPushButton('Guardar crudo')
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.addLegend()
        self.acc_plot = self.plot_widget.plot(pen='r', name='Acc X')
        self.gyr_plot = self.plot_widget.plot(pen='b', name='Gyr X')
        layout = QtWidgets.QVBoxLayout(self)
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.addr_edit); hl.addWidget(self.uuid_edit); hl.addWidget(self.start_btn); hl.addWidget(self.stop_btn); hl.addWidget(self.save_btn)
        layout.addLayout(hl)
        layout.addWidget(self.plot_widget)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.save_btn.clicked.connect(self.save_raw)
        self.worker = None
        self.buffer = []  # lista de diccionarios con muestras
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.setInterval(200)
        self.plot_timer.timeout.connect(self.update_plot)

    def start(self):
        if not HAS_BLEAK:
            QtWidgets.QMessageBox.critical(self, 'BLE', 'Bleak no está instalado o no disponible')
            return
        addr = self.addr_edit.text().strip()
        uuid = self.uuid_edit.text().strip()
        if not addr or not uuid:
            QtWidgets.QMessageBox.warning(self, 'Parámetros', 'Proporciona dirección MAC y UUID')
            return
        self.buffer = []
        self.worker = BLEWorker(addr, uuid)
        self.worker.sample_signal.connect(self.on_sample)
        self.worker.status_signal.connect(lambda s: print('[BLE]', s))
        self.worker.start()
        self.plot_timer.start()

    def stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(timeout=2000)
            self.worker = None
        self.plot_timer.stop()
        QtWidgets.QMessageBox.information(self, 'Stop', 'Adquisición detenida. Guarda el crudo si deseas procesarlo.')

    def on_sample(self, sample: dict):
        if 'time' not in sample:
            sample['time'] = time.time()
        self.buffer.append(sample)

    def update_plot(self):
        if not self.buffer:
            return
        N = 500
        buf = self.buffer[-N:]
        times = np.array([s['time'] for s in buf])
        ax = np.array([s.get('Acceleration X(g)', np.nan) for s in buf])
        gx = np.array([s.get('Gyroscope X(deg/s)', np.nan) for s in buf])
        if len(times) >= 2:
            self.acc_plot.setData(times - times[0], ax)
            self.gyr_plot.setData(times - times[0], gx)

    def save_raw(self):
        if not self.buffer:
            QtWidgets.QMessageBox.warning(self, 'Guardar', 'No hay datos para guardar')
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Guardar CSV crudo', filter='CSV Files (*.csv)')
        if not fname:
            return
        keys = list(self.buffer[0].keys())
        with open(fname, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.buffer:
                writer.writerow(row)
        QtWidgets.QMessageBox.information(self, 'Guardado', f'Archivo guardado en {fname}')
        res = QtWidgets.QMessageBox.question(self, 'Procesar ahora', '¿Procesar el bloque grabado ahora con el pipeline?')
        if res == QtWidgets.QMessageBox.Yes:
            df_proc, meta = preprocess_csv(fname, output_path=None)
            try:
                df_feat, indices, y_pred, y_proba = run_pipeline_on_processed_df(df_proc)
                QtWidgets.QMessageBox.information(self, 'Resultado', f'Predicciones: {list(y_pred)}')
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'Error pipeline', str(e))


class ModeCWidget(QtWidgets.QWidget):
    """Widget para procesamiento en tiempo real con ventanas deslizantes.

    - Añade muestras con add_sample(sample_dict)
    - Usa el modelo en models/k-NN/kNN_8_caracteristicas.joblib por defecto
    - Las recomendaciones para ventana/overlap se eligen automáticamente según fs estimada
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QtWidgets.QLabel('Clase actual: -')
        self.conf_label = QtWidgets.QLabel('Confianza: -')
        self.start_btn = QtWidgets.QPushButton('Start realtime')
        self.stop_btn = QtWidgets.QPushButton('Stop realtime')
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.conf_label)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        self.buffer = deque()
        self.running = False
        self.timer = QtCore.QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._on_tick)
        self.fs = 15.0  # valor asumido por defecto (puedes ajustarlo si sabes la fs real)
        self.model_path = Path('models/k-NN/kNN_8_caracteristicas.joblib')
        self.model = None
        self.window_sec = None
        self.overlap = None
        self.window_len = None
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

    def load_model(self):
        import joblib
        if not self.model_path.exists():
            QtWidgets.QMessageBox.critical(self, 'Modelo', f'Modelo no encontrado en {self.model_path}')
            return False
        self.model = joblib.load(self.model_path)
        return True

    def start(self):
        ok = self.load_model()
        if not ok:
            return
        self.running = True
        # elegir parámetros de ventana según fs
        wsec, ov = choose_window_params(self.fs)
        self.window_sec = wsec
        self.overlap = ov
        self.window_len = max(1, int(round(self.window_sec * self.fs)))
        self.timer.start()
        QtWidgets.QMessageBox.information(self, 'Realtime', f'Start realtime (fs={self.fs}Hz, ventana={self.window_sec}s, overlap={self.overlap})')

    def stop(self):
        self.running = False
        self.timer.stop()

    def add_sample(self, sample: dict):
        self.buffer.append(sample)
        maxlen = int(max(5*self.window_len, 1000))
        while len(self.buffer) > maxlen:
            self.buffer.popleft()

    def _on_tick(self):
        if not self.running:
            return
        if self.window_len is None or len(self.buffer) < self.window_len:
            return
        window = list(self.buffer)[-self.window_len:]
        import pandas as pd
        wdf = pd.DataFrame(window)
        try:
            from interfaz.core.top8 import window_df_to_top8
            feats = window_df_to_top8(wdf)
        except Exception as e:
            print('Error calculando features realtime', e)
            return
        X = feats.values.reshape(1, -1)
        try:
            pred = self.model.predict(X)[0]
            conf = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    proba = self.model.predict_proba(X)
                    conf = float(np.max(proba))
                except Exception:
                    conf = None
            self.label.setText(f'Clase actual: {pred}')
            self.conf_label.setText(f'Confianza: {conf:.3f}' if conf is not None else 'Confianza: -')
        except Exception as e:
            print('Error predicción realtime', e)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Interfaz AVD - IMU')
        tabs = QtWidgets.QTabWidget()
        self.mode_a = ModeAWidget()
        self.mode_b = ModeBWidget()
        self.mode_c = ModeCWidget()
        tabs.addTab(self.mode_a, 'Modo A - CSV')
        tabs.addTab(self.mode_b, 'Modo B - Adquisición BLE')
        tabs.addTab(self.mode_c, 'Modo C - Realtime')
        self.setCentralWidget(tabs)
        # temporizador para pasar muestras del buffer B al C
        self.timer_connect = QtCore.QTimer()
        self.timer_connect.setInterval(200)
        self.timer_connect.timeout.connect(self._connect_buffers)
        self.timer_connect.start()

    def _connect_buffers(self):
        # Si mode_b tiene buffer y mode_c existe, alimentar muestras nuevas
        if hasattr(self.mode_b, 'buffer') and hasattr(self.mode_c, 'add_sample'):
            buf = self.mode_b.buffer
            if buf:
                for s in buf[-200:]:
                    self.mode_c.add_sample(s)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
