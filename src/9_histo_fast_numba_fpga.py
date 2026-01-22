"""
Histograma en tiempo real con Numba - Datos FPGA via Okaertool
Basado en 8_histo_fast_numba.py pero usando eventos del sensor FPGA en lugar de archivo AEDAT.

CARACTERÍSTICAS:
- Captura eventos en tiempo real desde FPGA via okaertool
- Histograma acelerado con Numba (JIT compilation)
- Buffer circular eficiente para eventos
- Timestamps relativos: tick de 10ns, ajustados a tiempo real del sistema
- Ventana temporal deslizante
- PyQt5 puro con renderizado ultra-optimizado (compatible Qt 5.9.6)
"""

from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import threading
import time
from numba import njit
import sys
import os
import logging

# Agregar path de pyOKAERTool
sys.path.insert(0, 'src/pyOKAERTool')
import pyOKAERTool.main as okt

# ===== CONFIG =====
MAX_ADDRESSES = 256
WINDOW_SEC    = 0.5        # seconds to aggregate
UPDATE_MS     = 30         # GUI refresh (ms)
BUFFER_SIZE   = 3_000_000  # circular buffer size
TIMESTAMP_UNIT = 10e-9     # 10 nanoseconds per FPGA tick
BITFILE_PATH = 'src/CPNAS_okaertool.bit'
SELECTED_INPUTS = ['port_a']
INPUTS = ['port_a', 'port_b', 'port_c']
# ==================

# ===== Logging =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('FPGA_Histogram')

# ===== Circular buffer =====
timestamps = np.zeros(BUFFER_SIZE, dtype=np.float64)
addresses  = np.zeros(BUFFER_SIZE, dtype=np.int32)
write_idx  = 0
n_events   = 0
lock       = threading.Lock()
running    = True

# Variables para sincronización temporal
fpga_start_time = None      # Tiempo real del primer evento
first_fpga_timestamp = None # Primer timestamp FPGA recibido

# ===== FPGA Capture Thread =====
def fpga_capture_thread():
    """
    Captura eventos del FPGA en tiempo real usando okaertool.
    Los timestamps se convierten a tiempo real del sistema.
    """
    global write_idx, n_events, fpga_start_time, first_fpga_timestamp
    
    try:
        # Inicializar FPGA
        logger.info("Inicializando FPGA...")
        okaer = okt.Okaertool(bit_file=BITFILE_PATH)
        okaer.init()
        okaer.reset_board()
        logger.info("FPGA inicializado correctamente")
        
        # Iniciar live monitoring
        okaer.live_monitor(inputs=SELECTED_INPUTS)
        logger.info(f"Live monitoring activo en: {SELECTED_INPUTS}")
        
        batch_count = 0
        total_events = 0
        last_report_time = time.time()
        
        while running:
            spikes_data = okaer.get_live_spikes()
            current_time = time.time()
            
            # Inicializar tiempo de inicio en el primer batch
            if fpga_start_time is None:
                fpga_start_time = current_time
            
            if spikes_data is not None:
                for input_name in SELECTED_INPUTS:
                    input_index = INPUTS.index(input_name)
                    
                    if input_index < len(spikes_data):
                        spike_list = spikes_data[input_index]
                        
                        if (spike_list is not None and 
                            hasattr(spike_list, 'addresses') and 
                            hasattr(spike_list, 'timestamps')):
                            
                            raw_addresses = spike_list.addresses
                            raw_timestamps = spike_list.timestamps
                            
                            if raw_addresses is not None and len(raw_addresses) > 0:
                                # Convertir a arrays numpy
                                if not isinstance(raw_addresses, np.ndarray):
                                    raw_addresses = np.array(raw_addresses, dtype=np.uint16)
                                if not isinstance(raw_timestamps, np.ndarray):
                                    raw_timestamps = np.array(raw_timestamps, dtype=np.float64)
                                
                                # Convertir timestamps del FPGA a segundos
                                fpga_timestamps = raw_timestamps * TIMESTAMP_UNIT
                                
                                # === ESTRATEGIA DE TIMESTAMPS (basada en diagnóstico) ===
                                # Los timestamps FPGA son de un buffer circular - NO progresan entre batches
                                # Solución: tiempo real + distribución temporal relativa del FPGA
                                
                                elapsed_time = current_time - fpga_start_time
                                
                                # Extraer distribución temporal relativa dentro del batch
                                fpga_min = fpga_timestamps[0]
                                fpga_max = fpga_timestamps[-1]
                                fpga_span = fpga_max - fpga_min
                                
                                if first_fpga_timestamp is None:
                                    first_fpga_timestamp = fpga_min
                                    logger.info(f"Primera captura: span temporal = {fpga_span*1e6:.3f} μs")
                                
                                # Preservar distribución temporal del FPGA escalada a tiempo real
                                if fpga_span > 0:
                                    # Normalizar offsets del FPGA a [0, 1]
                                    relative_offsets = (fpga_timestamps - fpga_min) / fpga_span
                                    # Escalar al tiempo estimado del batch (3ms típico)
                                    dt_batch = 0.003
                                    absolute_timestamps = elapsed_time + relative_offsets * dt_batch
                                else:
                                    # Timestamps idénticos: distribuir uniformemente
                                    absolute_timestamps = elapsed_time + np.linspace(0, 0.001, len(fpga_timestamps))
                                
                                # Filtrar direcciones válidas
                                valid_mask = (raw_addresses >= 0) & (raw_addresses < MAX_ADDRESSES)
                                valid_addresses = raw_addresses[valid_mask].astype(np.int32)
                                valid_timestamps = absolute_timestamps[valid_mask]
                                
                                batch_len = len(valid_addresses)
                                
                                if batch_len > 0:
                                    # Escribir en buffer circular
                                    with lock:
                                        idxs = (write_idx + np.arange(batch_len)) % BUFFER_SIZE
                                        timestamps[idxs] = valid_timestamps
                                        addresses[idxs] = valid_addresses
                                        write_idx = (write_idx + batch_len) % BUFFER_SIZE
                                        n_events = min(n_events + batch_len, BUFFER_SIZE)
                                    
                                    batch_count += 1
                                    total_events += batch_len
                                    
                                    # Reporte periódico
                                    if current_time - last_report_time >= 2.0:
                                        event_rate = total_events / (current_time - fpga_start_time)
                                        logger.info(
                                            f"📊 Eventos: {total_events:,} | "
                                            f"Tasa: {event_rate:.0f} ev/s | "
                                            f"Batches: {batch_count} | "
                                            f"Buffer: {(n_events/BUFFER_SIZE)*100:.1f}%"
                                        )
                                        last_report_time = current_time
            
            time.sleep(0.001)  # 1ms entre polls
        
        # Limpieza
        logger.info("Deteniendo live monitoring...")
        okaer.live_monitor_stop()
        del okaer
        logger.info("FPGA desconectado")
        
    except Exception as e:
        logger.error(f"Error en thread de captura FPGA: {e}")
        import traceback
        traceback.print_exc()

# ===== Numba-accelerated histogram =====
@njit(cache=True, fastmath=True)
def fast_hist(ts, addrs, window_start, counts):
    """
    Fill `counts` (len=MAX_ADDRESSES) with the number of
    events whose timestamp >= window_start.
    Arrays are already contiguous and time-sorted.
    """
    n = ts.size
    # find first index >= window_start
    i0 = 0
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if ts[mid] < window_start:
            lo = mid + 1
        else:
            hi = mid
    i0 = lo
    # zero the histogram
    for i in range(counts.size):
        counts[i] = 0
    # accumulate counts
    for i in range(i0, n):
        a = addrs[i]
        if a < counts.size:
            counts[a] += 1

# ===== Widget ultra-optimizado con QPixmap =====
class FastHistogramWidget(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(1200, 600)
        self.setStyleSheet("background-color: black;")
        
        # Pixmap pre-allocado
        self.pixmap = QtGui.QPixmap(1200, 600)
        self.counts = np.zeros(MAX_ADDRESSES, dtype=np.int64)
        self.max_count = 10
        self.setPixmap(self.pixmap)
        
    def update_histogram(self, counts):
        """Actualizar histograma - ultra rápido con líneas verticales"""
        self.counts = counts
        self.max_count = max(10, counts.max())
        
        # Limpiar pixmap
        self.pixmap.fill(QtCore.Qt.black)
        
        # Pintar directamente al pixmap
        painter = QtGui.QPainter(self.pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        
        # Dimensiones
        w = 1200
        h = 600
        margin_left = 50
        margin_bottom = 30
        margin_top = 40
        plot_w = w - margin_left - 20
        plot_h = h - margin_top - margin_bottom
        
        bar_width = plot_w / MAX_ADDRESSES
        
        # Usar escala logarítmica si hay mucha variación
        use_log = self.max_count > 1000
        
        # Dibujar barras como líneas verticales (MUY RÁPIDO)
        pen = QtGui.QPen(QtGui.QColor(0, 180, 255))
        pen.setWidth(max(1, int(bar_width)))
        painter.setPen(pen)
        
        if self.max_count > 0:
            for i in range(MAX_ADDRESSES):
                if self.counts[i] > 0:
                    # Escala logarítmica para mejor visualización
                    if use_log:
                        bar_height = (np.log10(self.counts[i] + 1) / np.log10(self.max_count + 1)) * plot_h
                    else:
                        bar_height = (self.counts[i] / self.max_count) * plot_h
                    
                    x = int(margin_left + i * bar_width)
                    y_top = int(h - margin_bottom - bar_height)
                    y_bottom = h - margin_bottom
                    painter.drawLine(x, y_top, x, y_bottom)
        
        # Ejes
        painter.setPen(QtGui.QColor(80, 80, 80))
        painter.drawLine(margin_left, h - margin_bottom, w - 20, h - margin_bottom)
        painter.drawLine(margin_left, margin_top, margin_left, h - margin_bottom)
        
        # Título con escala
        painter.setPen(QtGui.QColor(200, 200, 200))
        painter.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        total = int(self.counts.sum())
        scale_text = " [escala LOG]" if use_log else ""
        painter.drawText(margin_left, 20, 
                        f"Eventos: {total:,} | Max: {int(self.max_count):,}{scale_text}")
        
        # Etiquetas del eje Y
        painter.setFont(QtGui.QFont("Arial", 8))
        painter.setPen(QtGui.QColor(150, 150, 150))
        num_ticks = 5
        for i in range(num_ticks + 1):
            if use_log:
                # Escala logarítmica
                frac = i / num_ticks
                val = int((10 ** (frac * np.log10(self.max_count + 1))) - 1)
            else:
                val = int((i / num_ticks) * self.max_count)
            
            y = h - margin_bottom - (i / num_ticks) * plot_h
            painter.drawText(5, int(y + 4), f"{val}")
        
        painter.end()
        
        # Actualizar display
        self.setPixmap(self.pixmap)

# ===== GUI Update =====
counts = np.zeros(MAX_ADDRESSES, dtype=np.int64)  # reused output array

def update_hist():
    """Actualizar histograma"""
    if fpga_start_time is None:
        return
    
    t_now = time.time() - fpga_start_time
    window_start = t_now - WINDOW_SEC

    with lock:
        ne = n_events
        wi = write_idx
        if ne == 0:
            counts[:] = 0
            hist_widget.update_histogram(counts)
            return

        start_idx = (wi - ne) % BUFFER_SIZE
        if start_idx < wi:
            ts_view = timestamps[start_idx:wi]
            ad_view = addresses[start_idx:wi]
        else:
            ts_view = np.concatenate((timestamps[start_idx:], timestamps[:wi]))
            ad_view = np.concatenate((addresses[start_idx:],  addresses[:wi]))

    # compute histogram *in place* with Numba
    fast_hist(ts_view, ad_view, window_start, counts)

    # update widget
    hist_widget.update_histogram(counts)

# ===== GUI Setup con PyQt5 ultra-optimizado =====
print("="*70)
print("HISTOGRAMA EN TIEMPO REAL - FPGA con Numba")
print("="*70)
print(f"Puerto FPGA: {SELECTED_INPUTS}")
print(f"Ventana temporal: {WINDOW_SEC}s")
print(f"Actualización GUI: {UPDATE_MS}ms ({1000/UPDATE_MS:.1f} FPS)")
print(f"Buffer size: {BUFFER_SIZE:,} eventos")
print(f"Timestamp unit: {TIMESTAMP_UNIT*1e9:.1f} ns")
print("="*70)
print("Iniciando captura de eventos FPGA...")
print("="*70)

app = QtWidgets.QApplication([])
win = QtWidgets.QMainWindow()
win.setWindowTitle("Histograma FPGA en Tiempo Real - Numba Accelerated")
win.resize(1200, 600)

# Widget central
hist_widget = FastHistogramWidget()
win.setCentralWidget(hist_widget)

# Timer
timer = QtCore.QTimer()
timer.timeout.connect(update_hist)
timer.start(UPDATE_MS)

def on_close():
    global running
    running = False
    logger.info("Cerrando aplicación...")

app.aboutToQuit.connect(on_close)

# ===== Start capture thread =====
capture_thread = threading.Thread(target=fpga_capture_thread, daemon=True)
capture_thread.start()

# Mostrar ventana
win.show()

if __name__ == "__main__":
    try:
        app.exec_()
    except KeyboardInterrupt:
        logger.info("Interrupción por teclado")
        running = False
