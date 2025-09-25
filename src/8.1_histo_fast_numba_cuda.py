from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
import threading
import time
from numba import njit, prange, cuda

from pyNAVIS.loaders import Loaders
from pyNAVIS.main_settings import MainSettings
from pyNAVIS.functions import Functions

# ===== CONFIG =====
MAX_ADDRESSES = 256
WINDOW_SEC = 0.5
UPDATE_MS = 30
CHUNK_SIZE = 100_000
BUFFER_SIZE = 50_000_000   # circular buffer
GPU_THRESHOLD = 2_000_000  # events needed to justify GPU
# ==================

timestamps = np.zeros(BUFFER_SIZE, dtype=np.float64)
addresses  = np.zeros(BUFFER_SIZE, dtype=np.uint16)
write_idx  = 0
n_events   = 0
lock       = threading.Lock()
running    = True

# ===== Load AEDAT file =====
settings = MainSettings(num_channels=128, mono_stereo=1,
                        address_size=4, ts_tick=0.1)
file = Loaders.loadAEDAT(
    'data/NAS128Stereo-2025-09-24T17-16-19+0200-ForceOne-0.aedat', settings)
Functions.adapt_timestamps(file, settings)
n_file_events = len(file.timestamps)

playback_start = time.time()

# ------------------ CPU FAST HIST ------------------
@njit(parallel=True)
def fast_hist_cpu(addrs, mask, max_addr):
    counts = np.zeros(max_addr, dtype=np.int32)
    for i in prange(addrs.size):
        if mask[i]:
            a = addrs[i]
            if a < max_addr:
                counts[a] += 1
    return counts

# ------------------ GPU HIST -----------------------
@cuda.jit
def fast_hist_gpu(ts, addrs, window_start, counts):
    i = cuda.grid(1)
    n = ts.size
    if i < n:
        if ts[i] >= window_start:
            a = addrs[i]
            if a < counts.size:
                cuda.atomic.add(counts, a, 1)

def gpu_histogram(ts, addrs, window_start, max_addr):
    ts_d = cuda.to_device(ts)
    addrs_d = cuda.to_device(addrs)
    counts_d = cuda.device_array(max_addr, dtype=np.int32)
    counts_d[:] = 0

    threads_per_block = 256
    blocks = (ts.size + threads_per_block - 1) // threads_per_block
    fast_hist_gpu[blocks, threads_per_block](ts_d, addrs_d,
                                             window_start, counts_d)
    return counts_d.copy_to_host()

# Check CUDA availability once
GPU_AVAILABLE = cuda.is_available()
print("CUDA available:", GPU_AVAILABLE)

# ===== Playback Thread =====
def aedat_playback():
    global write_idx, n_events
    file_start = file.timestamps[0]
    file_duration = (file.timestamps[-1] - file_start) / 1e6
    loop_count = 0
    while running:
        idx = 0
        while idx < n_file_events and running:
            end = min(idx + CHUNK_SIZE, n_file_events)
            t_batch = (file.timestamps[idx:end] - file_start) / 1e6
            t_batch += loop_count * file_duration
            t_batch = t_batch - t_batch[0] + (time.time() - playback_start)
            a_batch = file.addresses[idx:end]
            batch_len = len(t_batch)

            with lock:
                idxs = (write_idx + np.arange(batch_len)) % BUFFER_SIZE
                timestamps[idxs] = t_batch
                addresses[idxs] = a_batch
                write_idx = (write_idx + batch_len) % BUFFER_SIZE
                n_events = min(n_events + batch_len, BUFFER_SIZE)

            gap = t_batch[-1] - t_batch[0]
            if gap > 0:
                time.sleep(gap)
            idx = end
        loop_count += 1

# ===== GUI Update =====
def update_hist():
    t_now = time.time() - playback_start
    window_start = t_now - WINDOW_SEC

    with lock:
        if n_events == 0:
            counts = np.zeros(MAX_ADDRESSES, dtype=np.int32)
        else:
            idxs = (write_idx - n_events + np.arange(n_events)) % BUFFER_SIZE
            ts = timestamps[idxs]
            addrs = addresses[idxs]
            mask = ts >= window_start

            # choose CPU vs GPU
            if GPU_AVAILABLE and ts.size > GPU_THRESHOLD:
                counts = gpu_histogram(ts, addrs, window_start, MAX_ADDRESSES)
            else:
                counts = fast_hist_cpu(addrs, mask, MAX_ADDRESSES)

    bars.setOpts(height=counts)
    max_count = counts.max() if counts.size > 0 else 0
    plot.setYRange(0, max(10, max_count * 1.1))

# ===== GUI Setup =====
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Live Event Histogram")
plot = win.addPlot()
plot.setLabel('left', f'Events (last {WINDOW_SEC:.2f}s)')
plot.setLabel('bottom', 'Address')
plot.setXRange(-0.5, MAX_ADDRESSES - 0.5, padding=0)

x = np.arange(MAX_ADDRESSES)
bars = pg.BarGraphItem(x=x, height=np.zeros(MAX_ADDRESSES),
                       width=0.8, brush='dodgerblue')
plot.addItem(bars)

timer = QtCore.QTimer()
timer.timeout.connect(update_hist)
timer.start(UPDATE_MS)

def on_close():
    global running
    running = False

app.aboutToQuit.connect(on_close)

thread = threading.Thread(target=aedat_playback, daemon=True)
thread.start()

if __name__ == "__main__":
    app.exec()
