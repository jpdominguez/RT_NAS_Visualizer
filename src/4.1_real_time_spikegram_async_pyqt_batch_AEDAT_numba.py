from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
import threading
import time
from numba import njit
from pyNAVIS.loaders import Loaders
from pyNAVIS.main_settings import MainSettings
from pyNAVIS.functions import Functions

# ================= CONFIG =================
MAX_ADDRESSES    = 256
WINDOW_SEC       = 0.05
UPDATE_MS        = 30
CHUNK_SIZE       = 100_000
MAX_POINTS_PLOT  = 100_000
TIME_BINS        = 256
BUFFER_SIZE      = int(WINDOW_SEC * 3_000_000 * 2)
# ==========================================

# ===== Ring buffer =====
timestamps = np.empty(BUFFER_SIZE, dtype=np.float64)
addresses  = np.empty(BUFFER_SIZE, dtype=np.uint16)
write_idx  = 0
n_events   = 0
lock       = threading.Lock()
running    = True

# ===== Load AEDAT file =====
settings = MainSettings(num_channels=128, mono_stereo=1, address_size=4, ts_tick=1)
file = Loaders.loadAEDAT('data/NAS128Stereo-2025-09-24T17-16-19+0200-ForceOne-0.aedat', settings)
Functions.adapt_timestamps(file, settings)
n_file_events = len(file.timestamps)
playback_start_time = time.time()

# ===== Numba helpers =====
@njit
def filter_window(ts, addrs, n_events, write_idx, buffer_size, window_start, window_end):
    out_ts = np.empty(n_events, dtype=np.float64)
    out_ad = np.empty(n_events, dtype=np.uint16)
    count = 0
    for i in range(n_events):
        idx = (write_idx - 1 - i) % buffer_size
        t = ts[idx]
        if t < window_start:
            break
        if t <= window_end:
            out_ts[count] = t
            out_ad[count] = addrs[idx]
            count += 1
    return out_ts[count-1::-1], out_ad[count-1::-1]

@njit
def downsample_random(ts, ad, max_points):
    n = ts.size
    if n <= max_points:
        return ts, ad
    idxs = np.random.randint(0, n, max_points)
    return ts[idxs], ad[idxs]

@njit
def bin_spikegram(ts, ad, window_start, window_end, max_addr, time_bins):
    spike_image = np.zeros((max_addr, time_bins), dtype=np.uint8)
    if ts.size == 0:
        return spike_image
    bins = np.linspace(window_start, window_end, time_bins + 1)
    digitized = np.empty(ts.size, dtype=np.int32)
    for i in range(ts.size):
        for b in range(time_bins):
            if bins[b] <= ts[i] < bins[b + 1]:
                digitized[i] = b
                break
        else:
            digitized[i] = time_bins - 1
    for i in range(ts.size):
        a = ad[i]
        if 0 <= a < max_addr:
            spike_image[a, digitized[i]] += 1
    return spike_image

# ===== AEDAT playback thread =====
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
            t_batch = t_batch - t_batch[0] + (time.time() - playback_start_time)
            a_batch = file.addresses[idx:end]
            batch_len = len(t_batch)
            with lock:
                idxs = (write_idx + np.arange(batch_len)) % BUFFER_SIZE
                timestamps[idxs] = t_batch
                addresses[idxs]  = a_batch
                write_idx        = (write_idx + batch_len) % BUFFER_SIZE
                n_events         = min(n_events + batch_len, BUFFER_SIZE)
            gap = t_batch[-1] - t_batch[0]
            if gap > 0:
                time.sleep(gap)
            idx = end
        loop_count += 1

# ===== GUI update =====
def update_spikegram():
    t_now = time.time() - playback_start_time
    window_start = t_now - WINDOW_SEC
    with lock:
        if n_events == 0:
            spikegram_img.setImage(np.zeros((MAX_ADDRESSES, TIME_BINS), dtype=np.uint8))
            return
        ts_win, ad_win = filter_window(
            timestamps, addresses, n_events,
            write_idx, BUFFER_SIZE, window_start, t_now
        )
    ts_ds, ad_ds = downsample_random(ts_win, ad_win, MAX_POINTS_PLOT)
    spike_image = bin_spikegram(ts_ds, ad_ds, window_start, t_now, MAX_ADDRESSES, TIME_BINS)
    spike_image = np.clip(spike_image * 50, 0, 255)

    # === 🔄 Rotate image 90° to the left ===
    spike_image = np.rot90(spike_image, k=1)

    spikegram_img.setImage(spike_image, autoLevels=False)

# ===== GUI setup =====
app = QtWidgets.QApplication([])
win = QtWidgets.QWidget()
layout = QtWidgets.QVBoxLayout()
win.setLayout(layout)
win.setWindowTitle("2D Spikegram Real-Time Viewer (Rotated 90° Left)")

plot_widget = pg.PlotWidget()
plot_widget.setLabel('left', 'Time (rotated)')
plot_widget.setLabel('bottom', 'Address (rotated)')
plot_widget.setYRange(0, TIME_BINS)
plot_widget.setXRange(0, MAX_ADDRESSES)
layout.addWidget(plot_widget)

spikegram_img = pg.ImageItem()
plot_widget.addItem(spikegram_img)

timer = QtCore.QTimer()
timer.timeout.connect(update_spikegram)
timer.start(UPDATE_MS)

def on_close():
    global running
    running = False
app.aboutToQuit.connect(on_close)

thread = threading.Thread(target=aedat_playback, daemon=True)
thread.start()

win.show()
if __name__ == "__main__":
    app.exec()
