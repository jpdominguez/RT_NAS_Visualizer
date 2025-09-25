# real_time_aedat_relative.py
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
import threading
from collections import deque
import time
from pyNAVIS.loaders import Loaders
from pyNAVIS.main_settings import MainSettings
from pyNAVIS.functions import Functions

# ================= CONFIG =================
MAX_ADDRESSES = 256
WINDOW_SEC = 0.05               # moving window in seconds
UPDATE_MS = 30                  # GUI update interval
CHUNK_SIZE = 100_000            # AEDAT events per batch
MAX_POINTS_PLOT = 100_000       # max points to plot
EVENT_RATE_EST = 3_000_000      # estimated events/sec for buffer sizing
# =========================================

# ===== Moving window buffer =====
events = deque(maxlen=int(WINDOW_SEC * EVENT_RATE_EST))
lock = threading.Lock()
running = True

# ===== Load AEDAT file =====
# settings = MainSettings(num_channels=128, mono_stereo=1, address_size=2, ts_tick=1)
# file = Loaders.loadAEDAT('data/sweep_20Hz_5cyc_256ch.aedat.aedat', settings)
settings = MainSettings(num_channels=128, mono_stereo=1, address_size=4, ts_tick=1)
file = Loaders.loadAEDAT('data/NAS128Stereo-2025-09-24T17-16-19+0200-ForceOne-0.aedat', settings)
Functions.adapt_timestamps(file, settings)
n_file_events = len(file.timestamps)

# ===== Playback start reference =====
playback_start_time = time.time()

# ===== AEDAT playback thread =====
def aedat_playback():
    global playback_start_time
    file_start = file.timestamps[0]
    file_duration = (file.timestamps[-1] - file_start) / 1e6  # seconds
    loop_count = 0

    while running:
        idx = 0
        while idx < n_file_events and running:
            end = min(idx + CHUNK_SIZE, n_file_events)

            # Relative timestamps for this chunk (seconds)
            t_batch = (file.timestamps[idx:end] - file_start) / 1e6
            t_batch += loop_count * file_duration
            a_batch = file.addresses[idx:end]

            # Align to playback start (relative seconds)
            t_batch_relative = t_batch - t_batch[0] + (time.time() - playback_start_time)

            with lock:
                events.extend(zip(t_batch_relative, a_batch))

            # Sleep to simulate real-time playback
            chunk_duration = t_batch[-1] - t_batch[0]
            if chunk_duration > 0:
                time.sleep(chunk_duration)

            idx = end

        loop_count += 1  # restart file loop

# ===== GUI update =====
def update_gui():
    t_now = time.time() - playback_start_time
    window_start = t_now - WINDOW_SEC

    with lock:
        if not events:
            scatter.setData([], [])
            return

        ts, addrs = zip(*events)
        ts = np.array(ts)
        addrs = np.array(addrs)

        mask = ts >= window_start
        ts_filtered = ts[mask]
        addrs_filtered = addrs[mask]

        # Downsample for performance
        if len(ts_filtered) > MAX_POINTS_PLOT:
            idxs = np.random.choice(len(ts_filtered), MAX_POINTS_PLOT, replace=False)
            ts_filtered = ts_filtered[idxs]
            addrs_filtered = addrs_filtered[idxs]

    scatter.setData(x=ts_filtered, y=addrs_filtered)
    plot.setXRange(max(0, window_start), t_now, padding=0)  # X-axis starts at 0

# ===== GUI setup =====
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="AEDAT Real-Time Scatter (Relative X-axis)")
plot = win.addPlot()
plot.setLabel('left', 'Address')
plot.setLabel('bottom', 'Time (s)')
plot.setYRange(0, MAX_ADDRESSES)
scatter = pg.ScatterPlotItem(size=2, pen=None)
plot.addItem(scatter)

timer = QtCore.QTimer()
timer.timeout.connect(update_gui)
timer.start(UPDATE_MS)

def on_close():
    global running
    running = False

app.aboutToQuit.connect(on_close)

# ===== Start playback thread =====
thread = threading.Thread(target=aedat_playback, daemon=True)
thread.start()

if __name__ == "__main__":
    app.exec()
