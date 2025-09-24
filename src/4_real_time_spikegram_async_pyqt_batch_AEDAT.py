from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
import threading
import time
import random
from collections import deque
from pyNAVIS.loaders import Loaders
from pyNAVIS.main_settings import MainSettings
from pyNAVIS.functions import Functions

# ===== CONFIG =====
MAX_ADDRESSES = 256
WINDOW_SEC = 0.25
UPDATE_MS = 30
EVENT_RATE = 3_000_000       # events/sec
MAX_POINTS_PLOT = 30_000_000      # max points displayed
# ==================

events = deque(maxlen=MAX_POINTS_PLOT)
lock = threading.Lock()
start_time = time.time()
running = True

# ===== Generate per-address colors =====
# colors = [pg.mkBrush(pg.intColor(i, MAX_ADDRESSES, alpha=200)) for i in range(MAX_ADDRESSES)]

def event_generator():
    """Simulate high-rate events continuously."""
    batch_size = 10000
    while running:
        t_now = time.time() - start_time
        addrs = np.random.randint(0, MAX_ADDRESSES, batch_size)
        times_arr = np.random.uniform(t_now - WINDOW_SEC, t_now, batch_size)
        with lock:
            events.extend(zip(times_arr, addrs))

settings = MainSettings(num_channels=128, mono_stereo=1, address_size=4, ts_tick=0.1)
file = Loaders.loadAEDAT('data/NAS128Stereo-2025-09-24T17-16-19+0200-ForceOne-0.aedat', settings)
Functions.adapt_timestamps(file, settings)


def aedat_playback():
    """Fast continuous playback of AEDAT file in a loop with batching."""
    chunk_size = 10000       # number of events to append at once
    n_events = len(file.timestamps)

    while running:
        loop_start = time.time()
        idx = 0
        while running and idx < n_events:
            end = min(idx + chunk_size, n_events)

            # Slice a batch of timestamps/addresses
            t_batch = file.timestamps[idx:end] / 1000.0   # ms -> s
            a_batch = file.addresses[idx:end]

            # Option 1: Play as fast as possible (no sleep)
            # ------------------------------------------
            t_now = time.time() - start_time
            with lock:
                events.extend(zip(t_now + (t_batch - t_batch[0]), a_batch))

            # Optional: to approximately match real-time speed,
            # sleep for the time gap of this chunk
            gap = t_batch[-1] - t_batch[0]
            time.sleep(gap)

            idx = end
        # loop automatically restarts from beginning


# def event_generator():
#     inter_event = 1.0 / EVENT_RATE
#     while running:
#         t = time.time() - start_time
#         addr = random.randint(0, MAX_ADDRESSES - 1)
#         with lock:
#             events.append((t, addr))
#         time.sleep(inter_event)  # can batch sleep if EVENT_RATE too high

def update_gui():
    t_now = time.time() - start_time
    window_start = t_now - WINDOW_SEC

    with lock:
        filtered = [(t, a) for t, a in events if t >= window_start]
        n = len(filtered)
        if n > MAX_POINTS_PLOT:
            indices = np.random.choice(n, MAX_POINTS_PLOT, replace=False)
            filtered = [filtered[i] for i in indices]

    if filtered:
        t_arr, a_arr = zip(*filtered)
        # Assign color per address
        # brushes = [colors[a] for a in a_arr]
        scatter.setData(x=np.array(t_arr), y=np.array(a_arr))#, brush=brushes)

    else:
        scatter.setData([], [])

    plot.setXRange(max(0, window_start), t_now, padding=0)

# ===== GUI =====
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Dynamic Real-Time Scatter")
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

thread = threading.Thread(target=aedat_playback, daemon=True)
thread.start()

if __name__ == "__main__":
    app.exec()
