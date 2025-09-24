from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
import threading
import time
from collections import deque
from pyNAVIS.loaders import Loaders
from pyNAVIS.main_settings import MainSettings
from pyNAVIS.functions import Functions

# ===== CONFIG =====
MAX_ADDRESSES   = 256      # number of channels
WINDOW_SEC      = 1      # activity window in seconds
UPDATE_MS       = 30       # GUI update rate (ms)
CHUNK_SIZE      = 10000    # number of events per batch
MAX_EVENTS      = 100_000
# ==================

events = deque(maxlen=MAX_EVENTS)
lock   = threading.Lock()
start_time = time.time()
running = True

# ===== Load AEDAT file =====
settings = MainSettings(num_channels=128, mono_stereo=1, address_size=4, ts_tick=1)
file = Loaders.loadAEDAT(
    'data/NAS128Stereo-2025-09-24T17-16-19+0200-ForceOne-0.aedat', settings
)
Functions.adapt_timestamps(file, settings)

# ===== Event playback (looped) =====
def aedat_playback():
    n_events = len(file.timestamps)
    while running:
        idx = 0
        while running and idx < n_events:
            end = min(idx + CHUNK_SIZE, n_events)
            t_batch = file.timestamps[idx:end] / 1000.0  # ms → s
            a_batch = file.addresses[idx:end]
            t_now   = time.time() - start_time
            with lock:
                events.extend(zip(t_now + (t_batch - t_batch[0]), a_batch))
            idx = end
        # when finished, automatically restart from the beginning

# ===== GUI =====
app  = QtWidgets.QApplication([])
win  = pg.GraphicsLayoutWidget(show=True, title="Channel Activity (Events in last %.1fs)" % WINDOW_SEC)
plot = win.addPlot()
plot.setLabel('left', 'Event Count')
plot.setLabel('bottom', 'Channel')
plot.setYRange(0, 1000)        # adjust dynamically later if needed
bars = pg.BarGraphItem(x=np.arange(MAX_ADDRESSES), height=np.zeros(MAX_ADDRESSES),
                       width=0.8, brush='b')
plot.addItem(bars)

def update_gui():
    t_now = time.time() - start_time
    window_start = t_now - WINDOW_SEC

    with lock:
        # count events per channel in the sliding window
        # convert to numpy for fast boolean filtering
        if len(events) == 0:
            counts = np.zeros(MAX_ADDRESSES, dtype=int)
        else:
            arr_t, arr_a = zip(*events)
            arr_t = np.fromiter(arr_t, dtype=float)
            arr_a = np.fromiter(arr_a, dtype=int)
            mask  = arr_t >= window_start
            counts = np.bincount(arr_a[mask], minlength=MAX_ADDRESSES)

    # update bar heights
    bars.setOpts(height=counts)

    # optional: dynamic Y range for better visibility
    plot.setYRange(0, max(10, counts.max()*1.1))

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
