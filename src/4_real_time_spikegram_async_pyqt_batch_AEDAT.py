from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
import threading
import time
import random
from collections import deque
from pyNAVIS.loaders import LoadAEDAT

# ===== CONFIG =====
MAX_ADDRESSES = 128
WINDOW_SEC = 3.0
UPDATE_MS = 1
EVENT_RATE = 3_000_000       # events/sec
MAX_POINTS_PLOT = 10_000      # max points displayed
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

thread = threading.Thread(target=event_generator, daemon=True)
thread.start()

if __name__ == "__main__":
    app.exec()
