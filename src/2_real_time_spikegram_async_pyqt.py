from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
import threading
import time
import random
from collections import deque

# ===== CONFIG =====
MAX_ADDRESSES = 128
WINDOW_SEC = 1.0
UPDATE_MS = 1
EVENT_RATE = 3_000_000_000_000    # simulated events/sec
MAX_POINTS_PLOT = 1_000_00_000_000   # number of points plotted (subset)
# ==================

events = deque(maxlen=MAX_POINTS_PLOT)  # store all events internally
lock = threading.Lock()
start_time = time.time()
running = True

def event_generator():
    inter_event = 1.0 / EVENT_RATE
    while running:
        t = time.time() - start_time
        addr = random.randint(0, MAX_ADDRESSES - 1)
        with lock:
            events.append((t, addr))
        time.sleep(inter_event)  # can batch sleep if EVENT_RATE too high

def update_gui():
    t_now = time.time() - start_time
    window_start = t_now - WINDOW_SEC

    with lock:
        # Copy events and filter only those in window
        filtered = [(t, a) for t, a in events if t >= window_start]
        if len(filtered) > MAX_POINTS_PLOT:
            # downsample to max points
            indices = np.random.choice(len(filtered), MAX_POINTS_PLOT, replace=False)
            filtered = [filtered[i] for i in indices]

    if filtered:
        t_arr, a_arr = zip(*filtered)
        scatter.setData(x=np.array(t_arr), y=np.array(a_arr))
    else:
        scatter.setData([], [])

    # update X axis to move forward
    plot.setXRange(max(0, window_start), t_now, padding=0)

# ===== PyQtGraph GUI =====
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Real-Time Scatter Plot")
plot = win.addPlot()
plot.setLabel('left', 'Address')
plot.setLabel('bottom', 'Time (s)')
plot.setYRange(0, MAX_ADDRESSES)
scatter = pg.ScatterPlotItem(size=3, pen=None, brush=pg.mkBrush(255, 0, 0, 255))
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
