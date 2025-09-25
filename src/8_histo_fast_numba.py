from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
import threading
import time
from numba import njit

# --- pyNAVIS imports ---
from pyNAVIS.loaders import Loaders
from pyNAVIS.main_settings import MainSettings
from pyNAVIS.functions import Functions
# -----------------------

# ===== CONFIG =====
MAX_ADDRESSES = 256
WINDOW_SEC    = 0.5        # seconds to aggregate
UPDATE_MS     = 30         # GUI refresh (ms)
CHUNK_SIZE    = 200_000    # events per file batch
BUFFER_SIZE   = 5_000_000  # circular buffer size (must fit >= WINDOW_SEC * rate)
# ==================

# ===== Circular buffer =====
timestamps = np.zeros(BUFFER_SIZE, dtype=np.float64)
addresses  = np.zeros(BUFFER_SIZE, dtype=np.int32)   # int32 for Numba
write_idx  = 0
n_events   = 0
lock       = threading.Lock()
running    = True

# ===== Load AEDAT file =====
settings = MainSettings(num_channels=128, mono_stereo=1, address_size=4, ts_tick=0.1)
file = Loaders.loadAEDAT(
    "data/NAS128Stereo-2025-09-24T17-16-19+0200-ForceOne-0.aedat",
    settings
)
Functions.adapt_timestamps(file, settings)
n_file_events = len(file.timestamps)

playback_start = time.time()

# ===== Playback Thread =====
def aedat_playback():
    """Continuously loop through AEDAT at real speed."""
    global write_idx, n_events
    file_start   = file.timestamps[0]
    file_dur_sec = (file.timestamps[-1] - file_start) / 1e6
    loop_count   = 0

    while running:
        idx = 0
        while idx < n_file_events and running:
            end = min(idx + CHUNK_SIZE, n_file_events)
            t_batch = (file.timestamps[idx:end] - file_start) / 1e6
            t_batch += loop_count * file_dur_sec
            t_batch = t_batch - t_batch[0] + (time.time() - playback_start)
            a_batch = file.addresses[idx:end].astype(np.int32)
            batch_len = t_batch.size

            with lock:
                idxs = (write_idx + np.arange(batch_len)) % BUFFER_SIZE
                timestamps[idxs] = t_batch
                addresses[idxs]  = a_batch
                write_idx = (write_idx + batch_len) % BUFFER_SIZE
                n_events  = min(n_events + batch_len, BUFFER_SIZE)

            gap = t_batch[-1] - t_batch[0]
            if gap > 0:
                time.sleep(gap)

            idx = end
        loop_count += 1  # loop file endlessly

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

# ===== GUI Update =====
counts = np.zeros(MAX_ADDRESSES, dtype=np.int64)  # reused output array

def update_hist():
    t_now = time.time() - playback_start
    window_start = t_now - WINDOW_SEC

    with lock:
        ne = n_events
        wi = write_idx
        if ne == 0:
            counts[:] = 0
            bars.setOpts(height=counts)
            plot.setYRange(0, 10)
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

    # update plot
    bars.setOpts(height=counts)
    plot.setYRange(0, max(10, counts.max() * 1.1))

# ===== GUI Setup =====
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Numba-Accelerated Live Histogram")
plot = win.addPlot()
plot.setLabel('left',  f'Events (last {WINDOW_SEC:.2f}s)')
plot.setLabel('bottom','Address')
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
