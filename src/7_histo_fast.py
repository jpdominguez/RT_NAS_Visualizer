# accelerated_live_hist.py
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# --- Replace these with your actual pyNAVIS imports ---
from pyNAVIS.loaders import Loaders
from pyNAVIS.main_settings import MainSettings
from pyNAVIS.functions import Functions
# -----------------------------------------------------

# ===== CONFIG (tune to your machine / sensor) =====
MAX_ADDRESSES = 256
WINDOW_SEC = 0.5            # seconds to aggregate
UPDATE_MS = 30              # GUI refresh period (ms)
CHUNK_SIZE = 200_000        # events per batch read from file
BUFFER_SIZE = 5_000_000     # circular buffer capacity (must be >= window*rate)
WORKER_THREADS = 10          # single background worker for histogramming
# ==================================================

# ===== Circular buffers (preallocated) =====
timestamps = np.zeros(BUFFER_SIZE, dtype=np.float64)
addresses = np.zeros(BUFFER_SIZE, dtype=np.int32)  # int for bincount
write_idx = 0
n_events = 0
lock = threading.Lock()
running = True

# ===== Load AEDAT file =====
settings = MainSettings(num_channels=128, mono_stereo=1, address_size=4, ts_tick=0.1)
file = Loaders.loadAEDAT('data/NAS128Stereo-2025-09-24T17-16-19+0200-ForceOne-0.aedat', settings)
Functions.adapt_timestamps(file, settings)
n_file_events = len(file.timestamps)

# playback reference
playback_start = time.time()

# --- ThreadPoolExecutor for heavy work ---
executor = ThreadPoolExecutor(max_workers=WORKER_THREADS)
pending_future = None
pending_future_lock = threading.Lock()
latest_counts = np.zeros(MAX_ADDRESSES, dtype=np.int64)  # shared latest result (read-only for GUI)

def aedat_playback():
    """High-throughput playback into circular buffer (runs in background thread)."""
    global write_idx, n_events, running
    file_start = file.timestamps[0]
    file_duration = (file.timestamps[-1] - file_start) / 1e6
    loop_count = 0

    while running:
        idx = 0
        while idx < n_file_events and running:
            end = min(idx + CHUNK_SIZE, n_file_events)

            # relative timestamps in seconds, shifted by loop count
            t_batch = (file.timestamps[idx:end] - file_start) / 1e6
            t_batch += loop_count * file_duration

            # align to playback wall-clock (relative seconds)
            t_batch_rel = t_batch - t_batch[0] + (time.time() - playback_start)
            a_batch = file.addresses[idx:end].astype(np.int32)

            batch_len = t_batch_rel.size

            # fast vectorized write to circular buffer under lock
            with lock:
                idxs = (write_idx + np.arange(batch_len)) % BUFFER_SIZE
                timestamps[idxs] = t_batch_rel
                addresses[idxs] = a_batch
                write_idx = (write_idx + batch_len) % BUFFER_SIZE
                n_events = min(n_events + batch_len, BUFFER_SIZE)

            # sleep approximate chunk duration to emulate real-time
            chunk_duration = t_batch_rel[-1] - t_batch_rel[0]
            if chunk_duration > 0:
                # Use a small busy-wait loop to improve sleep precision if desired,
                # but here we use time.sleep to avoid CPU hogging.
                time.sleep(chunk_duration)

            idx = end

        loop_count += 1  # loop file indefinitely

# ---------- Worker function (runs in executor) ----------
def compute_counts_snapshot(snapshot_timestamps, snapshot_addresses, window_start):
    """
    Given contiguous arrays (numpy) of timestamps and addresses,
    compute counts per address for events with timestamp >= window_start.
    Returns a 1D numpy array length MAX_ADDRESSES (int64).
    """
    # timestamps are monotonically increasing; find first index >= window_start
    if snapshot_timestamps.size == 0:
        return np.zeros(MAX_ADDRESSES, dtype=np.int64)

    i0 = np.searchsorted(snapshot_timestamps, window_start, side='left')
    if i0 >= snapshot_timestamps.size:
        return np.zeros(MAX_ADDRESSES, dtype=np.int64)

    addr_window = snapshot_addresses[i0:]
    # np.bincount in C is extremely fast; ensure minlength to cover all addresses
    counts = np.bincount(addr_window, minlength=MAX_ADDRESSES).astype(np.int64)
    return counts

# ---------- Timer-driven scheduling ----------
def schedule_histogram_job():
    """If previous job finished, snapshot buffer and submit new job to executor."""
    global pending_future, latest_counts

    # If a job is already pending, skip scheduling this frame
    with pending_future_lock:
        if pending_future is not None and not pending_future.done():
            return

    # Snapshot minimal info under lock (fast)
    with lock:
        ne = n_events
        wi = write_idx
        if ne == 0:
            # no data; we can update zeros immediately
            latest_counts[:] = 0
            return

        # compute contiguous view start..end indices
        start_idx = (wi - ne) % BUFFER_SIZE
        if start_idx < wi:
            # contiguous slice
            ts_view = timestamps[start_idx:wi].copy()
            addrs_view = addresses[start_idx:wi].copy()
        else:
            # wrapped: need to copy tail then head
            # Note: two copies but still C-level memcopies
            tail = timestamps[start_idx:].copy()
            head = timestamps[:wi].copy()
            ts_view = np.concatenate((tail, head))
            tail_a = addresses[start_idx:].copy()
            head_a = addresses[:wi].copy()
            addrs_view = np.concatenate((tail_a, head_a))

    # compute window_start in same relative time base
    t_now = time.time() - playback_start
    window_start = t_now - WINDOW_SEC

    # Submit worker job
    with pending_future_lock:
        pending_future = executor.submit(compute_counts_snapshot, ts_view, addrs_view, window_start)
        # when it completes, a callback will store the result
        pending_future.add_done_callback(on_histogram_done)

def on_histogram_done(fut):
    """Callback executed in worker thread when fut completes. Copy result to latest_counts."""
    global latest_counts
    try:
        counts = fut.result()
    except Exception as e:
        # On error, don't crash GUI; print and zero counts
        print("Histogram worker error:", e)
        counts = np.zeros(MAX_ADDRESSES, dtype=np.int64)

    # Copy into shared array (atomic for numpy arrays read-only by GUI)
    latest_counts[:] = counts

# ---------- GUI update: read latest_counts and draw ----------
def gui_update_from_latest():
    """Called by Qt timer on GUI thread - just read latest_counts and update plot"""
    # read once into local array to avoid race with worker
    counts = latest_counts.copy()
    bars.setOpts(height=counts)

    # autoscale Y with headroom
    max_count = int(counts.max()) if counts.size > 0 else 0
    plot.setYRange(0, max(10, int(max_count * 1.1)))

# ---------- Startup GUI ----------
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Accelerated Live Histogram")
plot = win.addPlot()
plot.setLabel('left', 'Events (last {:.3f}s)'.format(WINDOW_SEC))
plot.setLabel('bottom', 'Address')
plot.setXRange(-0.5, MAX_ADDRESSES - 0.5, padding=0)

x = np.arange(MAX_ADDRESSES)
bars = pg.BarGraphItem(x=x, height=np.zeros(MAX_ADDRESSES),
                       width=0.8, brush='dodgerblue')
plot.addItem(bars)

# Timer 1: schedule worker job at UPDATE_MS cadence
schedule_timer = QtCore.QTimer()
schedule_timer.timeout.connect(schedule_histogram_job)
schedule_timer.start(UPDATE_MS)

# Timer 2: update GUI from latest_counts at ~UPDATE_MS (can be slightly faster)
gui_timer = QtCore.QTimer()
gui_timer.timeout.connect(gui_update_from_latest)
gui_timer.start(UPDATE_MS)  # same cadence ensures smoothness

def on_close():
    global running
    running = False
    executor.shutdown(wait=False)

app.aboutToQuit.connect(on_close)

# ---------- Start playback thread ----------
playback_thread = threading.Thread(target=aedat_playback, daemon=True)
playback_thread.start()

if __name__ == "__main__":
    app.exec()
