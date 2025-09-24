import threading
import random
import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==== CONFIG ====
WINDOW_SIZE = 2               # seconds of history shown on X axis
MAX_ADDRESSES = 64              # number of possible addresses (Y values)
EVENT_INTERVAL = (0.000001, 0.000005)   # random interval between events (seconds)
UPDATE_INTERVAL_MS = 0.005         # refresh rate of the plot (milliseconds)
MAX_POINTS = 2_000_000            # safety cap for memory

# ==== SHARED DATA ====
times = deque(maxlen=MAX_POINTS)
addresses = deque(maxlen=MAX_POINTS)
lock = threading.Lock()
start_time = time.time()
running = True

def event_generator():
    """
    Background thread producing random events asynchronously.
    """
    global running
    while running:
        time.sleep(random.uniform(*EVENT_INTERVAL))
        t = time.time() - start_time
        addr = random.randint(1, MAX_ADDRESSES)
        with lock:
            times.append(t)
            addresses.append(addr)

def on_close(_):
    """
    Stop the generator thread when the window is closed.
    """
    global running
    running = False

def update(_):
    """
    Called at a fixed interval to refresh the scatter plot and
    slide the X axis based on the current clock time.
    """
    now = time.time() - start_time
    window_start = now - WINDOW_SIZE

    # Copy current data under lock
    with lock:
        if times:
            t_arr = np.array(times)
            a_arr = np.array(addresses)
            mask = t_arr >= window_start
            t_arr = t_arr[mask]
            a_arr = a_arr[mask]
        else:
            t_arr = np.empty(0)
            a_arr = np.empty(0)

    if len(t_arr):
        scat.set_offsets(np.column_stack((t_arr, a_arr)))
    else:
        # no points visible
        scat.set_offsets(np.empty((0, 2)))

    # Advance x-axis with wall-clock time (continuous movement)
    ax.set_xlim(max(0, window_start), now)
    return scat,

if __name__ == "__main__":
    # Start background generator
    thread = threading.Thread(target=event_generator, daemon=True)
    thread.start()

    # Build figure
    fig, ax = plt.subplots()
    scat = ax.scatter([], [], s=8, alpha=0.7)
    ax.set_ylim(0.5, MAX_ADDRESSES + 0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Address")
    ax.set_title("Real-Time Address Events (Continuous X-Axis)")
    fig.canvas.mpl_connect("close_event", on_close)

    # Animation updates at a *fixed* rate independent of event arrivals
    ani = animation.FuncAnimation(fig, update,
                                  interval=UPDATE_INTERVAL_MS,
                                  blit=False)
    plt.show()
