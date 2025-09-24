import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time
from collections import deque

# Parameters
WINDOW_SIZE = 2          # seconds to display on the X axis
UPDATE_INTERVAL = 0.5     # ms between plot updates
MAX_ADDRESSES = 64        # max unique address IDs
EVENT_RATE = 8          # probability of a new event per update

# Data buffers
times = deque()
addresses = deque()

start_time = time.time()

def generate_event():
    """
    Simulate an incoming event:
    - X: current time (seconds since start)
    - Y: random address ID
    """
    t = time.time() - start_time
    addr = random.randint(1, MAX_ADDRESSES)
    return t, addr

# Plot setup
fig, ax = plt.subplots()
scat = ax.scatter([], [], c="blue", alpha=0.7)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Address")
ax.set_title("Real-Time Address Events")
ax.set_ylim(0, MAX_ADDRESSES + 1)

def update(frame):
    # Randomly add an event
    if random.random() < EVENT_RATE:
        t, addr = generate_event()
        times.append(t)
        addresses.append(addr)

    # Remove old points outside the window
    t_now = time.time() - start_time
    while times and times[0] < t_now - WINDOW_SIZE:
        times.popleft()
        addresses.popleft()

    # Update scatter data
    scat.set_offsets(list(zip(times, addresses)))

    # Update X axis limits to create moving window
    ax.set_xlim(max(0, t_now - WINDOW_SIZE), t_now)

    return scat,

ani = animation.FuncAnimation(
    fig, update, interval=UPDATE_INTERVAL, blit=False
)

plt.tight_layout()
plt.show()
