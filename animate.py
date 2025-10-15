import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

# ------------------------
# Parameters
# ------------------------
bits = "10110"         # fixed bit sequence
bit_duration = 1.0     # time units per bit
pulse_width = 0.5      # width of optical pulse
num_steps = 200        # simulation steps

# Time axis
t = np.linspace(0, len(bits) * bit_duration, num_steps)

# Input signal (laser pulses)
input_signal = np.zeros_like(t)
for i, b in enumerate(bits):
    if b == "1":
        mask = (t >= i * bit_duration) & (t < i * bit_duration + pulse_width)
        input_signal[mask] = 1.0

# Output signal (NOT operation at drop port)
output_signal = 1.0 - input_signal

# ------------------------
# Geometry: bus + ring
# ------------------------
bus_x = np.linspace(-1, 1, num_steps)
bus_y = np.zeros(num_steps)
bus_z = np.zeros(num_steps)

ring_theta = np.linspace(0, 2 * np.pi, num_steps)
ring_x = 0.5 * np.cos(ring_theta)
ring_y = 0.5 * np.sin(ring_theta)
ring_z = np.zeros(num_steps)

# ------------------------
# Frames for 3D animation
# ------------------------
frames = []
for i in range(num_steps):
    frames.append(go.Frame(
        data=[
            go.Scatter3d(x=bus_x[:i], y=bus_y[:i], z=bus_z[:i],
                         mode="lines", line=dict(color="red", width=8),
                         name="Bus (Input)"),
            go.Scatter3d(x=ring_x[:i], y=ring_y[:i], z=ring_z[:i],
                         mode="lines", line=dict(color="blue", width=5),
                         name="Ring"),
            go.Scatter3d(x=[0.0], y=[-1.0], z=[0.0],
                         mode="markers", marker=dict(size=6, color="green"),
                         name="Drop (Output)")
        ],
        name=f"frame{i}"
    ))

# ------------------------
# Combined Figure
# ------------------------
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scene"}, {"type": "xy"}]],
    subplot_titles=("3D Microring Animation", "Logic Waveforms")
)

# Base 3D scene
fig.add_trace(
    go.Scatter3d(x=bus_x, y=bus_y, z=bus_z, mode="lines",
                 line=dict(color="gray", width=2), name="Bus"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter3d(x=ring_x, y=ring_y, z=ring_z, mode="lines",
                 line=dict(color="lightblue", width=2), name="Ring"),
    row=1, col=1
)

# Logic waveform traces
fig.add_trace(go.Scatter(x=t, y=input_signal, mode="lines", name="Input (bits)"),
              row=1, col=2)
fig.add_trace(go.Scatter(x=t, y=output_signal, mode="lines", name="Output (NOT)"),
              row=1, col=2)

# Layout with animation controls
fig.update_layout(
    title="Microring NOT Gate Simulation (Multi-bit)",
    scene=dict(
        xaxis=dict(range=[-2, 2]),
        yaxis=dict(range=[-2, 2]),
        zaxis=dict(range=[-1, 1]),
    ),
    xaxis2=dict(title="Time (a.u.)"),
    yaxis2=dict(title="Logic Level", range=[-0.1, 1.1]),
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        buttons=[dict(label="Play",
                      method="animate",
                      args=[None, {"frame": {"duration": 50, "redraw": True},
                                   "fromcurrent": True, "mode": "immediate"}])]
    )],
    height=600, width=1100
)

# Attach frames (outside update_layout!)
fig.frames = frames

# Save as single HTML page
pyo.plot(fig, filename="microring_not_combined.html")
