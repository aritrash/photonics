# photonic_not_sim_gui.py
import sys
import numpy as np
from math import log10
from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QPushButton, QFrame
)
from PyQt6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Physical constant
c = 3e8  # m/s

class RingResonatorSimulator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photonic NOT Gate Simulator — Interactive (Fixed layout)")
        self.setGeometry(100, 100, 1400, 750)
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # -----------------------------
        # Left: Controls (small width)
        # -----------------------------
        control_frame = QFrame()
        control_frame.setFixedWidth(320)
        control_layout = QVBoxLayout(control_frame)
        control_layout.setSpacing(12)

        title = QLabel("<b>Controls</b>")
        control_layout.addWidget(title)

        # slider spec: (label_text, min_val, max_val, scale, default)
        # scale used to map integer slider to float: value = slider.value() / scale
        self.slider_specs = {
            "Radius (µm)": (5, 50, 1, 10),          # maps directly (no scale)
            "n_eff (approx)": (100, 400, 100, 240), # will divide by 100 -> 1.00..4.00
            "Loss (dB/cm)": (0, 3000, 1000, 100),  # divide by 1000 -> 0.000..3.000 dB/cm
            "Coupling κ": (10, 900, 1000, 200),    # divide by 1000 -> 0.010..0.900
            "λ min (nm)": (1500, 1580, 1, 1520),
            "λ max (nm)": (1521, 1620, 1, 1580),
        }

        self.sliders = {}
        self.value_labels = {}
        for key, (mn, mx, scale, default) in self.slider_specs.items():
            lbl = QLabel(f"{key}:")
            control_layout.addWidget(lbl)
            sld = QSlider(Qt.Orientation.Horizontal)
            sld.setMinimum(mn)
            sld.setMaximum(mx)
            sld.setSingleStep(1)
            # compute initial slider integer value from default and scale:
            init_val = default if scale == 1 else int(default)
            sld.setValue(init_val)
            sld.valueChanged.connect(self.on_param_change)
            control_layout.addWidget(sld)
            vlbl = QLabel(" ")  # live value display
            control_layout.addWidget(vlbl)
            self.sliders[key] = (sld, scale)
            self.value_labels[key] = vlbl

        # Add a small run/refresh button (mostly redundant since sliders auto-update)
        btn = QPushButton("Refresh")
        btn.clicked.connect(self.on_param_change)
        control_layout.addWidget(btn)

        control_layout.addStretch()
        main_layout.addWidget(control_frame)

        # -----------------------------
        # Middle: Plots (expands)
        # -----------------------------
        center_frame = QFrame()
        center_layout = QVBoxLayout(center_frame)
        center_layout.setContentsMargins(6, 6, 6, 6)

        # Matplotlib figure
        self.fig = Figure(figsize=(8, 6), constrained_layout=True)
        self.ax1 = self.fig.add_subplot(311)  # transmission
        self.ax2 = self.fig.add_subplot(312)  # simple NOT schematic
        self.ax3 = self.fig.add_subplot(313)  # switch speed comparison
        self.canvas = FigureCanvas(self.fig)
        center_layout.addWidget(self.canvas)

        main_layout.addWidget(center_frame, stretch=2)

        # -----------------------------
        # Right: Metrics (prominent)
        # -----------------------------
        metrics_frame = QFrame()
        metrics_frame.setFixedWidth(360)
        metrics_layout = QVBoxLayout(metrics_frame)
        metrics_layout.setSpacing(12)

        mtitle = QLabel("<b>Performance Metrics</b>")
        metrics_layout.addWidget(mtitle)

        # Create metric labels
        self.lbl_er = QLabel("Extinction Ratio (ER): —")
        self.lbl_er.setStyleSheet("font-size:14px;")
        metrics_layout.addWidget(self.lbl_er)

        self.lbl_contrast = QLabel("Contrast: —")
        metrics_layout.addWidget(self.lbl_contrast)

        self.lbl_il = QLabel("Insertion Loss (IL): —")
        metrics_layout.addWidget(self.lbl_il)

        self.lbl_q = QLabel("Q-factor: —")
        metrics_layout.addWidget(self.lbl_q)

        self.lbl_reslam = QLabel("Resonant λ: —")
        metrics_layout.addWidget(self.lbl_reslam)

        self.lbl_fsr = QLabel("FSR (nm): —")
        metrics_layout.addWidget(self.lbl_fsr)

        self.lbl_tau = QLabel("Switching τ: —")
        metrics_layout.addWidget(self.lbl_tau)

        self.lbl_channels = QLabel("Possible channels in band: —")
        metrics_layout.addWidget(self.lbl_channels)

        self.lbl_notes = QLabel("")
        self.lbl_notes.setWordWrap(True)
        metrics_layout.addWidget(self.lbl_notes)

        metrics_layout.addStretch()
        main_layout.addWidget(metrics_frame)

        # Initial draw
        self.on_param_change()

    # -----------------------------
    # Utility: read slider mapped value
    # -----------------------------
    def slider_value(self, name):
        sld, scale = self.sliders[name]
        val = sld.value()
        if name == "n_eff (approx)":
            return val / 100.0
        if name == "Loss (dB/cm)":
            return val / 1000.0      # dB/cm
        if name == "Coupling κ":
            return val / 1000.0
        # other sliders are integer-represented directly
        return float(val)

    # -----------------------------
    # Main compute + update
    # -----------------------------
    def on_param_change(self):
        # read parameters
        R_um = self.slider_value("Radius (µm)")
        R = R_um * 1e-6
        neff = self.slider_value("n_eff (approx)")
        loss_db_per_cm = self.slider_value("Loss (dB/cm)")
        kappa = self.slider_value("Coupling κ")
        lam_min_nm = int(self.slider_value("λ min (nm)"))
        lam_max_nm = int(self.slider_value("λ max (nm)"))
        if lam_max_nm <= lam_min_nm:
            lam_max_nm = lam_min_nm + 1

        # update value labels
        self.value_labels["Radius (µm)"].setText(f"{R_um:.2f} µm")
        self.value_labels["n_eff (approx)"].setText(f"{neff:.3f}")
        self.value_labels["Loss (dB/cm)"].setText(f"{loss_db_per_cm:.4f} dB/cm")
        self.value_labels["Coupling κ"].setText(f"{kappa:.3f}")
        self.value_labels["λ min (nm)"].setText(f"{lam_min_nm} nm")
        self.value_labels["λ max (nm)"].setText(f"{lam_max_nm} nm")

        # -----------------------------
        # Physics / transfer function
        # -----------------------------
        lam = np.linspace(lam_min_nm * 1e-9, lam_max_nm * 1e-9, 2500)
        L = 2 * np.pi * R

        # convert loss dB/cm -> dB/m
        loss_db_per_m = loss_db_per_cm * 100.0
        # amplitude round-trip factor a_rt
        # power attenuation over round-trip = 10^(-alpha_db_per_m * L / 10)
        # amplitude attenuation -> sqrt(power attenuation) = 10^(-alpha_db_per_m * L / 20)
        if loss_db_per_m <= 0:
            a_rt = 1.0
        else:
            a_rt = 10 ** ( - (loss_db_per_m * L) / 20.0 )

        # coupling amplitude t
        if kappa >= 1.0:
            kappa = 0.999
        t = np.sqrt(max(0.0, 1 - kappa**2))

        beta = 2.0 * np.pi * neff / lam
        phi = beta * L
        exp_term = a_rt * np.exp(-1j * phi)

        # Through-port amplitude (side-coupled single ring)
        E_through = (t - exp_term) / (1 - t * exp_term)
        T = np.abs(E_through)**2

        # Normalize T for plotting clarity (but keep absolute values for IL)
        T_plot = T / np.max(T) if np.max(T) > 0 else T

        # -----------------------------
        # Find resonances (local minima)
        # -----------------------------
        minima_idx = []
        N = len(T)
        for i in range(1, N - 1):
            if T[i] < T[i - 1] and T[i] <= T[i + 1]:
                minima_idx.append(i)

        # Choose deepest minimum within sweep (if any)
        note_msgs = []
        if len(minima_idx) == 0:
            # no local minima found - likely no resonance inside sweep range
            found_resonance = False
        else:
            # compute depth of minima to pick the deepest
            depths = [np.max(T) - T[i] for i in minima_idx]
            idx_min_global = minima_idx[int(np.argmax(depths))]
            found_resonance = True

        # compute T_min and T_max for metrics (over sweep)
        T_max = np.max(T)
        T_min = np.min(T)

        # if the depth is very small, we treat as "no resonance"
        depth_total = T_max - T_min
        if depth_total < 1e-6:
            found_resonance = False
            note_msgs.append("No significant resonance in sweep range. Try changing Radius / λ range / coupling.")

        # FWHM calculation (only if resonance found)
        if found_resonance:
            idx = idx_min_global
            lam_res = lam[idx]
            half_level = (T_max + T[idx]) / 2.0  # for dip, half between top and min
            # find left crossing
            left = idx
            while left > 0 and T[left] < half_level:
                left -= 1
            right = idx
            while right < (N - 1) and T[right] < half_level:
                right += 1
            # handle edge cases
            if left == 0 or right == N - 1:
                delta_lambda = 0.0
            else:
                delta_lambda = lam[right] - lam[left]

            # compute Q
            Q = lam_res / delta_lambda if delta_lambda > 0 else np.inf

            # FSR: if we found multiple minima, compute mean spacing
            if len(minima_idx) >= 2:
                lam_mins = np.array([lam[i] for i in minima_idx])
                diffs = np.diff(np.sort(lam_mins))
                FSR_nm = np.mean(diffs) * 1e9
            else:
                # fallback formula (approx using effective index as group index)
                FSR_nm = (lam_res ** 2) / (neff * L) * 1e9

            # ER (dB) and contrast%
            if T_min <= 0:
                ER_db = np.inf
                contrast_pct = 100.0
            else:
                ER_db = 10.0 * np.log10((T_max + 1e-18) / (T_min + 1e-18))
                contrast_pct = float((T_max - T_min) / (T_max + 1e-18) * 100.0)

            # IL (dB): -10*log10(T_max) (input normalized to 1)
            IL_db = -10.0 * np.log10(T_max + 1e-18) if T_max > 0 else np.inf

            # switching time tau from Q
            tau = (Q * lam_res) / (2.0 * np.pi * c)  # seconds
            tau_ps = tau * 1e12

            # estimate channels in the sweep
            sweep_nm = (lam_max_nm - lam_min_nm)
            channels = int(np.floor(sweep_nm / FSR_nm)) if FSR_nm > 1e-12 else 0

        else:
            # fill with N/A
            lam_res = None
            Q = None
            FSR_nm = None
            ER_db = None
            contrast_pct = None
            IL_db = None
            tau_ps = None
            channels = 0

        # -----------------------------
        # Update plots (clear then draw)
        # -----------------------------
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Transmission spectrum (center)
        self.ax1.plot(lam * 1e9, T_plot, lw=1.2)
        self.ax1.set_xlim(lam[0] * 1e9, lam[-1] * 1e9)
        self.ax1.set_xlabel("Wavelength (nm)")
        self.ax1.set_ylabel("Normalized Through Transmission")
        self.ax1.set_title("Transmission Spectrum (through port)")
        if found_resonance and lam_res is not None:
            self.ax1.axvline(lam_res * 1e9, color="tab:red", linestyle="--",
                              label=f"Res λ = {lam_res*1e9:.3f} nm")
            self.ax1.legend()

        # NOT gate cartoon (simple)
        self.ax2.step([0, 1], [1, 0], where='mid', lw=2)
        self.ax2.set_ylim(-0.2, 1.2)
        self.ax2.set_xticks([0, 1])
        self.ax2.set_yticks([0, 1])
        self.ax2.set_xlabel("Input bit")
        self.ax2.set_ylabel("Output bit")
        self.ax2.set_title("NOT Gate (logical)")

        # Switching speed comparison (log scale)
        photonic_speed = tau_ps if tau_ps is not None else 1e9  # if none, push it far right
        device_names = ["Photonic NOT", "7404 TTL", "CMOS ~28nm", "CMOS ~7nm"]
        # times in picoseconds
        times_ps = [photonic_speed, 1e4, 5.0, 1.0]  # 10 ns = 1e4 ps
        colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]
        bars = self.ax3.bar(device_names, times_ps, color=colors)
        self.ax3.set_yscale("log")
        self.ax3.set_ylabel("Switching time (ps) [log scale]")
        self.ax3.set_title("Switching Time Comparison")
        # annotate photonic bar with value
        if tau_ps is not None and tau_ps < 1e9:
            self.ax3.text(0, times_ps[0] * 1.2, f"{tau_ps:.2f} ps", ha='center', va='bottom')

        # finalize draw
        self.fig.suptitle("Microring-based Photonic NOT Gate Simulator", fontsize=14)
        self.canvas.draw_idle()

        # -----------------------------
        # Update metrics panel
        # -----------------------------
        if found_resonance:
            # ER and contrast
            er_text = f"{ER_db:.2f} dB" if np.isfinite(ER_db) else "∞ (very deep)"
            self.lbl_er.setText(f"<b>Extinction Ratio (ER):</b> {er_text}")
            self.lbl_contrast.setText(f"<b>Contrast:</b> {contrast_pct:.2f} %")
            # IL
            il_text = f"{IL_db:.2f} dB" if np.isfinite(IL_db) else "—"
            status_il = "Good" if (IL_db is not None and IL_db < 3.0) else "Poor"
            self.lbl_il.setText(f"<b>Insertion Loss (IL):</b> {il_text} → <i>{status_il}</i>")
            # Q
            q_text = f"{Q:.2e}" if Q is not None and np.isfinite(Q) else "—"
            self.lbl_q.setText(f"<b>Q-factor:</b> {q_text}")
            self.lbl_reslam.setText(f"<b>Resonant λ:</b> {lam_res * 1e9:.4f} nm")
            self.lbl_fsr.setText(f"<b>FSR:</b> {FSR_nm:.4f} nm")
            tau_text = f"{tau_ps:.3f} ps" if tau_ps is not None and tau_ps < 1e9 else "—"
            self.lbl_tau.setText(f"<b>Switching τ:</b> {tau_text}")
            self.lbl_channels.setText(f"<b>Channels in band:</b> {channels}")
            self.lbl_notes.setText("\n".join(note_msgs) if note_msgs else "Resonance found and metrics computed.")
            # color-code IL label
            if IL_db is not None and IL_db < 3.0:
                self.lbl_il.setStyleSheet("color:darkgreen; font-weight:600;")
            else:
                self.lbl_il.setStyleSheet("color:darkred; font-weight:600;")
        else:
            self.lbl_er.setText("<b>Extinction Ratio (ER):</b> N/A")
            self.lbl_contrast.setText("<b>Contrast:</b> N/A")
            self.lbl_il.setText("<b>Insertion Loss (IL):</b> N/A")
            self.lbl_q.setText("<b>Q-factor:</b> N/A")
            self.lbl_reslam.setText("<b>Resonant λ:</b> N/A")
            self.lbl_fsr.setText("<b>FSR:</b> N/A")
            self.lbl_tau.setText("<b>Switching τ:</b> N/A")
            self.lbl_channels.setText("<b>Channels in band:</b> 0")
            self.lbl_notes.setText("No clear resonance found in the sweep. Adjust Radius / λ-range / coupling.")
            self.lbl_il.setStyleSheet("color:black;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = RingResonatorSimulator()
    w.show()
    app.exec()
