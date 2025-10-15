# microring_gui_robustness.py
import sys
import numpy as np
import math
import pandas as pd
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QFormLayout, QHBoxLayout, QSplitter, QMessageBox, QFrame
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------------------------
# Physics helpers
# ---------------------------
c = 3e8

def through_drop_spectrum(lam, R_um, n_eff, kappa, loss_db_per_cm):
    """Return (T_through, T_drop) arrays for provided wavelength vector lam (nm).
       R_um is ring radius in microns.
       This uses a simple analytic coupled-mode inspired model.
    """
    R_m = R_um * 1e-6
    L = 2.0 * math.pi * R_m
    # convert loss dB/cm -> amplitude round-trip
    loss_db_per_m = loss_db_per_cm * 100.0
    if loss_db_per_m <= 0:
        a_rt = 1.0
    else:
        a_rt = 10 ** ( - (loss_db_per_m * L) / 20.0 )

    # protect kappa bounds
    kappa = float(max(min(kappa, 0.9999), 1e-9))
    t = math.sqrt(max(0.0, 1.0 - kappa**2))

    lam_m = lam * 1e-9
    beta = 2.0 * math.pi * n_eff / lam_m
    phi = beta * L
    exp_term = a_rt * np.exp(-1j * phi)

    E_th = (t - exp_term) / (1.0 - t * exp_term)
    T_th = np.abs(E_th)**2

    E_drop = (1j * kappa * np.sqrt(a_rt) * np.exp(-1j * phi/2.0)) / (1.0 - t * exp_term)
    T_drop = np.abs(E_drop)**2

    return T_th, T_drop

def analyze_spectrum(lam, T, R_um, n_eff):
    """Extract metrics from spectrum T (lam in nm)."""
    Tmax = float(np.max(T))
    Tmin = float(np.min(T))
    ER_db = 10.0 * np.log10((Tmax + 1e-18) / (Tmin + 1e-18)) if Tmin > 0 else np.inf
    contrast_pct = float((Tmax - Tmin) / (Tmax + 1e-18) * 100.0) if Tmax > 0 else 0.0

    idx_min = int(np.argmin(T))
    lam_res = float(lam[idx_min])

    half = (Tmax + Tmin) / 2.0
    indices = np.where(T < half)[0]
    if len(indices) > 1:
        delta_lambda = float(lam[indices[-1]] - lam[indices[0]])
        Q = lam_res / delta_lambda if delta_lambda > 0 else np.inf
    else:
        delta_lambda = np.nan
        Q = np.inf

    R_m = R_um * 1e-6
    L = 2.0 * math.pi * R_m
    FSR_nm = (lam_res**2) / (n_eff * L) * 1e9

    tau_ps = (Q * lam_res * 1e-9) / (2.0 * math.pi * c) * 1e12 if np.isfinite(Q) else np.nan

    return {
        'Tmax': Tmax, 'Tmin': Tmin, 'ER_db': ER_db, 'contrast_pct': contrast_pct,
        'lam_res_nm': lam_res, 'delta_lambda_nm': delta_lambda, 'Q': Q,
        'FSR_nm': FSR_nm, 'tau_ps': tau_ps
    }

# ---------------------------
# Matplotlib canvas wrapper
# ---------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, figsize=(10, 7)):
        self.fig = Figure(figsize=figsize)
        super().__init__(self.fig)

# ---------------------------
# Main GUI
# ---------------------------
class RingSimulatorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microring Simulator — with Robustness Sweep")
        self.setGeometry(80, 80, 1400, 840)
        self._build_ui()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter()

        # Left control panel
        left = QFrame()
        left.setMinimumWidth(340)
        form = QFormLayout()

        # device params
        self.radius_input = QLineEdit("2.428571")   # um
        self.neff_input = QLineEdit("2.4")
        self.kappa_input = QLineEdit("0.6")
        self.loss_input = QLineEdit("0.1")         # dB/cm
        self.lam_min_input = QLineEdit("1520")
        self.lam_max_input = QLineEdit("1580")
        self.N_input = QLineEdit("4000")

        form.addRow(QLabel("<b>Device parameters</b>"))
        form.addRow(QLabel("Radius (µm):"), self.radius_input)
        form.addRow(QLabel("n_eff:"), self.neff_input)
        form.addRow(QLabel("Coupling κ:"), self.kappa_input)
        form.addRow(QLabel("Loss (dB/cm):"), self.loss_input)
        form.addRow(QLabel("λ min (nm):"), self.lam_min_input)
        form.addRow(QLabel("λ max (nm):"), self.lam_max_input)
        form.addRow(QLabel("Samples N:"), self.N_input)

        # Simulate button
        self.sim_btn = QPushButton("Simulate")
        self.sim_btn.clicked.connect(self.on_simulate)
        form.addRow(self.sim_btn)

        # Results label
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)
        form.addRow(QLabel("<b>Results (drop-port):</b>"))
        form.addRow(self.results_label)

        # Robustness controls
        form.addRow(QLabel("<b>Robustness sweep</b>"))
        self.detune_max_input = QLineEdit("0.2")   # nm
        self.detune_points_input = QLineEdit("81")
        self.bits_per_detune_input = QLineEdit("500")  # bits used to estimate BER
        self.noise_rms_input = QLineEdit("0.001")  # detector noise RMS
        form.addRow(QLabel("Max detuning (± nm):"), self.detune_max_input)
        form.addRow(QLabel("Detuning points:"), self.detune_points_input)
        form.addRow(QLabel("Bits per detuning:"), self.bits_per_detune_input)
        form.addRow(QLabel("Detector noise RMS (a.u.):"), self.noise_rms_input)

        self.robust_btn = QPushButton("Run Robustness Sweep")
        self.robust_btn.clicked.connect(self.on_run_robustness)
        form.addRow(self.robust_btn)

        # Help note
        help_lbl = QLabel("Note: robustness sim locks laser to nominal resonance and\n"
                          "simulates resonance shift (detuning) + detector noise.\nLarge sweeps may take time.")
        help_lbl.setWordWrap(True)
        form.addRow(help_lbl)

        left.setLayout(form)
        splitter.addWidget(left)

        # Right: Plot canvas
        right = QFrame()
        right_layout = QVBoxLayout()
        self.canvas = MplCanvas(figsize=(11,8))
        right_layout.addWidget(self.canvas)
        right.setLayout(right_layout)
        splitter.addWidget(right)

        splitter.setSizes([350, 1024])
        main_layout.addWidget(splitter)

    # -----------------------
    # Single-run simulation (update embedded plots)
    # -----------------------
    def on_simulate(self):
        try:
            R = float(self.radius_input.text())
            neff = float(self.neff_input.text())
            kappa = float(self.kappa_input.text())
            loss = float(self.loss_input.text())
            lam_min = float(self.lam_min_input.text())
            lam_max = float(self.lam_max_input.text())
            N = int(self.N_input.text())
        except Exception as e:
            QMessageBox.critical(self, "Input error", f"Invalid input: {e}")
            return

        lam = np.linspace(lam_min, lam_max, N)
        T_th, T_drop = through_drop_spectrum(lam, R, neff, kappa, loss)
        m_th = analyze_spectrum(lam, T_th, R, neff)
        m_drop = analyze_spectrum(lam, T_drop, R, neff)

        # update canvas with 2x2 layout (spectrum, logic, time compare, zoom)
        fig = self.canvas.fig
        fig.clf()
        axs = fig.subplots(2, 2)

        ax0 = axs[0,0]
        ax0.plot(lam, T_th, label='Through')
        ax0.plot(lam, T_drop, label='Drop')
        ax0.set_xlabel("Wavelength (nm)")
        ax0.set_ylabel("Transmission")
        ax0.set_title("Transmission Spectrum")
        ax0.legend()
        ax0.grid(True)

        # simple logic cartoon
        ax1 = axs[0,1]
        bit_seq = np.random.randint(0,2,50)
        ax1.step(range(len(bit_seq)), bit_seq, where='mid', label='Input')
        ax1.step(range(len(bit_seq)), 1-bit_seq, where='mid', label='NOT Output')
        ax1.set_ylim(-0.2,1.2)
        ax1.set_xlabel("Bit index")
        ax1.set_title("Logic (idealized)")
        ax1.legend()
        ax1.grid(True)

        # switching time comparison
        ax2 = axs[1,0]
        labels = ["7404 TTL","28nm CMOS","7nm CMOS","Photonic (drop)"]
        times = [1e4, 10.0, 2.0, m_drop['tau_ps'] if not math.isnan(m_drop['tau_ps']) else 1e9]
        ax2.bar(labels, times, color=['red','green','orange','blue'])
        ax2.set_yscale('log'); ax2.set_ylabel("Switching time (ps, log)")
        ax2.set_title("Switching Comparison")
        ax2.grid(True, which='both', ls='--', alpha=0.3)

        # zoom on drop resonance
        ax3 = axs[1,1]
        idx_res = int(np.argmin(T_drop))
        lam_res = lam[idx_res]
        mask = (lam > lam_res - 1) & (lam < lam_res + 1)
        ax3.plot(lam[mask], T_drop[mask])
        ax3.set_xlabel("Wavelength (nm)"); ax3.set_title("Drop resonance (zoom)")
        ax3.grid(True)

        fig.tight_layout()
        self.canvas.draw()

        # results label
        res_text = (f"ER = {m_drop['ER_db']:.2f} dB    Contrast = {m_drop['contrast_pct']:.2f}%\n"
                    f"Q = {m_drop['Q']:.2e}    τ = {m_drop['tau_ps']:.3f} ps\n"
                    f"Res λ = {m_drop['lam_res_nm']:.4f} nm    FSR = {m_drop['FSR_nm']:.3f} nm")
        self.results_label.setText(res_text)

    # -----------------------
    # Robustness sweep
    # -----------------------
    def on_run_robustness(self):
        try:
            R = float(self.radius_input.text())
            neff = float(self.neff_input.text())
            kappa = float(self.kappa_input.text())
            loss = float(self.loss_input.text())
            lam_min = float(self.lam_min_input.text())
            lam_max = float(self.lam_max_input.text())
            N = int(self.N_input.text())

            detune_max = float(self.detune_max_input.text())   # nm
            n_points = int(self.detune_points_input.text())
            bits_per = int(self.bits_per_detune_input.text())
            noise_rms = float(self.noise_rms_input.text())
        except Exception as e:
            QMessageBox.critical(self, "Input error", f"Invalid input: {e}")
            return

        if n_points < 3 or bits_per < 10:
            QMessageBox.warning(self, "Parameters", "Choose at least 3 detuning points and >=10 bits per point.")
            return

        # Compute baseline spectrum
        lam = np.linspace(lam_min, lam_max, N)
        T_th, T_drop = through_drop_spectrum(lam, R, neff, kappa, loss)
        idx_res = int(np.argmin(T_th))
        lam_res_nominal = float(lam[idx_res])
        # Laser locked to nominal resonance wavelength
        lam_signal = lam_res_nominal

        # Build detuning sweep (nm)
        detunes = np.linspace(-detune_max, detune_max, n_points)

        # Prepare results arrays
        ber_list = np.zeros_like(detunes, dtype=float)
        on_power_list = np.zeros_like(detunes, dtype=float)
        contrast_list = np.zeros_like(detunes, dtype=float)
        snr_list = np.zeros_like(detunes, dtype=float)

        # For interpolation
        lam_arr = lam
        Tdrop_arr = T_drop

        # Bits sequence per detuning (we keep same random seed for reproducibility)
        rng = np.random.default_rng(12345)
        for i, d in enumerate(detunes):
            # If the ring resonance shifts by +d, the effect at fixed laser wavelength lam_signal
            # equals baseline spectrum evaluated at lam_signal - d (nm)
            sample_wavelength = lam_signal - d
            # Interpolate sample value on Tdrop spectrum
            if sample_wavelength < lam_arr[0] or sample_wavelength > lam_arr[-1]:
                # out of computed spectral range: set small value
                T_on = 0.0
            else:
                T_on = float(np.interp(sample_wavelength, lam_arr, Tdrop_arr))
            # Off level = background (small), we approximate as 0 (or small background)
            T_off = 0.0

            # Save on-power
            on_power_list[i] = T_on
            # contrast (on relative to off-background)
            contrast_list[i] = (T_on - T_off) / (T_on + 1e-18) * 100.0 if T_on > 0 else 0.0
            # SNR estimate: on_power / noise_rms
            snr_list[i] = (T_on / noise_rms) if noise_rms > 0 else np.inf

            # Generate random bits and simulate detection
            bits = rng.integers(0,2, size=bits_per)
            # Model: input power = 1 * T_on for bits==1, background ~0 for bits==0. Add Gaussian noise
            noise = rng.normal(loc=0.0, scale=noise_rms, size=bits_per)
            detected_power = np.where(bits==1, T_on, T_off) + noise
            # Decision threshold (midpoint of mean_on/mean_off)
            mean_on = np.mean(detected_power[bits==1]) if np.any(bits==1) else 0.0
            mean_off = np.mean(detected_power[bits==0]) if np.any(bits==0) else 0.0
            thresh = 0.5 * (mean_on + mean_off)
            detected_bit = (detected_power >= thresh).astype(int)
            # Photonic NOT = invert detected input
            photonic_not = 1 - detected_bit
            expected_not = 1 - bits
            errors = np.sum(photonic_not != expected_not)
            ber = errors / bits_per
            ber_list[i] = ber

        # Save CSV
        df = pd.DataFrame({
            'detune_nm': detunes,
            'T_on': on_power_list,
            'contrast_pct': contrast_list,
            'SNR_est': snr_list,
            'BER_est': ber_list
        })
        df.to_csv("robustness_sweep_results.csv", index=False)

        # Plot the sweep in a new figure window
        fig, axs = plt.subplots(3,1, figsize=(8,10))
        axs[0].plot(detunes, on_power_list, '-o')
        axs[0].set_title("Drop-port sampled transmission (on) vs Detuning")
        axs[0].set_xlabel("Detuning (nm)")
        axs[0].set_ylabel("T_drop at laser (a.u.)")
        axs[0].grid(True)

        axs[1].semilogy(detunes, np.maximum(ber_list, 1e-12), '-o')
        axs[1].set_title("Estimated BER vs Detuning")
        axs[1].set_xlabel("Detuning (nm)")
        axs[1].set_ylabel("BER (log scale)")
        axs[1].grid(True, which='both', ls='--')

        axs[2].plot(detunes, contrast_list, '-o', label='Contrast %')
        axs2 = axs[2].twinx()
        axs2.plot(detunes, snr_list, '-.s', color='orange', label='SNR (a.u.)')
        axs[2].set_xlabel("Detuning (nm)")
        axs[2].set_ylabel("Contrast (%)")
        axs2.set_ylabel("SNR (on / noise RMS)")
        axs[2].set_title("Contrast % and SNR vs Detuning")
        axs[2].grid(True)

        fig.tight_layout()
        plt.show()

        QMessageBox.information(self, "Robustness Sweep", f"Done. Results saved to 'robustness_sweep_results.csv'")

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    import math
    app = QApplication(sys.argv)
    gui = RingSimulatorGUI()
    gui.show()
    sys.exit(app.exec())
