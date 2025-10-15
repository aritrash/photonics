import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

c = 3e8  # Speed of light (m/s)

class RingResonatorSimulator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photonic NOT Gate Simulator")
        self.setGeometry(200, 200, 1000, 600)

        layout = QHBoxLayout()
        self.setLayout(layout)

        # Left panel - inputs
        input_panel = QVBoxLayout()

        self.r_input = QLineEdit("10e-6")   # radius (m)
        self.neff_input = QLineEdit("2.4")  # effective refractive index
        self.alpha_input = QLineEdit("0.001") # loss coefficient
        self.kappa_input = QLineEdit("0.2") # coupling coefficient
        self.lmin_input = QLineEdit("1.50e-6") # wavelength min (m)
        self.lmax_input = QLineEdit("1.60e-6") # wavelength max (m)

        for label, widget in [
            ("Radius (m):", self.r_input),
            ("n_eff:", self.neff_input),
            ("Loss α:", self.alpha_input),
            ("Coupling κ:", self.kappa_input),
            ("λ min (m):", self.lmin_input),
            ("λ max (m):", self.lmax_input),
        ]:
            input_panel.addWidget(QLabel(label))
            input_panel.addWidget(widget)

        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)
        input_panel.addWidget(self.run_button)

        layout.addLayout(input_panel)

        # Right panel - plots
        plot_panel = QVBoxLayout()
        self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        plot_panel.addWidget(self.canvas)

        # Metrics output
        self.metrics_box = QTextEdit()
        plot_panel.addWidget(self.metrics_box)

        layout.addLayout(plot_panel)

    def run_simulation(self):
        # Get user parameters
        R = float(self.r_input.text())
        neff = float(self.neff_input.text())
        alpha = float(self.alpha_input.text())
        kappa = float(self.kappa_input.text())
        lmin = float(self.lmin_input.text())
        lmax = float(self.lmax_input.text())

        # Wavelength sweep
        lam = np.linspace(lmin, lmax, 1000)
        L = 2 * np.pi * R
        beta = 2 * np.pi * neff / lam
        phi = beta * L
        t = np.sqrt(1 - kappa**2)

        # Transmission spectrum (through port)
        E_out = (t - np.exp(-alpha * L / 2) * np.exp(-1j * phi)) / (1 - t * np.exp(-alpha * L / 2) * np.exp(-1j * phi))
        T = np.abs(E_out)**2

        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()

        # Plot transmission spectrum
        self.ax1.plot(lam * 1e9, T)
        self.ax1.set_title("Transmission Spectrum")
        self.ax1.set_xlabel("Wavelength (nm)")
        self.ax1.set_ylabel("Transmission")

        # Logic NOT behavior (binary simplification)
        input_signal = np.array([0, 1])
        output_signal = 1 - input_signal
        self.ax2.step(input_signal, output_signal, where="mid")
        self.ax2.set_title("NOT Gate Behavior")
        self.ax2.set_xlabel("Input")
        self.ax2.set_ylabel("Output")

        # Extract performance metrics
        T_min = np.min(T)
        T_max = np.max(T)
        ER = 10 * np.log10(T_max / T_min + 1e-12)
        IL = 10 * np.log10(1 / T_max + 1e-12)

        # Find resonance dip
        idx_min = np.argmin(T)
        lam_res = lam[idx_min]
        half_max = (T_max + T_min) / 2
        indices = np.where(T < half_max)[0]
        if len(indices) > 1:
            delta_lambda = (lam[indices[-1]] - lam[indices[0]])
        else:
            delta_lambda = 0
        Q = lam_res / delta_lambda if delta_lambda > 0 else np.inf

        # FSR
        res_indices = np.where(T < (T_min + 0.1))[0]
        if len(res_indices) > 10:
            FSR = (lam[res_indices[-1]] - lam[res_indices[0]]) / (len(res_indices) - 1)
        else:
            FSR = 0

        # Switching speed
        tau = Q * lam_res / (2 * np.pi * c)
        speed_ns = tau * 1e9
        comparison = "Faster than CMOS" if speed_ns < 0.1 else "Comparable/Slower"

        # Display metrics
        metrics_text = f"""
        Extinction Ratio (ER): {ER:.2f} dB
        Insertion Loss (IL): {IL:.2f} dB ({'Good' if IL < 3 else 'Poor'})
        Q-factor: {Q:.2e}
        Resonant λ: {lam_res*1e9:.2f} nm
        FSR: {FSR*1e9:.2f} nm
        Switching Speed (τ): {speed_ns:.3f} ns → {comparison}
        """
        self.metrics_box.setText(metrics_text)

        # Refresh plots
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    sim = RingResonatorSimulator()
    sim.show()
    sys.exit(app.exec())
