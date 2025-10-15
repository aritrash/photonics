# thermal_tuning.py
import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QHBoxLayout
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ThermalTuningWindow(QWidget):
    def __init__(self, lambda_res=1550.0, n_eff=2.4):
        super().__init__()

        self.setWindowTitle("Microring Resonator Thermal Tuning")
        self.setGeometry(200, 200, 900, 600)

        # Default parameters
        self.lambda_res = lambda_res  # nm
        self.n_eff = n_eff            # effective index
        self.dndT = 1.86e-4           # 1/K (thermo-optic coeff, Si)
        self.eta = 0.2                # K/mW (heater efficiency)
        self.P_heater = 0.0           # mW

        # Layouts
        layout = QHBoxLayout()
        control_layout = QVBoxLayout()
        plot_layout = QVBoxLayout()

        # Sliders + labels
        self.sliders = {}
        self.labels = {}

        self.add_slider("Heater Power (mW)", "P_heater", 0, 50, self.P_heater, control_layout)
        self.add_slider("dn/dT (1/K)", "dndT", 1, 5, self.dndT * 1e4, control_layout, scale=1e-4)
        self.add_slider("Heater Efficiency (K/mW)", "eta", 1, 100, self.eta * 10, control_layout, scale=0.1)

        # Output labels
        self.result_label = QLabel("Results will appear here", self)
        control_layout.addWidget(self.result_label)

        # Matplotlib figure
        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)

        layout.addLayout(control_layout, 2)
        layout.addLayout(plot_layout, 5)
        self.setLayout(layout)

        self.update_plot()

    def add_slider(self, text, attr, min_val, max_val, init_val, layout, scale=1.0):
        """Helper to add a labeled slider"""
        label = QLabel(f"{text}: {init_val:.3f}", self)
        slider = QSlider(Qt.Orientation.Horizontal, self)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(int(init_val))
        slider.valueChanged.connect(lambda val, a=attr, s=scale, l=label, t=text: self.slider_changed(a, val, s, l, t))
        layout.addWidget(label)
        layout.addWidget(slider)
        self.sliders[attr] = slider
        self.labels[attr] = label

    def slider_changed(self, attr, val, scale, label, text):
        setattr(self, attr, val * scale)
        label.setText(f"{text}: {getattr(self, attr):.4f}")
        self.update_plot()

    def update_plot(self):
        """Recalculate and update plot + results"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Calculate ΔT, Δn, Δλ
        delta_T = self.eta * self.P_heater
        delta_n = self.dndT * delta_T
        delta_lambda = self.lambda_res * delta_n / self.n_eff
        new_lambda_res = self.lambda_res + delta_lambda

        # Assume FSR (you can pass this as a parameter too)
        FSR = 12.0  # nm, typical for ~5 μm radius ring
        power_per_FSR = (FSR * self.n_eff) / (self.lambda_res * self.dndT * self.eta)

        # Simulated spectrum
        lambdas = np.linspace(self.lambda_res - 5, self.lambda_res + 5, 500)
        transmission = 1 - np.exp(-((lambdas - new_lambda_res) ** 2) / 0.2**2)

        ax.plot(lambdas, transmission, label="Transmission Spectrum")
        ax.axvline(new_lambda_res, color='r', linestyle='--', label="New Resonance")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmission (a.u.)")
        ax.set_title("Thermally Tuned Microring Resonance")
        ax.legend()
        self.canvas.draw()

        # Results (with FSR tuning power)
        self.result_label.setText(
            f"ΔT = {delta_T:.2f} K | Δn = {delta_n:.2e}\n"
            f"Δλ = {delta_lambda:.3f} nm | New λ_res = {new_lambda_res:.3f} nm\n"
            f"Power required to shift 1 FSR ≈ {power_per_FSR:.2f} mW"
        )



# For standalone testing
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ThermalTuningWindow()
    window.show()
    sys.exit(app.exec())
