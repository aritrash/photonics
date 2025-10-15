import sys
import numpy as np
import matplotlib
matplotlib.use("QtAgg")  # ensure Qt backend
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QFormLayout, QHBoxLayout, QSplitter
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# -------------------------------
# Microring functions
# -------------------------------

def through_drop_spectrum(lam, R, n_eff, kappa, loss_dB_per_cm):
    c = 3e8
    L = 2 * np.pi * R * 1e-6  # circumference (m)
    loss_lin = 10 ** (-loss_dB_per_cm * L * 100 / 20)

    beta = 2 * np.pi * n_eff / (lam * 1e-9)
    phi = beta * L

    t = np.sqrt(1 - kappa)
    a = np.sqrt(loss_lin)
    denom = 1 - a * t * np.exp(-1j * phi)

    t_through = (t - a * np.exp(-1j * phi)) / denom
    t_drop = 1j * np.sqrt(a * (1 - t**2)) * np.exp(-1j * phi / 2) / denom

    return np.abs(t_through) ** 2, np.abs(t_drop) ** 2


def metrics(lam, T, R, n_eff):
    Tmax = np.max(T)
    Tmin = np.min(T)
    ER_db = 10 * np.log10((Tmax + 1e-12) / (Tmin + 1e-12))
    contrast = (Tmax - Tmin) / (Tmax + 1e-12) * 100

    idx_min = np.argmin(T)
    lam_res = lam[idx_min]

    half = (Tmax + Tmin) / 2
    idxs = np.where(T < half)[0]
    if len(idxs) > 1:
        lam1, lam2 = lam[idxs[0]], lam[idxs[-1]]
        delta_lambda = lam2 - lam1
    else:
        delta_lambda = np.nan

    Q = lam_res / delta_lambda if delta_lambda and not np.isnan(delta_lambda) else np.inf

    c = 3e8
    tau = (Q * lam_res * 1e-9) / (2 * np.pi * c) * 1e12  # ps

    L = 2 * np.pi * R * 1e-6
    FSR = lam_res**2 / (n_eff * L * 1e9)

    return dict(
        Tmax=Tmax,
        Tmin=Tmin,
        ER_db=ER_db,
        contrast_pct=contrast,
        lam_res_nm=lam_res,
        Q=Q,
        tau_ps=tau,
        FSR_nm=FSR,
    )


# -------------------------------
# PyQt6 GUI + Matplotlib Canvas
# -------------------------------

class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(10, 8))
        super().__init__(self.fig)
        self.axs = self.fig.subplots(2, 2)


class RingSimulator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microring Resonator Simulator")
        self.setGeometry(100, 100, 1200, 800)

        # Layout
        layout = QHBoxLayout(self)
        splitter = QSplitter()
        layout.addWidget(splitter)

        # Left: Input panel
        form_widget = QWidget()
        form_layout = QFormLayout()

        self.radius_input = QLineEdit("2.428571")
        self.neff_input = QLineEdit("2.4")
        self.kappa_input = QLineEdit("0.6")
        self.loss_input = QLineEdit("0.1")
        self.lam_min_input = QLineEdit("1520")
        self.lam_max_input = QLineEdit("1580")
        self.N_input = QLineEdit("4000")

        form_layout.addRow(QLabel("Radius (μm):"), self.radius_input)
        form_layout.addRow(QLabel("n_eff:"), self.neff_input)
        form_layout.addRow(QLabel("Coupling κ:"), self.kappa_input)
        form_layout.addRow(QLabel("Loss (dB/cm):"), self.loss_input)
        form_layout.addRow(QLabel("λ min (nm):"), self.lam_min_input)
        form_layout.addRow(QLabel("λ max (nm):"), self.lam_max_input)
        form_layout.addRow(QLabel("Samples N:"), self.N_input)

        self.sim_btn = QPushButton("Simulate")
        self.sim_btn.clicked.connect(self.run_sim)
        form_layout.addRow(self.sim_btn)

        self.results_label = QLabel("")
        form_layout.addRow(QLabel("Results:"), self.results_label)

        form_widget.setLayout(form_layout)
        splitter.addWidget(form_widget)

        # Right: Plots
        self.canvas = MplCanvas()
        splitter.addWidget(self.canvas)

        splitter.setSizes([300, 900])  # give plots more space

    def run_sim(self):
        try:
            R = float(self.radius_input.text())
            neff = float(self.neff_input.text())
            kappa = float(self.kappa_input.text())
            loss = float(self.loss_input.text())
            lam_min = float(self.lam_min_input.text())
            lam_max = float(self.lam_max_input.text())
            N = int(self.N_input.text())

            lam = np.linspace(lam_min, lam_max, N)
            T_th, T_drop = through_drop_spectrum(lam, R, neff, kappa, loss)

            m_th = metrics(lam, T_th, R, neff)
            m_drop = metrics(lam, T_drop, R, neff)

            # Clear plots
            for ax in self.canvas.axs.flat:
                ax.clear()

            # Transmission spectrum
            ax = self.canvas.axs[0, 0]
            ax.plot(lam, T_th, label="Through-port")
            ax.plot(lam, T_drop, label="Drop-port")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Transmission")
            ax.set_title("Transmission Spectrum")
            ax.legend()

            # Logic operation
            ax = self.canvas.axs[0, 1]
            bit_seq = np.random.randint(0, 2, 50)
            logic_in = bit_seq
            logic_out = 1 - bit_seq
            ax.step(range(len(bit_seq)), logic_in, label="Input Bits")
            ax.step(range(len(bit_seq)), logic_out, label="NOT Output")
            ax.set_ylim(-0.2, 1.2)
            ax.set_xlabel("Bit Index")
            ax.set_ylabel("Logic Value")
            ax.set_title("Logic Operation")
            ax.legend()

            # Switching time comparison
            ax = self.canvas.axs[1, 0]
            labels = ["TTL (7404)", "28nm CMOS", "7nm CMOS", "Photonic"]
            times = [1e4, 10, 2, m_drop["tau_ps"]]
            ax.bar(labels, times, color="orange")
            ax.set_yscale("log")
            ax.set_ylabel("Switching time (ps, log scale)")
            ax.set_title("Switching Speed Comparison")

            # Drop-port zoom
            ax = self.canvas.axs[1, 1]
            idx_res = np.argmin(T_drop)
            lam_res = lam[idx_res]
            mask = (lam > lam_res - 1) & (lam < lam_res + 1)
            ax.plot(lam[mask], T_drop[mask])
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Drop Transmission")
            ax.set_title("Drop-port Resonance")

            self.canvas.fig.tight_layout()
            self.canvas.draw()

            # Show results in label
            text = (
                f"Drop-port ER = {m_drop['ER_db']:.2f} dB\n"
                f"Contrast = {m_drop['contrast_pct']:.2f}%\n"
                f"IL = {10*np.log10(m_drop['Tmax']+1e-12):.2f} dB\n"
                f"Q = {m_drop['Q']:.2e}\n"
                f"FSR = {m_drop['FSR_nm']:.2f} nm\n"
                f"Switching Time = {m_drop['tau_ps']:.2f} ps"
            )
            self.results_label.setText(text)

        except Exception as e:
            self.results_label.setText(f"Error: {e}")


# -------------------------------
# Run GUI
# -------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RingSimulator()
    win.show()
    sys.exit(app.exec())
