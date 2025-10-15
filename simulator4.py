import sys
import numpy as np
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QFormLayout, QHBoxLayout
)


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


def simulate(R_um, n_eff, kappa, loss, lam_min, lam_max, N):
    lam = np.linspace(lam_min, lam_max, N)
    T_th, T_drop = through_drop_spectrum(lam, R_um, n_eff, kappa, loss)

    m_th = metrics(lam, T_th, R_um, n_eff)
    m_drop = metrics(lam, T_drop, R_um, n_eff)

    print("\n=== Microring Simulation Results ===")
    print(f"Parameters: R = {R_um:.3f} μm, n_eff = {n_eff}, loss = {loss} dB/cm, κ = {kappa:.3f}")
    print("--------------------------------------------------")
    print("Through-port metrics:")
    for k, v in m_th.items():
        print(f"  {k}: {v}")
    print("--------------------------------------------------")
    print("Drop-port metrics:")
    for k, v in m_drop.items():
        print(f"  {k}: {v}")

    # Plots
    plt.figure(figsize=(12, 8))

    # Transmission spectrum
    plt.subplot(2, 2, 1)
    plt.plot(lam, T_th, label="Through-port")
    plt.plot(lam, T_drop, label="Drop-port")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.title("Transmission Spectrum")
    plt.legend()

    # Logic operation
    bit_seq = np.random.randint(0, 2, 50)
    logic_in = bit_seq
    logic_out = 1 - bit_seq
    plt.subplot(2, 2, 2)
    plt.step(range(len(bit_seq)), logic_in, label="Input Bits")
    plt.step(range(len(bit_seq)), logic_out, label="NOT Output")
    plt.ylim(-0.2, 1.2)
    plt.xlabel("Bit Index")
    plt.ylabel("Logic Value")
    plt.title("Logic Operation (Idealized)")
    plt.legend()

    # Switching time comparison
    plt.subplot(2, 2, 3)
    labels = ["TTL (7404)", "28nm CMOS", "7nm CMOS", "Photonic"]
    times = [1e4, 10, 2, m_drop["tau_ps"]]
    plt.bar(labels, times, color="orange")
    plt.yscale("log")
    plt.ylabel("Switching time (ps, log scale)")
    plt.title("Switching Speed Comparison")

    # Drop-port zoom
    plt.subplot(2, 2, 4)
    idx_res = np.argmin(T_drop)
    lam_res = lam[idx_res]
    mask = (lam > lam_res - 1) & (lam < lam_res + 1)
    plt.plot(lam[mask], T_drop[mask])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Drop Transmission")
    plt.title("Drop-port Resonance")

    plt.tight_layout()
    plt.show()


# -------------------------------
# PyQt6 GUI
# -------------------------------

class RingSimulator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microring Resonator Simulator")

        layout = QVBoxLayout()

        form = QFormLayout()

        self.radius_input = QLineEdit("2.428571")
        self.neff_input = QLineEdit("2.4")
        self.kappa_input = QLineEdit("0.6")
        self.loss_input = QLineEdit("0.1")
        self.lam_min_input = QLineEdit("1520")
        self.lam_max_input = QLineEdit("1580")
        self.N_input = QLineEdit("4000")

        form.addRow(QLabel("Radius (μm):"), self.radius_input)
        form.addRow(QLabel("n_eff:"), self.neff_input)
        form.addRow(QLabel("Coupling κ:"), self.kappa_input)
        form.addRow(QLabel("Loss (dB/cm):"), self.loss_input)
        form.addRow(QLabel("λ min (nm):"), self.lam_min_input)
        form.addRow(QLabel("λ max (nm):"), self.lam_max_input)
        form.addRow(QLabel("Samples N:"), self.N_input)

        layout.addLayout(form)

        self.sim_btn = QPushButton("Simulate")
        self.sim_btn.clicked.connect(self.run_sim)
        layout.addWidget(self.sim_btn)

        self.setLayout(layout)

    def run_sim(self):
        try:
            R = float(self.radius_input.text())
            neff = float(self.neff_input.text())
            kappa = float(self.kappa_input.text())
            loss = float(self.loss_input.text())
            lam_min = float(self.lam_min_input.text())
            lam_max = float(self.lam_max_input.text())
            N = int(self.N_input.text())

            simulate(R, neff, kappa, loss, lam_min, lam_max, N)
        except Exception as e:
            print("Error:", e)


# -------------------------------
# Run GUI
# -------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RingSimulator()
    win.show()
    sys.exit(app.exec())
