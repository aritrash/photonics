# microring_fab_montecarlo.py
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QFormLayout, QHBoxLayout, QSplitter, QMessageBox, QFrame, QProgressBar
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------------------------
# Physics helpers (same analytic CMT-style model)
# ---------------------------
c = 3e8

def through_drop_spectrum(lam_nm, R_um, n_eff, kappa, loss_db_per_cm):
    """Return (T_through, T_drop) arrays for provided wavelength vector lam_nm (nm).
       R_um in microns.
    """
    R_m = R_um * 1e-6
    L = 2.0 * math.pi * R_m
    # convert loss dB/cm -> amplitude round-trip factor
    loss_db_per_m = loss_db_per_cm * 100.0
    if loss_db_per_m <= 0:
        a_rt = 1.0
    else:
        a_rt = 10 ** ( - (loss_db_per_m * L) / 20.0 )

    kappa = float(max(min(kappa, 0.9999), 1e-9))
    t = math.sqrt(max(0.0, 1.0 - kappa**2))

    lam_m = lam_nm * 1e-9
    beta = 2.0 * math.pi * n_eff / lam_m
    phi = beta * L
    exp_term = a_rt * np.exp(-1j * phi)

    E_th = (t - exp_term) / (1.0 - t * exp_term)
    T_th = np.abs(E_th)**2

    E_drop = (1j * kappa * np.sqrt(a_rt) * np.exp(-1j * phi/2.0)) / (1.0 - t * exp_term)
    T_drop = np.abs(E_drop)**2

    return T_th, T_drop

def analyze_spectrum(lam_nm, T, R_um, n_eff):
    """Return dictionary of metrics from spectrum T (lam_nm in nm)."""
    Tmax = float(np.max(T))
    Tmin = float(np.min(T))
    ER_db = 10.0 * np.log10((Tmax + 1e-18) / (Tmin + 1e-18)) if Tmin > 0 else np.inf
    contrast_pct = float((Tmax - Tmin) / (Tmax + 1e-18) * 100.0) if Tmax > 0 else 0.0

    idx_min = int(np.argmin(T))
    lam_res = float(lam_nm[idx_min])

    half = (Tmax + Tmin) / 2.0
    indices = np.where(T < half)[0]
    if len(indices) > 1:
        delta_lambda = float(lam_nm[indices[-1]] - lam_nm[indices[0]])
        Q = lam_res / delta_lambda if delta_lambda > 0 else np.inf
    else:
        delta_lambda = np.nan
        Q = np.inf

    R_m = R_um * 1e-6
    L = 2.0 * math.pi * R_m
    FSR_nm = (lam_res**2) / (n_eff * L) * 1e9 if (n_eff>0 and L>0) else np.nan

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
    def __init__(self, figsize=(11,8)):
        self.fig = Figure(figsize=figsize)
        super().__init__(self.fig)

# ---------------------------
# Main GUI
# ---------------------------
class FabricationMonteCarloGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microring Simulator — Fabrication Monte Carlo")
        self.setGeometry(60, 60, 1500, 900)
        self._build_ui()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter()

        # Left control panel
        left = QFrame()
        left.setMinimumWidth(380)
        form = QFormLayout()

        # Device params
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
        form.addRow(QLabel("Samples N (spectrum):"), self.N_input)

        # Buttons for quick sim / baseline
        self.sim_btn = QPushButton("Simulate Baseline")
        self.sim_btn.clicked.connect(self.on_simulate)
        form.addRow(self.sim_btn)

        # Monte Carlo controls
        form.addRow(QLabel("<b>Fabrication defect model (Monte Carlo)</b>"))
        self.n_samples_input = QLineEdit("500")
        self.bits_per_sample_input = QLineEdit("500")
        self.sigma_R_nm_input = QLineEdit("20")        # nm
        self.sigma_kappa_pct_input = QLineEdit("5")    # percent
        self.sigma_neff_input = QLineEdit("0.005")     # absolute
        self.sigma_loss_pct_input = QLineEdit("5")     # percent
        self.noise_rms_input = QLineEdit("0.001")      # detector noise RMS

        form.addRow(QLabel("Monte Carlo samples:"), self.n_samples_input)
        form.addRow(QLabel("Bits per sample (BER est):"), self.bits_per_sample_input)
        form.addRow(QLabel("σ_radius (nm):"), self.sigma_R_nm_input)
        form.addRow(QLabel("σ_kappa (%):"), self.sigma_kappa_pct_input)
        form.addRow(QLabel("σ_n_eff (abs):"), self.sigma_neff_input)
        form.addRow(QLabel("σ_loss (%):"), self.sigma_loss_pct_input)
        form.addRow(QLabel("Detector noise RMS:"), self.noise_rms_input)

        # yield thresholds
        form.addRow(QLabel("<b>Yield criteria (per-sample)</b>"))
        self.er_thresh_input = QLineEdit("10.0")   # dB
        self.tau_thresh_input = QLineEdit("5.0")   # ps
        self.il_thresh_input = QLineEdit("6.0")    # dB (through-port insertion loss)
        form.addRow(QLabel("ER threshold (dB):"), self.er_thresh_input)
        form.addRow(QLabel("τ threshold (ps):"), self.tau_thresh_input)
        form.addRow(QLabel("IL (through) threshold (dB):"), self.il_thresh_input)

        # Monte Carlo button and progress bar
        self.monte_btn = QPushButton("Run Fabrication Monte Carlo")
        self.monte_btn.clicked.connect(self.on_run_montecarlo)
        form.addRow(self.monte_btn)

        self.progress = QProgressBar()
        form.addRow(self.progress)

        # Results summary label
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        form.addRow(QLabel("<b>Monte Carlo summary</b>"))
        form.addRow(self.summary_label)

        # Save results quick button
        self.save_btn = QPushButton("Save last results CSV")
        self.save_btn.clicked.connect(self.save_last_csv)
        form.addRow(self.save_btn)

        left.setLayout(form)
        splitter.addWidget(left)

        # Right: Plot canvas
        right = QFrame()
        right_layout = QVBoxLayout()
        self.canvas = MplCanvas(figsize=(11,9))
        right_layout.addWidget(self.canvas)
        right.setLayout(right_layout)
        splitter.addWidget(right)

        splitter.setSizes([380, 1100])
        main_layout.addWidget(splitter)

        # internal
        self.last_mc_df = None

    # -----------------------
    # Baseline single simulation
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

        lam_nm = np.linspace(lam_min, lam_max, N)
        T_th, T_drop = through_drop_spectrum(lam_nm, R, neff, kappa, loss)
        m_drop = analyze_spectrum(lam_nm, T_drop, R, neff)
        m_th = analyze_spectrum(lam_nm, T_th, R, neff)

        # draw 2x2 plots: spectrum, drop-zoom, hist placeholder, metrics text
        fig = self.canvas.fig
        fig.clf()
        axs = fig.subplots(2,2)

        ax0 = axs[0,0]
        ax0.plot(lam_nm, T_th, label='Through')
        ax0.plot(lam_nm, T_drop, label='Drop')
        ax0.set_xlabel("Wavelength (nm)"); ax0.set_ylabel("Transmission")
        ax0.set_title("Baseline Transmission")
        ax0.legend(); ax0.grid(True)

        ax1 = axs[0,1]
        idx = int(np.argmin(T_drop))
        lam_res = lam_nm[idx]
        mask = (lam_nm > lam_res - 1) & (lam_nm < lam_res + 1)
        ax1.plot(lam_nm[mask], T_drop[mask])
        ax1.set_title("Drop-port resonance (zoom)"); ax1.grid(True)

        ax2 = axs[1,0]
        # placeholder histogram with just the baseline point
        ax2.hist([m_drop['ER_db']], bins=8)
        ax2.set_title("ER (example)")

        ax3 = axs[1,1]
        txt = (f"Drop ER = {m_drop['ER_db']:.2f} dB\n"
               f"Drop contrast = {m_drop['contrast_pct']:.2f}%\n"
               f"Drop τ = {m_drop['tau_ps']:.3f} ps\n"
               f"Through IL = {-10*math.log10(m_th['Tmax']+1e-18):.2f} dB\n"
               f"Res λ = {m_drop['lam_res_nm']:.4f} nm\n"
               f"FSR = {m_drop['FSR_nm']:.3f} nm")
        ax3.text(0.05, 0.95, txt, va='top', fontsize=10)
        ax3.axis('off')

        fig.tight_layout()
        self.canvas.draw()

    # -----------------------
    # Monte Carlo sampling
    # -----------------------
    def on_run_montecarlo(self):
        try:
            # base params
            R_nom = float(self.radius_input.text())
            neff_nom = float(self.neff_input.text())
            kappa_nom = float(self.kappa_input.text())
            loss_nom = float(self.loss_input.text())
            lam_min = float(self.lam_min_input.text())
            lam_max = float(self.lam_max_input.text())
            N_lambda = int(self.N_input.text())

            # MC params
            n_samples = int(self.n_samples_input.text())
            bits_per = int(self.bits_per_sample_input.text())
            sigma_R_nm = float(self.sigma_R_nm_input.text())
            sigma_kappa_pct = float(self.sigma_kappa_pct_input.text()) / 100.0
            sigma_neff = float(self.sigma_neff_input.text())
            sigma_loss_pct = float(self.sigma_loss_pct_input.text()) / 100.0
            noise_rms = float(self.noise_rms_input.text())

            # yield thresholds
            er_thresh = float(self.er_thresh_input.text())
            tau_thresh = float(self.tau_thresh_input.text())
            il_thresh = float(self.il_thresh_input.text())
        except Exception as e:
            QMessageBox.critical(self, "Input error", f"Invalid input: {e}")
            return

        # Safety checks
        if n_samples < 5 or bits_per < 10:
            QMessageBox.warning(self, "Params", "Increase samples (>=5) and bits per sample (>=10) for meaningful stats.")
            return

        # Precompute baseline spectrum & nominal resonance (laser locked there)
        lam_nm = np.linspace(lam_min, lam_max, N_lambda)
        T_th_nom, T_drop_nom = through_drop_spectrum(lam_nm, R_nom, neff_nom, kappa_nom, loss_nom)
        m_th_nom = analyze_spectrum(lam_nm, T_th_nom, R_nom, neff_nom)
        m_drop_nom = analyze_spectrum(lam_nm, T_drop_nom, R_nom, neff_nom)
        lam_signal = m_th_nom['lam_res_nm']  # laser locked to nominal through-port resonance
        # if needed, we could lock to drop-port resonance but through-port nominal is fine.

        # Arrays to store results
        rows = []
        rng = np.random.default_rng(2025)
        progress_step = max(1, n_samples // 100)
        self.progress.setRange(0, n_samples)
        self.progress.setValue(0)

        for i in range(n_samples):
            # sample defects
            delta_R_nm = rng.normal(0.0, sigma_R_nm)
            R_sample = R_nom + (delta_R_nm / 1000.0)  # nm -> um

            delta_kappa_frac = rng.normal(0.0, sigma_kappa_pct)
            kappa_sample = kappa_nom * (1.0 + delta_kappa_frac)
            kappa_sample = float(max(min(kappa_sample, 0.9999), 1e-6))

            delta_neff = rng.normal(0.0, sigma_neff)
            neff_sample = neff_nom + delta_neff

            delta_loss_frac = rng.normal(0.0, sigma_loss_pct)
            loss_sample = loss_nom * max(0.0, (1.0 + delta_loss_frac))

            # compute spectrum for sample
            T_th_s, T_drop_s = through_drop_spectrum(lam_nm, R_sample, neff_sample, kappa_sample, loss_sample)
            m_th_s = analyze_spectrum(lam_nm, T_th_s, R_sample, neff_sample)
            m_drop_s = analyze_spectrum(lam_nm, T_drop_s, R_sample, neff_sample)

            # sample drop-port value at fixed laser (locked to nominal)
            if lam_signal < lam_nm[0] or lam_signal > lam_nm[-1]:
                T_on = 0.0
            else:
                T_on = float(np.interp(lam_signal, lam_nm, T_drop_s))
            T_off = 0.0  # background level approximate

            # Estimate SNR and BER via Monte Carlo sampling of bits
            bits = rng.integers(0,2,size=bits_per)
            noise = rng.normal(0.0, noise_rms, size=bits_per)
            detected = np.where(bits==1, T_on, T_off) + noise
            mean_on = np.mean(detected[bits==1]) if np.any(bits==1) else 0.0
            mean_off = np.mean(detected[bits==0]) if np.any(bits==0) else 0.0
            thresh = 0.5 * (mean_on + mean_off)
            detected_bits = (detected >= thresh).astype(int)
            photonic_not = 1 - detected_bits
            expected_not = 1 - bits
            errors = int(np.sum(photonic_not != expected_not))
            ber = errors / bits_per

            # through-port insertion loss (dB)
            IL_th_dB = -10.0 * math.log10(m_th_s['Tmax'] + 1e-18) if (m_th_s['Tmax']>0) else np.nan

            row = {
                'sample_idx': i,
                'R_um': R_sample, 'deltaR_nm': delta_R_nm,
                'kappa': kappa_sample, 'neff': neff_sample, 'loss_dB_cm': loss_sample,
                'T_on_at_nominal_laser': T_on, 'BER_est': ber,
                'ER_drop_db': m_drop_s['ER_db'], 'contrast_drop_pct': m_drop_s['contrast_pct'],
                'tau_ps': m_drop_s['tau_ps'], 'Q_drop': m_drop_s['Q'],
                'IL_th_dB': IL_th_dB
            }
            rows.append(row)

            # update progress
            if (i % progress_step) == 0 or i == n_samples-1:
                self.progress.setValue(i+1)
                QApplication.processEvents()

        # Build dataframe and compute stats
        df = pd.DataFrame(rows)
        self.last_mc_df = df
        csvname = "fabrication_montecarlo_results.csv"
        df.to_csv(csvname, index=False)

        # compute yield by thresholds
        cond = (df['ER_drop_db'] >= float(self.er_thresh_input.text())) & \
               (df['tau_ps'] <= float(self.tau_thresh_input.text())) & \
               (df['IL_th_dB'] <= float(self.il_thresh_input.text()))
        yield_frac = float(np.sum(cond)) / len(df)

        # summary text
        summary = (
            f"Samples: {len(df)}\n"
            f"Drop ER (mean±std): {df['ER_drop_db'].mean():.2f} ± {df['ER_drop_db'].std():.2f} dB\n"
            f"τ (mean±std): {np.nanmean(df['tau_ps']):.3f} ± {np.nanstd(df['tau_ps']):.3f} ps\n"
            f"BER mean: {df['BER_est'].mean():.3e}\n"
            f"Yield (ER≥{self.er_thresh_input.text()}dB, τ≤{self.tau_thresh_input.text()}ps, IL≤{self.il_thresh_input.text()}dB): {yield_frac*100:.2f}%\n"
            f"Results saved to: {csvname}"
        )
        self.summary_label.setText(summary)

        # Plot results: histogram ER, histogram tau, scatter ER vs tau colored by BER
        fig = self.canvas.fig
        fig.clf()
        axs = fig.subplots(2,2)

        ax0 = axs[0,0]
        ax0.hist(df['ER_drop_db'].replace([np.inf, -np.inf], np.nan).dropna(), bins=40)
        ax0.set_title("Drop-port ER histogram (dB)")
        ax0.set_xlabel("ER (dB)")

        ax1 = axs[0,1]
        # tau may have nan values (infinite Q)
        tau_plot = df['tau_ps'].copy()
        tau_plot = tau_plot.replace([np.inf], np.nan)
        ax1.hist(tau_plot.dropna(), bins=40)
        ax1.set_title("τ histogram (ps)")
        ax1.set_xlabel("tau (ps)")

        ax2 = axs[1,0]
        sc = ax2.scatter(df['ER_drop_db'], df['tau_ps'], c=df['BER_est'], cmap='viridis', s=18)
        ax2.set_xscale('linear')
        ax2.set_yscale('log')
        ax2.set_xlabel("ER (dB)")
        ax2.set_ylabel("tau (ps, log)")
        ax2.set_title("ER vs τ (color = BER)")
        fig.colorbar(sc, ax=ax2, label='BER')

        ax3 = axs[1,1]
        ax3.plot(df['T_on_at_nominal_laser'], '.', alpha=0.6)
        ax3.set_xlabel("Sample index")
        ax3.set_ylabel("T_on at nominal laser")
        ax3.set_title("T_on sampled at nominal laser (per sample)")

        fig.tight_layout()
        self.canvas.draw()

        QMessageBox.information(self, "Monte Carlo done", f"Monte Carlo finished. Results saved to '{csvname}'.")

    def save_last_csv(self):
        if self.last_mc_df is None:
            QMessageBox.warning(self, "No results", "No Monte Carlo results to save yet.")
            return
        name = "fabrication_montecarlo_results_last.csv"
        self.last_mc_df.to_csv(name, index=False)
        QMessageBox.information(self, "Saved", f"Saved last results to {name}.")


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = FabricationMonteCarloGUI()
    gui.show()
    sys.exit(app.exec())
