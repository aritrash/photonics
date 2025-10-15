# microring_not_desktop.py
# Desktop-style PyQt6 photonic NOT gate simulator with contrast sweep and electronics comparison
import sys
import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QRadioButton, QLabel, QSlider, QDockWidget, QPushButton, QDialog,
    QButtonGroup, QRadioButton, QGroupBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PyQt6.QtGui import QAction  # Add this at the top


# ------------------ Physics Helpers ------------------
def ring_through_transmission(k, a, phi=0.0):
    # Microring through port transmission
    k = float(min(max(k, 1e-6), 0.9999))
    a = float(min(max(a, 1e-6), 0.9999))
    t = np.sqrt(max(0.0, 1.0 - k))
    exp_mjphi = np.exp(-1j * phi)
    numerator = t - a * exp_mjphi
    denominator = 1.0 - a * t * exp_mjphi
    E_through = numerator / denominator
    return float(np.abs(E_through)**2)

# ------------------ Contrast Calculation ------------------
def compute_contrast(k, a, phi=0.0, carrier_amp=1.0):
    # Output when input = 0 (ON)
    I_ON = carrier_amp * ring_through_transmission(k, a, phi)
    # Output when input = 1 (OFF)
    I_OFF = (carrier_amp + 1.0) * ring_through_transmission(k, a, phi)
    contrast_pct = ((I_ON - I_OFF)/I_ON*100) if I_ON>0 else 0
    er_db = 10*np.log10(I_ON/I_OFF) if I_OFF>0 else np.inf
    return I_ON, I_OFF, contrast_pct, er_db

# ------------------ Contrast Sweep Dialog ------------------
class SweepDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Sweep Parameter")
        self.setMinimumWidth(250)
        layout = QVBoxLayout()
        self.param_group = QButtonGroup(self)
        self.rb_k = QRadioButton("Sweep Coupling k")
        self.rb_a = QRadioButton("Sweep Loss a")
        self.rb_k.setChecked(True)
        self.param_group.addButton(self.rb_k)
        self.param_group.addButton(self.rb_a)
        layout.addWidget(self.rb_k)
        layout.addWidget(self.rb_a)
        btn_run = QPushButton("Run Sweep")
        btn_run.clicked.connect(self.accept)
        layout.addWidget(btn_run)
        self.setLayout(layout)

    def selected_parameter(self):
        if self.rb_k.isChecked():
            return 'k'
        else:
            return 'a'

# ------------------ Main Window ------------------
class MicroringNOTMain(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microring NOT Simulator - Desktop Style")
        self.setMinimumSize(1200, 700)

        # Central widget for matplotlib canvas
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_layout = QVBoxLayout(self.central_widget)

        # Matplotlib canvas
        self.fig, self.ax = plt.subplots(figsize=(8,5))
        self.canvas = FigureCanvas(self.fig)
        self.central_layout.addWidget(self.canvas)

        # ------------------ Dock for Controls ------------------
        self.dock = QDockWidget("Controls", self)
        self.dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.dock)
        ctrl_widget = QWidget()
        dock_layout = QVBoxLayout(ctrl_widget)

        # Input selector
        gb_input = QGroupBox("Input Selector")
        input_layout = QVBoxLayout()
        self.rb0 = QRadioButton("Input = 0 (OFF)")
        self.rb1 = QRadioButton("Input = 1 (ON)")
        self.rb0.setChecked(True)
        input_layout.addWidget(self.rb0)
        input_layout.addWidget(self.rb1)
        gb_input.setLayout(input_layout)
        dock_layout.addWidget(gb_input)

        # Coupling k slider
        dock_layout.addWidget(QLabel("Coupling k (0.01..0.9)"))
        self.slider_k = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_k.setMinimum(1)
        self.slider_k.setMaximum(90)
        self.slider_k.setValue(20)
        dock_layout.addWidget(self.slider_k)

        # Loss a slider
        dock_layout.addWidget(QLabel("Round-trip amplitude a (0.50..0.999)"))
        self.slider_a = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_a.setMinimum(50)
        self.slider_a.setMaximum(999)
        self.slider_a.setValue(980)
        dock_layout.addWidget(self.slider_a)

        # Metrics labels
        dock_layout.addSpacing(20)
        dock_layout.addWidget(QLabel("Live Metrics:"))
        self.lbl_Ion = QLabel("Output(ON) = 0.0")
        self.lbl_Ioff = QLabel("Output(OFF) = 0.0")
        self.lbl_contrast = QLabel("Contrast % = 0.0")
        self.lbl_er = QLabel("ER (dB) = 0.0")
        dock_layout.addWidget(self.lbl_Ion)
        dock_layout.addWidget(self.lbl_Ioff)
        dock_layout.addWidget(self.lbl_contrast)
        dock_layout.addWidget(self.lbl_er)

        dock_layout.addStretch(1)
        self.dock.setWidget(ctrl_widget)

        # ------------------ Menu Bar ------------------
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_exit = QAction("Exit", self)
        file_exit.triggered.connect(self.close)
        file_menu.addAction(file_exit)

        plots_menu = menubar.addMenu("Plots")
        self.plot_transfer_action = QAction("Photonic NOT Transfer Curve", self)
        self.plot_transfer_action.triggered.connect(self.plot_transfer)
        plots_menu.addAction(self.plot_transfer_action)

        self.plot_sweep_action = QAction("Contrast Sweep", self)
        self.plot_sweep_action.triggered.connect(self.plot_contrast_sweep_dialog)
        plots_menu.addAction(self.plot_sweep_action)

        self.plot_electronics_action = QAction("Electronics Comparison", self)
        self.plot_electronics_action.triggered.connect(self.plot_electronics_comparison)
        plots_menu.addAction(self.plot_electronics_action)

        # ------------------ Signals ------------------
        self.rb0.toggled.connect(self.update_metrics)
        self.rb1.toggled.connect(self.update_metrics)
        self.slider_k.valueChanged.connect(self.update_metrics)
        self.slider_a.valueChanged.connect(self.update_metrics)

        # Initial plot
        self.plot_transfer()

    # ------------------ Plot Methods ------------------
    def plot_transfer(self):
        self.ax.clear()
        k = self.slider_k.value()/100
        a = self.slider_a.value()/1000
        phi = 0.0
        inp_vals = [0,1]
        outputs = []
        for inp in inp_vals:
            I_ON, I_OFF, _, _ = compute_contrast(k, a, phi)
            if inp==0:
                outputs.append(I_ON)
            else:
                outputs.append(I_OFF)
        self.ax.bar(['Input=0','Input=1'], outputs, color=['green','red'])
        self.ax.set_ylabel("Output intensity")
        self.ax.set_title("Photonic NOT Gate Transfer")
        self.canvas.draw()
        self.update_metrics()

    def plot_contrast_sweep_dialog(self):
        dlg = SweepDialog(self)
        if dlg.exec():
            param = dlg.selected_parameter()
            self.plot_contrast_sweep(param)

    def plot_contrast_sweep(self, sweep_param='k'):
        self.ax.clear()
        steps = 50
        k_base = self.slider_k.value()/100
        a_base = self.slider_a.value()/1000
        phi = 0.0
        sweep_vals = []
        contrast_vals = []
        if sweep_param=='k':
            sweep_vals = np.linspace(0.01,0.9,steps)
            for k in sweep_vals:
                _, _, contrast, _ = compute_contrast(k, a_base, phi)
                contrast_vals.append(contrast)
            xlabel = "Coupling k"
        elif sweep_param=='a':
            sweep_vals = np.linspace(0.50,0.999,steps)
            for a in sweep_vals:
                _, _, contrast, _ = compute_contrast(k_base, a, phi)
                contrast_vals.append(contrast)
            xlabel = "Round-trip amplitude a"
        self.ax.plot(sweep_vals, contrast_vals, marker='o')
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel("Contrast %")
        self.ax.set_title(f"Contrast vs {xlabel}")
        self.ax.grid(True)
        self.canvas.draw()

    def plot_electronics_comparison(self):
        self.ax.clear()
        names = ["7404 TTL","28nm CMOS","7nm CMOS","Photonic NOT"]
        delay_vals_ps = [15000,50,3,1]  # example delays in ps
        energy_vals_fJ = [10000,100,10,1] # example energy per bit in fJ
        x = np.arange(len(names))
        width=0.35
        self.ax.bar(x - width/2, delay_vals_ps, width, label="Delay (ps)")
        self.ax.bar(x + width/2, energy_vals_fJ, width, label="Energy (fJ)")
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(names)
        self.ax.set_yscale("log")
        self.ax.set_ylabel("Log scale")
        self.ax.set_title("Electronics vs Photonic NOT")
        self.ax.legend()
        self.ax.grid(True, which="both", axis="y", linestyle='--')
        self.canvas.draw()

    # ------------------ Metrics Update ------------------
    def update_metrics(self):
        k = self.slider_k.value()/100
        a = self.slider_a.value()/1000
        phi = 0.0
        I_ON, I_OFF, contrast, er = compute_contrast(k, a, phi)
        self.lbl_Ion.setText(f"Output(ON) = {I_ON:.3f}")
        self.lbl_Ioff.setText(f"Output(OFF) = {I_OFF:.3f}")
        self.lbl_contrast.setText(f"Contrast % = {contrast:.2f}")
        self.lbl_er.setText(f"ER (dB) = {er:.2f}")

# ------------------ Main ------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MicroringNOTMain()
    win.show()
    sys.exit(app.exec())
