# ring_tradeoff_sweep.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from tqdm import tqdm

# constants
c = 3e8
lam0 = 1550e-9  # reference wavelength for tau calc (m)

# === User inputs / sweep ranges ===
# You can edit these ranges as you like
R_vals_um = np.linspace(3.0, 12.0, 20)   # ring radius in microns
kappa_vals = np.linspace(0.005, 0.6, 60)  # coupling coefficient (0..1)
neff = 2.4
loss_db_per_cm = 0.1   # dB/cm (your stated)
lam_min_nm = 1520.0
lam_max_nm = 1580.0
N_lambda = 5000

# performance targets
ER_target_db = 12.0   # desired extinction ratio threshold (dB)
tau_target_ps = 5.0   # desired max tau in ps to beat 28nm CMOS

# helper functions
def compute_ring_response(R_m, kappa, neff, loss_db_per_cm, lam_min, lam_max, Nlam=2000):
    L = 2.0 * math.pi * R_m
    # amplitude round-trip factor a_rt (convert loss db/cm -> db/m)
    loss_db_per_m = loss_db_per_cm * 100.0
    if loss_db_per_m <= 0:
        a_rt = 1.0
    else:
        # power attenuation over round-trip = 10^(-loss_db_per_m * L / 10)
        # amplitude attenuation = sqrt(power attenuation)
        a_rt = 10 ** ( - (loss_db_per_m * L) / 20.0 )

    # ensure kappa valid
    kappa = min(max(kappa, 1e-6), 0.9999)
    t = math.sqrt(max(0.0, 1.0 - kappa**2))

    lam = np.linspace(lam_min, lam_max, Nlam) * 1e-9  # m
    beta = 2.0 * np.pi * neff / lam
    phi = beta * L
    exp_term = a_rt * np.exp(-1j * phi)
    E_through = (t - exp_term) / (1.0 - t * exp_term)
    T = np.abs(E_through)**2

    # metrics from spectrum
    Tmax = float(np.max(T))
    Tmin = float(np.min(T))
    # protect against zero
    if Tmin <= 0:
        ER_db = np.inf
        contrast_pct = 100.0
    else:
        ER_db = 10.0 * np.log10((Tmax + 1e-18) / (Tmin + 1e-18))
        contrast_pct = float((Tmax - Tmin) / (Tmax + 1e-18) * 100.0)

    IL_db = -10.0 * np.log10(Tmax + 1e-18) if Tmax > 0 else np.inf

    # find deepest local minimum for FWHM/Q
    idx_min = int(np.argmin(T))
    lam_res = lam[idx_min]
    # estimate FWHM: find points where T crosses half-level between top and min
    half = (Tmax + Tmin) / 2.0
    # left
    left = idx_min
    while left > 0 and T[left] < half:
        left -= 1
    right = idx_min
    while right < (len(T)-1) and T[right] < half:
        right += 1
    if left == 0 or right == len(T)-1 or left==right:
        delta_lambda = 0.0
        Q = np.inf
    else:
        delta_lambda = lam[right] - lam[left]
        Q = lam_res / delta_lambda if delta_lambda>0 else np.inf

    # FSR approx using group index ~ neff
    FSR_nm = (lam_res**2) / (neff * L) * 1e9

    # tau in seconds and then ps
    if np.isfinite(Q):
        tau_s = (Q * lam_res) / (2.0 * np.pi * c)
        tau_ps = tau_s * 1e12
    else:
        tau_ps = np.nan

    return {
        'Tmax':Tmax, 'Tmin':Tmin, 'ER_db':ER_db, 'contrast_pct':contrast_pct,
        'IL_db':IL_db, 'lam_res_nm':lam_res*1e9, 'delta_lambda_nm':delta_lambda*1e9,
        'Q':Q, 'FSR_nm':FSR_nm, 'tau_ps':tau_ps
    }

# === sweep ===
rows = []
for R_um in tqdm(R_vals_um, desc="R sweep"):
    R_m = R_um * 1e-6
    for kappa in kappa_vals:
        metrics = compute_ring_response(R_m, kappa, neff, loss_db_per_cm, lam_min_nm, lam_max_nm, N_lambda)
        rows.append({
            'R_um':R_um,
            'kappa':kappa,
            **metrics
        })

df = pd.DataFrame(rows)

# filter usable solutions
candidates = df[(df['ER_db'] >= ER_target_db) & (df['tau_ps'] <= tau_target_ps) & (df['IL_db'] <= 6.0)]
candidates_sorted = candidates.sort_values(['tau_ps','ER_db'])
print(f"Found {len(candidates_sorted)} candidate parameter sets matching ER>={ER_target_db} dB and tau<={tau_target_ps} ps")

# save results
df.to_csv("ring_sweep_results_all.csv", index=False)
candidates_sorted.to_csv("ring_sweep_candidates.csv", index=False)

# quick plots
plt.figure(figsize=(8,6))
plt.scatter(df['tau_ps'], df['ER_db'], c=df['R_um'], cmap='viridis', s=12)
plt.colorbar(label='R (um)')
plt.xlabel('tau (ps)')
plt.ylabel('ER (dB)')
plt.xscale('log')
plt.title('ER vs tau (color -> R)')
plt.grid(True, which='both', ls='--', lw=0.4)
plt.savefig("ER_vs_tau_scatter.png", dpi=200)

# heatmap: ER for R vs kappa (pivot)
pivot = df.pivot_table(index='R_um', columns='kappa', values='ER_db', aggfunc='mean')
plt.figure(figsize=(8,6))
plt.imshow(pivot.values, origin='lower', aspect='auto', extent=[kappa_vals[0], kappa_vals[-1], R_vals_um[0], R_vals_um[-1]])
plt.colorbar(label='ER (dB)')
plt.xlabel('kappa')
plt.ylabel('R (um)')
plt.title('ER (dB) heatmap')
plt.savefig("ER_heatmap.png", dpi=200)

# print top candidates
print(candidates_sorted.head(10).to_string(index=False))
