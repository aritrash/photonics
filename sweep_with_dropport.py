# sweep_with_dropport.py
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

c = 3e8

# user ranges (tweak)
R_vals_um = np.linspace(2.0, 8.0, 15)   # try smaller radii too
kappa_vals = np.linspace(0.005, 0.6, 120)
neff = 2.4
loss_db_per_cm = 0.1
lam_min_nm = 1520.0
lam_max_nm = 1580.0
N_lambda = 3000

# targets
ER_target_db = 10.0   # required extinction at through or drop port (dB)
tau_target_ps = 5.0

def compute_both_ports(R_m, kappa, neff, loss_db_per_cm, lam_min_nm, lam_max_nm, Nlam):
    L = 2*math.pi*R_m
    loss_db_per_m = loss_db_per_cm * 100.0
    if loss_db_per_m <= 0:
        a_rt = 1.0
    else:
        a_rt = 10 ** ( - (loss_db_per_m * L) / 20.0 )

    kappa = max(min(kappa, 0.9999), 1e-6)
    t = math.sqrt(max(0.0, 1.0 - kappa**2))

    lam = np.linspace(lam_min_nm, lam_max_nm, Nlam)*1e-9
    beta = 2*np.pi*neff/lam
    phi = beta*L
    exp_term = a_rt * np.exp(-1j*phi)

    # through port amplitude (side-coupled single ring)
    E_th = (t - exp_term) / (1 - t*exp_term)
    T_th = np.abs(E_th)**2

    # add-drop ring model: assume symmetric coupling to add port (for simplicity)
    # drop amplitude (approx) uses cross-coupling; simple model:
    # E_drop ~ (-j * kappa * sqrt(a_rt) * exp(-j*phi/2)) / (1 - t*exp_term)
    # This is a simplification—still gives usable trends.
    E_drop = (1j * kappa * np.sqrt(a_rt) * np.exp(-1j*phi/2.0)) / (1 - t*exp_term)
    T_drop = np.abs(E_drop)**2

    Tmax_th = float(np.max(T_th)); Tmin_th = float(np.min(T_th))
    Tmax_drop = float(np.max(T_drop)); Tmin_drop = float(np.min(T_drop))

    ER_th_db = 10*np.log10((Tmax_th+1e-18)/(Tmin_th+1e-18)) if Tmin_th>0 else np.inf
    ER_drop_db = 10*np.log10((Tmax_drop+1e-18)/(Tmin_drop+1e-18)) if Tmin_drop>0 else np.inf

    contrast_th = (Tmax_th - Tmin_th)/(Tmax_th + 1e-18) *100.0
    contrast_drop = (Tmax_drop - Tmin_drop)/(Tmax_drop + 1e-18) *100.0

    # estimate Q & tau from through-port FWHM of the deepest dip
    idx_min = int(np.argmin(T_th))
    lam_res = lam[idx_min]
    half = (Tmax_th + Tmin_th)/2.0
    left = idx_min; right = idx_min
    while left>0 and T_th[left] < half: left -=1
    while right < len(T_th)-1 and T_th[right] < half: right+=1
    if left==0 or right==len(T_th)-1 or left==right:
        delta_lambda = 0.0
        Q = np.inf
    else:
        delta_lambda = lam[right]-lam[left]
        Q = lam_res/delta_lambda if delta_lambda>0 else np.inf

    tau_ps = (Q*lam_res)/(2*np.pi*c) *1e12 if np.isfinite(Q) else np.nan
    FSR_nm = (lam_res**2)/(neff*L)*1e9

    return {
        'Tmax_th':Tmax_th,'Tmin_th':Tmin_th,'ER_th_db':ER_th_db,'contrast_th':contrast_th,
        'Tmax_drop':Tmax_drop,'Tmin_drop':Tmin_drop,'ER_drop_db':ER_drop_db,'contrast_drop':contrast_drop,
        'lam_res_nm':lam_res*1e9,'Q':Q,'tau_ps':tau_ps,'FSR_nm':FSR_nm
    }

rows=[]
for R_um in tqdm(R_vals_um, desc='R loop'):
    R = R_um*1e-6
    for kappa in kappa_vals:
        m = compute_both_ports(R, kappa, neff, loss_db_per_cm, lam_min_nm, lam_max_nm, N_lambda)
        m.update({'R_um':R_um,'kappa':kappa})
        rows.append(m)

df = pd.DataFrame(rows)
df.to_csv("sweep_dropport_results.csv", index=False)
print("Saved sweep_dropport_results.csv")

# Find candidates where either through or drop meets ER target and tau <= target
candidates = df[((df['ER_th_db']>=ER_target_db) | (df['ER_drop_db']>=ER_target_db)) & (df['tau_ps']<=tau_target_ps)]
print(f"Found {len(candidates)} candidates meeting ER>={ER_target_db} dB and tau<={tau_target_ps} ps")
candidates_sorted = candidates.sort_values(['tau_ps','ER_th_db'], ascending=[True, False])
print(candidates_sorted.head(10).to_string(index=False))

# Save subset and plots
candidates_sorted.to_csv("sweep_candidates_dropport.csv", index=False)

# plot ER_drop vs tau scatter (color by R)
plt.figure(figsize=(7,5))
plt.scatter(df['tau_ps'], df['ER_drop_db'], c=df['R_um'], s=10, cmap='viridis')
plt.xscale('log'); plt.xlabel('tau (ps)'); plt.ylabel('ER_drop (dB)')
plt.colorbar(label='R (um)')
plt.title('Drop-port ER vs tau')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.savefig("drop_ER_vs_tau.png", dpi=200)

plt.figure(figsize=(7,5))
plt.scatter(df['tau_ps'], df['contrast_drop'], c=df['R_um'], s=10, cmap='plasma')
plt.xscale('log'); plt.xlabel('tau (ps)'); plt.ylabel('Drop-port contrast (%)')
plt.title('Drop-port contrast vs tau')
plt.colorbar(label='R (um)')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.savefig("drop_contrast_vs_tau.png", dpi=200)

print("Plots saved: drop_ER_vs_tau.png, drop_contrast_vs_tau.png")
