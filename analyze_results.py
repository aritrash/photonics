# analyze_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csvfile = "ring_sweep_results_all.csv"  # change if needed
df = pd.read_csv(csvfile)

# ensure columns exist
if 'contrast_pct' not in df.columns or 'tau_ps' not in df.columns:
    print("CSV missing required columns 'contrast_pct' or 'tau_ps'. Columns found:", df.columns.tolist())
    raise SystemExit

# scatter
plt.figure(figsize=(7,5))
plt.scatter(df['tau_ps'], df['contrast_pct'], s=10, alpha=0.6)
plt.xscale('log')
plt.xlabel('tau (ps) [log scale]')
plt.ylabel('Contrast (%)')
plt.title('Contrast % vs Switching time τ (ps)')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.savefig("contrast_vs_tau.png", dpi=200)
plt.show()

# Pareto: largest contrast for tau <= thresholds
thresholds = [1, 2, 5, 10, 50, 100]  # ps
for th in thresholds:
    sub = df[df['tau_ps'] <= th]
    if len(sub)==0:
        print(f"No samples with tau <= {th} ps")
        continue
    best = sub.loc[sub['contrast_pct'].idxmax()]
    print(f"Best for tau<={th} ps: contrast={best['contrast_pct']:.2f}%, tau={best['tau_ps']:.3f} ps, params: R_um={best.get('R_um',np.nan)}, kappa={best.get('kappa',np.nan)}, ER_db={best.get('ER_db',np.nan):.2f}")

# build Pareto frontier: maximize contrast for given tau
data = df[['tau_ps','contrast_pct']].dropna()
data = data.sort_values('tau_ps')
pareto = []
best_so_far = -1
for _, row in data.iterrows():
    if row['contrast_pct'] > best_so_far:
        pareto.append(row)
        best_so_far = row['contrast_pct']
pareto_df = pd.DataFrame(pareto)
pareto_df.to_csv("pareto_contrast_vs_tau.csv", index=False)
print("Saved pareto frontier to pareto_contrast_vs_tau.csv")
