import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.ticker import ScalarFormatter


# ========== Parameters ==========
TRUE_H0 = 67.66
H0_THRESHOLD = 80

# Load data
df_results = pd.read_csv("final_bootstrap_results_comb_50k_z10v3.csv", comment="#")
df_results = pd.read_csv("final_bootstrap_results_comb_50k_z10_e.csv", comment="#")
print("Columns in df_results:", df_results.columns)

true_values = [67.66, 10, 0, 33.4, 0,
               67.66, 10, 0, 33.4, 0,
               67.66, 10, 0, 33.4, 0,
               67.66, 10, 0, 33.4, 0]

params = ['H0_1', 'mass_low_1','k_low_1','mass_high_1','k_high_1',
          'H0_2', 'mass_low_2','k_low_2','mass_high_2','k_high_2',
          'H0_3', 'mass_low_3','k_low_3','mass_high_3','k_high_3',
          'H0_4', 'mass_low_4','k_low_4','mass_high_4','k_high_4']

# Save KDE curves for combined plot
kde_results = {}

# ========== KDE Plotting Function ==========
def compute_kde(samples, label=None):
    """Compute KDE and peak/uncertainty for given samples."""
    kde = gaussian_kde(samples)
    x_vals = np.linspace(samples.min(), samples.max(), 500)
    kde_vals = kde(x_vals)

    peak = x_vals[np.argmax(kde_vals)]
    half_max = np.max(kde_vals) / 2
    left_idx = np.where(kde_vals[:np.argmax(kde_vals)] <= half_max)[0][-1]
    right_idx = np.where(kde_vals[np.argmax(kde_vals):] <= half_max)[0][0] + np.argmax(kde_vals)
    uncertainty = (x_vals[right_idx] - x_vals[left_idx]) / 2

    if label:
        print(f"Peak {label}: {peak:.2f}")
        print(f"Uncertainty in peak: {uncertainty:.2f}")

    return x_vals, kde_vals, peak, uncertainty

def plot_kde(samples, true_value, filename, label):
    hist_data, bin_edges = np.histogram(samples, bins=100, density=True)
    kde = gaussian_kde(samples)
    x_vals = np.linspace(bin_edges[0], bin_edges[-1], 500)
    kde_vals = kde(x_vals)

    # Save KDE for combined plot
    if label.startswith("H0_"):
        kde_results[label] = (x_vals, kde_vals)

    peak = x_vals[np.argmax(kde_vals)]
    half_max = np.max(kde_vals) / 2
    left_idx = np.where(kde_vals[:np.argmax(kde_vals)] <= half_max)[0][-1]
    right_idx = np.where(kde_vals[np.argmax(kde_vals):] <= half_max)[0][0] + np.argmax(kde_vals)
    uncertainty = (x_vals[right_idx] - x_vals[left_idx]) / 2

    plt.figure(figsize=(8, 6))
    plt.hist(samples, bins=100, density=True, alpha=0.6, color='blue')
    plt.plot(x_vals, kde_vals, color='red', label = fr'KDE Fit')#label=fr'KDE Fit (Peak = {peak:.2f})')
    plt.axvline(peak, color='red', linestyle='--', label=fr'Peak $H_0$ = {peak:.2f}')
    plt.axvline(peak - uncertainty, color='green', linestyle='--', label=fr'Uncertainty = ±{uncertainty:.2f}')
    plt.axvline(peak + uncertainty, color='green', linestyle='--')
    plt.axvline(true_value, color="purple", linestyle="--", linewidth=2, label=fr'True $H_0$ = {true_value:.2f}')

    plt.ylabel('Probability Density', fontsize=14)
    plt.xlabel(r'$H_0$', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"Peak {label}: {peak:.2f}")
    print(f"Uncertainty in peak: {uncertainty:.2f}")

def plot_combined_H0_kdes(csv_files, labels, output="combined_H0_kde_z.png"):
    """Overlay KDEs of H0_1 from multiple CSVs."""
    plt.figure(figsize=(8, 6))
    for file, label in zip(csv_files, labels):
        df = pd.read_csv(file)
        if 'H0_1' not in df.columns:
            print(f"'H0_1' not in {file}, skipping...")
            continue
        h0_vals = df['H0_1'].dropna()
        if label == "0 < z < 10":   h0_vals = df['H0_2'].dropna()

        h0_vals = h0_vals[h0_vals < H0_THRESHOLD]
        if h0_vals.empty:
            print(f"No valid H0_1 data in {file}")
            continue
        x, y, peak, _ = compute_kde(h0_vals, label)
        plt.plot(x, y, label=f"{label}")

    plt.axvline(TRUE_H0, color='black', linestyle='--', label=fr'$H_0$ = {TRUE_H0}')
    plt.xlabel(r'$H_0$', fontsize=16)
    plt.ylabel('Probability Density', fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13)
    plt.xlim(58, H0_THRESHOLD)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_kde_std_vs_N(csv_files, labels, N_values, column='H0_1', output="kde_std_vs_N.png"):
    """
    Plots the standard deviation of KDE distributions as a function of number of events (N),
    and compares it to 1/sqrt(N) scaling with a smooth reference curve.
    """
    kde_stds = []

    for file in csv_files:
        df = pd.read_csv(file)
        if column not in df.columns:
            print(f"Column '{column}' not in {file}, skipping...")
            kde_stds.append(np.nan)
            continue

        h0_vals = df[column].dropna()
        h0_vals = h0_vals[h0_vals < H0_THRESHOLD]

        if h0_vals.empty:
            print(f"No valid {column} data in {file}")
            kde_stds.append(np.nan)
            continue

        kde = gaussian_kde(h0_vals)
        x_vals = np.linspace(h0_vals.min(), h0_vals.max(), 500)
        kde_vals = kde(x_vals)

        mean = np.trapz(x_vals * kde_vals, x_vals)
        var = np.trapz((x_vals - mean) ** 2 * kde_vals, x_vals)
        std_dev = np.sqrt(var)
        kde_stds.append(std_dev)

    # Smooth reference curve for 1/sqrt(N)
    N_min = min(N_values)
    N_max = max(N_values)
    N_smooth = np.linspace(N_min, N_max, 300)
    inv_sqrt_N_smooth = 1 / np.sqrt(N_smooth)
    ref_scale = kde_stds[0] / (1 / np.sqrt(N_values[0]))
    inv_sqrt_N_scaled = inv_sqrt_N_smooth * ref_scale

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(N_values, kde_stds, marker='o', label='KDE Std Dev', color='blue', zorder=5)
    plt.plot(N_smooth, inv_sqrt_N_scaled, linestyle='--', color='gray', label=r'$\propto \frac{1}{\sqrt{N}}$', zorder=3)

    for x, y, lbl in zip(N_values, kde_stds, labels):
        plt.annotate(lbl, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=11)

    plt.xlabel('Number of Events (N)', fontsize=14)
    plt.ylabel(fr'$H_0$ Std Deviation', fontsize=14)
    plt.title('Spread of $H_0$ vs. Sample Size', fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # Format x-axis as 3e4, 4e4, etc.
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(4,4))  # Customize as needed

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()

def plot_k_distributions_with_cuts(csv_file="final_bootstrap_results_comb_50k_z10v3.csv"):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    df = pd.read_csv(csv_file)

    run_labels = ['A', 'B', 'C', 'D']
    k_values = [(0.0, 0.0), (0.1, 0.4), (0.5, 0.5), (0.4, -0.4)]  # (k1, k2) for each run
    k_low_cols = ['k_low_1', 'k_low_2', 'k_low_3', 'k_low_4']
    k_high_cols = ['k_high_1', 'k_high_2', 'k_high_3', 'k_high_4']

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    colors = ["red", "blue", "green", "orange"]

    for i, run_label, (k1_true, k2_true), k1_col, k2_col, color in zip(
        range(4), run_labels, k_values, k_low_cols, k_high_cols, colors):


        k1_vals = df[k1_col].dropna().copy()
        k2_vals = df[k2_col].dropna().copy()

        # Apply cuts
        if run_label == 'B':
            k1_vals = k1_vals[k1_vals >= 0.02]
        elif run_label in ['C', 'D']:
            k1_vals = k1_vals[k1_vals >= 0.2]

        k2_vals = k2_vals[k2_vals >= -1]

        for ax, vals, true_val, label in zip(
            axs,
            [k1_vals, k2_vals],
            [k1_true, k2_true],
            ['k₁', 'k₂']
        ):
            if len(vals) == 0:
                continue

            kde = gaussian_kde(vals)
            x = np.linspace(vals.min(), vals.max(), 500)
            y = kde(x)  # Already normalized PDF

            ax.plot(x, y, label=f'{label} Run {run_label}', color=color)

            # Only label the dotted line for run D
            if run_label == 'D':
                ax.axvline(true_val, linestyle=':', color=color, linewidth=1.5, label='Target')
            else:
                ax.axvline(true_val, linestyle=':', color=color, linewidth=1.5)

    # Final touches
    axs[0].set_xlabel(r'$k_1$', fontsize=28)
    axs[1].set_xlabel(r'$k_2$', fontsize=28)
    axs[0].set_ylabel('Probability Density', fontsize=28)

    for ax in axs:
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)

    plt.tight_layout()
    plt.savefig("k1_k2_distributions.png")
    plt.close()



"""
# Run the plots
for i, param in enumerate(params):
    if "H0" in param:

        if param not in df_results.columns:
            print(f"Column '{param}' not found in dataframe!")
            continue

        col_data = df_results[param].dropna().values

        col_data = col_data[col_data < 80]


        if len(col_data) == 0:
            print(f"No valid data for column '{param}' after filtering")
            continue

        plot_kde(
            col_data,
            true_values[i],
            f"kde_{param}_comb.png",
            param
        )

# Create combined KDE plot for H0 runs
plt.figure(figsize=(8, 6))
labels_map = {'H0_1': 'Run A', 'H0_2': 'Run B', 'H0_3': 'Run C', 'H0_4': 'Run D'}
labels_map = {'H0_1': 'No error', 'H0_2': 'Mass error', 'H0_3': 'Distance error', 'H0_4': 'Mass & distance error'}
colors = ['red', 'blue', 'green', 'orange']

for i, (key, (x, y)) in enumerate(kde_results.items()):
    label = labels_map.get(key, key)
    plt.plot(x, y, label=label, color=colors[i])
    if i == 3:
        plt.axvline(67.66, label = r"$H_0 = 67.66$", c= "black", ls = "--")

plt.xlabel(r'$H_0$', fontsize=16)
plt.ylabel('Probability Density', fontsize=16)
plt.xticks(fontsize=13)
plt.xlim(58,80)
plt.yticks(fontsize=13)
plt.legend(fontsize=14,loc="upper right")
plt.tight_layout()
plt.savefig("combined_H0_kde.png")
plt.close()


csv_files = [
    "final_bootstrap_results_comb_30k.csv",
    "final_bootstrap_results_comb.csv",
    "final_bootstrap_results_comb_100k.csv"

]

labels = [
    "N=30,000",
    "N=50,000",
    "N=100,000"
]

N_values = [30000, 50000, 100000]
plot_kde_std_vs_N(csv_files, labels, N_values)


csv_files = [
    "final_bootstrap_results_comb_50k_z3.csv",
    "final_bootstrap_results_comb_50k_z5v2.csv",
    "final_bootstrap_results_comb_50k_z10v3.csv"

]

labels = [
    "0 < z < 3",
    "0 < z < 5",
    "0 < z < 10"
]
"""

plot_k_distributions_with_cuts()

#plot_combined_H0_kdes(csv_files, labels,output="combined_H0_kde_z.png")
