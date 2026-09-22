import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION: Paths, Datasets & Methods
# ============================================================
CSV_FILE = "/projectnb/ace-ig/enrique/unsupervised_clustering/_stab_UFS/biomedical/results/ksweep_features_k100_complete.csv"
DATA_DIR = "/projectnb/ace-ig/enrique/unsupervised_clustering/_stab_UFS/biomedical/data/clean_data/"

# Subsampling parameters for clustering stability
N_SUBSAMPLES = 20       # Number of subsampling iterations per condition
SUBSAMPLE_RATIO = 0.8   # Proportion of validation samples to draw each time

# 1. Specify original base names of the 7 datasets
SELECTED_DATASETS = [
    'ADHD200_cleaned', 
    'ALLAML_cleaned', 
    'brca_cleaned', 
    'TADPOLE_cleaned',
    'nci60_cleaned',
    'mecfs_cleaned',
    'GLI_85_cleaned'
]

# 2. Clean display names for dataset titles
DATASET_NAME_MAP = {
    'ADHD200_cleaned': 'ADHD-200',
    'ALLAML_cleaned': 'ALLAML',
    'brca_cleaned': 'CPTAC-BRCA',
    'TADPOLE_cleaned': 'TADPOLE',
    'nci60_cleaned': 'NCI-60 GI50',
    'mecfs_cleaned': 'ME CFS',
    'GLI_85_cleaned': 'GLI 85'
}

# Method configuration: ordering, display labels, and colors
METHOD_CONFIG = {
    'laplacian': {'name': 'Laplacian', 'color': '#ffd166'},
    'ica':       {'name': 'ICA',       'color': '#ff9f1c'},
    'mcfs':      {'name': 'MCFS',      'color': '#f4a261'},
    'spec':      {'name': 'SPEC',      'color': '#e76f51'},
    'cae':       {'name': 'CAE',       'color': '#e63946'},
    'groupfs':   {'name': 'GroupFS',   'color': '#800f2f'},
    'stacsfs':   {'name': 'CLASP',     'color': '#0077b6'}
}

# The order they will appear in the bar chart and legend
METHOD_ORDER = ['laplacian', 'ica', 'mcfs', 'spec', 'cae', 'groupfs', 'stacsfs']

# Derived display tools for Seaborn
DISPLAY_ORDER = [METHOD_CONFIG[m]['name'] for m in METHOD_ORDER if m in METHOD_CONFIG]
PALETTE = {METHOD_CONFIG[m]['name']: METHOD_CONFIG[m]['color'] for m in METHOD_ORDER if m in METHOD_CONFIG}

SPECTRAL_METHODS = []

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def parse_int_list(val_str):
    matches = re.findall(r'int64\((\d+)\)', str(val_str))
    if not matches:
        matches = re.findall(r'\b\d+\b', str(val_str).replace('64', ''))
    return [int(m) for m in matches]

def compute_clustering_stability(X_reduced, k, raw_method, n_subsamples=N_SUBSAMPLES, subsample_ratio=SUBSAMPLE_RATIO, seed=43):
    """
    Subsamples validation data `n_subsamples` times, clusters each subsample, 
    and computes the average consensus (Pairwise ARI) across overlapping points.
    """
    n_samples = X_reduced.shape[0]
    if n_samples < k or X_reduced.shape[1] == 0:
        return np.nan
    
    subsample_size = max(k + 1, int(n_samples * subsample_ratio))
    if subsample_size > n_samples:
        subsample_size = n_samples

    subsampled_indices = []
    subsampled_labels = []

    # 1. Generate clustering runs on random subsamples
    for i in range(n_subsamples):
        rng = np.random.RandomState(seed + i)
        idx = rng.choice(n_samples, size=subsample_size, replace=False)
        X_sub = X_reduced[idx]

        if raw_method in SPECTRAL_METHODS:
            clusterer = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=seed + i)
        else:
            clusterer = KMeans(n_clusters=k, random_state=seed + i)

        try:
            labels = clusterer.fit_predict(X_sub)
            if len(set(labels)) > 1:
                subsampled_indices.append(idx)
                subsampled_labels.append(labels)
        except Exception:
            continue

    if len(subsampled_labels) < 2:
        return np.nan

    # 2. Compute consensus (ARI) on overlapping samples across pairs of subsamples
    ari_scores = []
    n_runs = len(subsampled_labels)
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            idx_i = subsampled_indices[i]
            idx_j = subsampled_indices[j]

            # Intersecting sample points between subsample i and j
            common_indices, pos_i, pos_j = np.intersect1d(idx_i, idx_j, return_indices=True)

            if len(common_indices) > 1:
                lbls_i = subsampled_labels[i][pos_i]
                lbls_j = subsampled_labels[j][pos_j]

                score = adjusted_rand_score(lbls_i, lbls_j)
                ari_scores.append(score)

    return np.mean(ari_scores) if ari_scores else np.nan

# ============================================================
# MAIN EXECUTION & STABILITY COMPUTATION
# ============================================================
# 1. Read and filter results CSV
results_df = pd.read_csv(CSV_FILE)

# Filter for chosen datasets and configured methods
results_df = results_df[
    results_df['Dataset'].isin(SELECTED_DATASETS) & 
    results_df['Method'].isin(METHOD_CONFIG.keys())
].copy()

results_df['Selected_Features'] = results_df['Selected_Features'].apply(parse_int_list)
results_df['Validation_Indices'] = results_df['Validation_Indices'].apply(parse_int_list)

# Load raw dataset matrices
raw_datasets = {}
for ds_name in SELECTED_DATASETS:
    filepath = os.path.join(DATA_DIR, f"{ds_name}.csv")
    if os.path.exists(filepath):
        raw_datasets[ds_name] = pd.read_csv(filepath).values

stability_records = []

# 2. Iterate through rows and calculate consensus stability
for idx, row in results_df.iterrows():
    ds_name = row['Dataset']
    if ds_name not in raw_datasets:
        continue

    X = raw_datasets[ds_name]
    k = row['TargetClusters']
    raw_method = row['Method']
    selected_features = row['Selected_Features']
    val_indices = row['Validation_Indices']

    if not selected_features or not val_indices:
        continue

    # Scale using training split and apply to validation split
    all_indices = set(range(X.shape[0]))
    train_indices = list(all_indices - set(val_indices))

    scaler = StandardScaler()
    scaler.fit(X[train_indices])
    X_val_scaled = scaler.transform(X[val_indices])
    X_val_reduced = X_val_scaled[:, selected_features]

    # Calculate stability consensus via subsampling
    stab_score = compute_clustering_stability(X_val_reduced, k, raw_method)

    stability_records.append({
        'Dataset': ds_name,
        'Method': raw_method,
        'TargetClusters': k,
        'Clustering Stability': stab_score
    })

# 3. Format Display Names & Compute Average Grid Panel
stab_df = pd.DataFrame(stability_records)

# Apply display mappings
stab_df['Dataset_Display'] = stab_df['Dataset'].map(DATASET_NAME_MAP).fillna(stab_df['Dataset'])
stab_df['Method_Display'] = stab_df['Method'].map(lambda x: METHOD_CONFIG.get(x, {}).get('name', x))

# Compute average across datasets per (Method, TargetClusters) pair
avg_df = (
    stab_df.groupby(['Method_Display', 'TargetClusters'], observed=False)['Clustering Stability']
    .mean()
    .reset_index()
)
avg_df['Dataset_Display'] = 'Average'

# Append Average dataset panel
stab_df = pd.concat([stab_df, avg_df], ignore_index=True)

# Order datasets so Average is in 8th position (bottom-right of 2x4 grid)
ordered_display_names = [DATASET_NAME_MAP.get(ds, ds) for ds in SELECTED_DATASETS] + ['Average']
stab_df['Dataset_Display'] = pd.Categorical(stab_df['Dataset_Display'], categories=ordered_display_names, ordered=True)

# ============================================================
# PLOTTING
# ============================================================
sns.set_theme(style="white")

g = sns.catplot(
    data=stab_df, 
    x='TargetClusters', 
    y='Clustering Stability', 
    hue='Method_Display', 
    hue_order=DISPLAY_ORDER,
    palette=PALETTE,
    col='Dataset_Display', 
    col_wrap=4, 
    kind='bar',
    height=3, 
    aspect=1.2,
    edgecolor='black',
    linewidth=0.8,
    errorbar=None
)

# Apply consistent aesthetic refinements across all subplots
for ax in g.axes.flat:
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

g.set_titles(col_template="{col_name}", weight='bold')
g.set_axis_labels(r'Target Clusters ($k$)', 'ARI Consensus')

# Format top legend dynamically based on active methods
sns.move_legend(
    g, 
    loc="lower center", 
    bbox_to_anchor=(0.5, 1.02), 
    ncol=len(DISPLAY_ORDER), 
    title="Method",
    frameon=False
)

plt.tight_layout()

OUTPUT_PATH = "/projectnb/ace-ig/enrique/unsupervised_clustering/_stab_UFS/biomedical/results/clustering_stability_results_plot.png"
g.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

# Print Summary Tables
print("=== Overall Average Clustering Stability by Method ===")
print(avg_df.groupby('Method_Display')['Clustering Stability'].mean().sort_values(ascending=False).reset_index().to_string(index=False))

print("\n=== Clustering Stability per Method by Dataset ===")
print(stab_df.groupby(['Dataset_Display', 'Method_Display'], observed=False)['Clustering Stability'].mean().unstack())