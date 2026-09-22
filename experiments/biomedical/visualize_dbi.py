import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import davies_bouldin_score
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION: Paths, Datasets & Methods
# ============================================================
CSV_FILE = "/projectnb/ace-ig/enrique/unsupervised_clustering/_stab_UFS/biomedical/results/ksweep_features_k100_complete.csv"
DATA_DIR = "/projectnb/ace-ig/enrique/unsupervised_clustering/_stab_UFS/biomedical/data/clean_data/"

# 1. Specify the exact base names of the 7 datasets
SELECTED_DATASETS = [
    'ADHD200_cleaned', 
    'ALLAML_cleaned', 
    'brca_cleaned', 
    'TADPOLE_cleaned',
    'nci60_cleaned',
    'mecfs_cleaned',
    'GLI_85_cleaned'
]

# 2. Define clean display names for dataset titles
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

# ============================================================
# MAIN EXECUTION & DBI COMPUTATION
# ============================================================
# 1. Read and filter the results CSV
results_df = pd.read_csv(CSV_FILE)

# Filter for chosen datasets and configured methods
results_df = results_df[
    results_df['Dataset'].isin(SELECTED_DATASETS) & 
    results_df['Method'].isin(METHOD_CONFIG.keys())
].copy()

results_df['Selected_Features'] = results_df['Selected_Features'].apply(parse_int_list)
results_df['Validation_Indices'] = results_df['Validation_Indices'].apply(parse_int_list)

# Load only selected raw datasets into memory
raw_datasets = {}
for ds_name in SELECTED_DATASETS:
    filename = f"{ds_name}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        raw_datasets[ds_name] = pd.read_csv(filepath).values

dbi_records = []

# 2. Iterate through rows to evaluate DBI
for idx, row in results_df.iterrows():
    ds_name = row['Dataset']
    if ds_name not in raw_datasets:
        continue
        
    X = raw_datasets[ds_name]
    k = row['TargetClusters']
    method = row['Method']
    selected_features = row['Selected_Features']
    val_indices = row['Validation_Indices']
    
    if not selected_features or not val_indices:
        continue

    # Reconstruct training indices 
    all_indices = set(range(X.shape[0]))
    train_indices = list(all_indices - set(val_indices))
    
    # Scale and reduce
    scaler = StandardScaler()
    scaler.fit(X[train_indices])
    X_val_scaled = scaler.transform(X[val_indices])
    X_val_reduced = X_val_scaled[:, selected_features]
    
    # Route clustering
    if method in SPECTRAL_METHODS:
        clusterer = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=43)
    else:
        clusterer = KMeans(n_clusters=k, random_state=43)
        
    try:
        labels = clusterer.fit_predict(X_val_reduced)
        if len(set(labels)) > 1:
            dbi = davies_bouldin_score(X_val_reduced, labels)
        else:
            dbi = np.nan
    except Exception:
        dbi = np.nan
        
    dbi_records.append({
        'Dataset': ds_name,
        'Method': method,
        'TargetClusters': k,
        'DBI': dbi
    })

# 3. Format Display Names and Categorical Ordering
dbi_df = pd.DataFrame(dbi_records)

# Apply mappings
dbi_df['Dataset_Display'] = dbi_df['Dataset'].map(DATASET_NAME_MAP).fillna(dbi_df['Dataset'])
dbi_df['Method_Display'] = dbi_df['Method'].map(lambda x: METHOD_CONFIG.get(x, {}).get('name', x))

# Calculate average DBI across datasets per (Method, TargetClusters)
avg_df = (
    dbi_df.groupby(['Method_Display', 'TargetClusters'], observed=False)['DBI']
    .mean()
    .reset_index()
)
avg_df['Dataset_Display'] = 'Average'

# Append Average to main dataframe
dbi_df = pd.concat([dbi_df, avg_df], ignore_index=True)

# Maintain exact ordering so 'Average' lands as 8th panel (bottom right in 2x4 grid)
ordered_display_names = [DATASET_NAME_MAP.get(ds, ds) for ds in SELECTED_DATASETS] + ['Average']
dbi_df['Dataset_Display'] = pd.Categorical(dbi_df['Dataset_Display'], categories=ordered_display_names, ordered=True)

# ============================================================
# PLOTTING
# ============================================================
sns.set_theme(style="white")

g = sns.catplot(
    data=dbi_df, 
    x='TargetClusters', 
    y='DBI', 
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
g.set_axis_labels(r'Target Clusters ($k$)', 'Davies-Bouldin Index')

# Format top legend dynamically based on number of active methods
sns.move_legend(
    g, 
    loc="lower center", 
    bbox_to_anchor=(0.5, 1.02), 
    ncol=len(DISPLAY_ORDER), 
    title="Method",
    frameon=False
)

plt.tight_layout()

OUTPUT_PATH = "/projectnb/ace-ig/enrique/unsupervised_clustering/_stab_UFS/biomedical/results/dbi_results_plot.png"
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

# Print Summary Outputs
print("=== Overall Average Davies-Bouldin Index by Method ===")
print(avg_df.groupby('Method_Display')['DBI'].mean().sort_values().reset_index().to_string(index=False))

print("\n=== Davies-Bouldin Index per Method by Dataset ===")
print(dbi_df.groupby(['Dataset_Display', 'Method_Display'], observed=False)['DBI'].mean().unstack())