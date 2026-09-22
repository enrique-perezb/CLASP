import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ==========================================
# CONFIGURATION: Datasets & Methods
# ==========================================
SELECTED_DATASETS = [
    'ADHD200_cleaned', 
    'ALLAML_cleaned', 
    'brca_cleaned', 
    'TADPOLE_cleaned',
    'nci60_cleaned',
    'mecfs_cleaned',
    'GLI_85_cleaned'
]

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
    'ndfs':      {'name': 'NDFS',      'color': "#b1452a"},
    'cae':       {'name': 'CAE',       'color': '#e63946'},
    'groupfs':   {'name': 'GroupFS',   'color': '#800f2f'},
    'stacsfs':   {'name': 'CLASP',     'color': '#0077b6'}
}

# The order they will appear in the bar chart and legend
METHOD_ORDER = ['laplacian', 'ica', 'mcfs', 'spec', 'ndfs', 'cae', 'groupfs', 'stacsfs']

# Derived display tools for Seaborn
DISPLAY_ORDER = [METHOD_CONFIG[m]['name'] for m in METHOD_ORDER if m in METHOD_CONFIG]
PALETTE = {METHOD_CONFIG[m]['name']: METHOD_CONFIG[m]['color'] for m in METHOD_ORDER if m in METHOD_CONFIG}

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def parse_features(feat_str):
    """Extracts integers from string representation of selected features."""
    matches = re.findall(r'int64\((\d+)\)', str(feat_str))
    if not matches:
        matches = re.findall(r'\b\d+\b', str(feat_str).replace('64', ''))
    return [int(m) for m in matches]

def nogueira_stability(features_list, d):
    """Calculates Nogueira stability metric."""
    M = len(features_list)
    if M <= 1: 
        return np.nan
    
    k_bar = np.mean([len(f) for f in features_list])
    if k_bar == 0 or d == 0: 
        return np.nan
    
    freqs = {}
    for f_set in features_list:
        for feat in f_set:
            freqs[feat] = freqs.get(feat, 0) + 1
            
    sum_s_squared = 0
    for f_j in freqs.values():
        p_hat_j = f_j / M
        s_j_squared = (M / (M - 1)) * p_hat_j * (1 - p_hat_j)
        sum_s_squared += s_j_squared
        
    numerator = sum_s_squared / d
    denominator = (k_bar / d) * (1 - k_bar / d)
    
    if denominator == 0: 
        return np.nan
        
    return 1 - (numerator / denominator)

# ==========================================
# DATA PROCESSING
# ==========================================
FILE_PATH = '/projectnb/ace-ig/enrique/unsupervised_clustering/_stab_UFS/biomedical/results/dimsweep_features_k2.csv'
df = pd.read_csv(FILE_PATH)

# Filter datasets and methods
df = df[
    df['Dataset'].isin(SELECTED_DATASETS) & 
    df['Method'].isin(METHOD_CONFIG.keys())
].copy()

# Apply mappings
df['Dataset_Display'] = df['Dataset'].map(DATASET_NAME_MAP).fillna(df['Dataset'])
df['Method_Display'] = df['Method'].map(lambda x: METHOD_CONFIG.get(x, {}).get('name', x))

df['Selected_Features_List'] = df['Selected_Features'].apply(parse_features)

# Estimate total feature dimension 'd'
max_feats = {}
for ds in df['Dataset_Display'].unique():
    all_feats = [feat for feats in df[df['Dataset_Display'] == ds]['Selected_Features_List'] for feat in feats]
    max_feats[ds] = max(all_feats) + 1 if all_feats else 10000

# Compute stability scores
stability_records = []
for (dataset, method, k), group in df.groupby(['Dataset_Display', 'Method_Display', 'TargetClusters'], observed=False):
    feats = group['Selected_Features_List'].tolist()
    stab = nogueira_stability(feats, d=max_feats[dataset])
    stability_records.append({
        'Dataset': dataset,
        'Method': method,
        'TargetClusters': k,
        'Nogueira Stability': stab
    })

stab_df = pd.DataFrame(stability_records)

# Compute Average across datasets
avg_df = (
    stab_df.groupby(['Method', 'TargetClusters'], observed=False)['Nogueira Stability']
    .mean()
    .reset_index()
)
avg_df['Dataset'] = 'Average'

# Combine individual datasets and average
stab_df = pd.concat([stab_df, avg_df], ignore_index=True)

# Maintain strict grid ordering
ordered_display_names = [DATASET_NAME_MAP.get(ds, ds) for ds in SELECTED_DATASETS] + ['Average']
stab_df['Dataset'] = pd.Categorical(stab_df['Dataset'], categories=ordered_display_names, ordered=True)

# ==========================================
# PLOTTING
# ==========================================
sns.set_theme(style="white")

g = sns.catplot(
    data=stab_df, 
    x='TargetClusters', 
    y='Nogueira Stability', 
    hue='Method', 
    hue_order=DISPLAY_ORDER,
    palette=PALETTE,
    col='Dataset', 
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
    ax.set_ylim(0, 1.05)

g.set_titles(col_template="{col_name}", weight='bold')
g.set_axis_labels(r'Target Clusters ($k$)', 'Nogueira Stability')

# Format top legend
sns.move_legend(
    g, 
    "lower center", 
    bbox_to_anchor=(0.5, 1.02), 
    ncol=len(DISPLAY_ORDER), 
    title="Method",
    frameon=False
)

plt.tight_layout()

OUTPUT_PATH = '/projectnb/ace-ig/enrique/unsupervised_clustering/_stab_UFS/biomedical/results/nogueira_results.png'
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

# Print Summary Outputs
print("=== Overall Average Nogueira Stability by Method ===")
print(avg_df.groupby('Method')['Nogueira Stability'].mean().sort_values(ascending=False).reset_index().to_string(index=False))

print("\n=== Average Stability per Method by Dataset ===")
print(stab_df.groupby(['Dataset', 'Method'], observed=False)['Nogueira Stability'].mean().unstack())