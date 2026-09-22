### Script to generate results used for Figs. 3, 4, 5. ###

# Necessary setup to ensure multi-core processing for CLASP
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from model.experiments_class import ExperimentalSuite

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
TARGET_DIM = 100
N_JOBS = 28
OUTPUT_DIR = ""  # FILL IN
OUTPUT_FEATURE_CSV = os.path.join(OUTPUT_DIR, "")  # FILL IN

N_SPLITS = 5
GLOBAL_RANDOM_STATE = 43

CSV_FILES = [
    "../data/ADHD200_NYU_cleaned.csv",
    "../data/ALLAML_cleaned.csv",
    "../data/brca_cleaned.csv",
    "../data/GLI_85_cleaned.csv",
    "../data/mecfs_cleaned.csv",
    "../data/nci60_cleaned.csv",
    "../data/ovarian_cleaned.csv",
    "../data/TADPOLE_cleaned.csv"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

METHODS = ["laplacian", "ica", "mcfs", "spec", "cae", "groupfs", "clasp"]

# ============================================================
# MAIN EXECUTION
# ============================================================

if os.path.exists(OUTPUT_FEATURE_CSV):
    feature_results = pd.read_csv(OUTPUT_FEATURE_CSV).to_dict('records')
    print(f"Loaded {len(feature_results)} existing feature records.")
else:
    feature_results = []

k_vals = [2, 3, 4, 5]

for k_val in k_vals:
    # Initialize the ExperimentalSuite for the current k
    suite = ExperimentalSuite(k=k_val)
    method_map = suite.get_methods_map()
    
    for dataset_counter, filepath in enumerate(CSV_FILES):
        
        if not os.path.exists(filepath):
            print(f"File not found, skipping: {filepath}")
            continue

        dataset_name = os.path.basename(filepath).replace(".csv", "")
        print(f"\nProcessing dataset: {dataset_name} (k={k_val})")

        df = pd.read_csv(filepath)
        X = df.values

        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=GLOBAL_RANDOM_STATE)

        for fold_idx, (train_index, val_index) in enumerate(kf.split(X), start=1):
            print(f"  Fold {fold_idx}/{N_SPLITS}")

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_index])

            fold_seed = GLOBAL_RANDOM_STATE + 1000 * dataset_counter + fold_idx

            for method_name in METHODS:
                print(f"    Evaluating {method_name}")

                print("Time start: ", datetime.now())
                
                # Execute feature selection dynamically via the suite mapping
                try:
                    if method_name == "clasp":
                        fit_result = method_map[method_name](X_train, d=TARGET_DIM, k=k_val, seed=fold_seed, n_jobs=N_JOBS)
                    else:
                        fit_result = method_map[method_name](X_train, d=TARGET_DIM, k=k_val, seed=fold_seed)
                except Exception as e:
                    print(f"      [!] Error running {method_name}: {e}")
                    continue

                selected = fit_result["selected_features"]
                
                # Save the raw selected features for this fold
                feature_results.append({
                    "Dataset": dataset_name,
                    "Fold": fold_idx,
                    "Method": method_name,
                    "TargetClusters": k_val,
                    "Selected_Features": list(selected),
                    "Validation_Indices": list(val_index)
                })

                print("End time: ", datetime.now())

        pd.DataFrame(feature_results).to_csv(OUTPUT_FEATURE_CSV, index=False)
        print(f"--- Saved progress for {dataset_name} ---")

print("\nAll datasets processed successfully.")
print(f"Feature Selections saved to: {OUTPUT_FEATURE_CSV}")