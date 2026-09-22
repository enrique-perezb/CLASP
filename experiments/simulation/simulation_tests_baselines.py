import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Move working directory to the project root so imports resolve
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT)

# Import existing ExperimentalSuite class
from model.experiments_class import ExperimentalSuite


# ============================================================
# BENCHMARK RUNNER
# ============================================================
def run_2d_sweep_features_only(
    suite: ExperimentalSuite,
    rho_values: list,
    gamma_values: list,
    methods_to_run: list,
    base_n: int = 200,
    base_p: int = 2000,
    base_d: int = 20,
    base_delta: float = 2.5,
):
    print(f"\nStarting 2D Feature Extraction Sweep: rho={rho_values} | gamma={gamma_values}")
    print(f"Methods: {methods_to_run} | Repeats: {suite.n_repeats}")
    
    all_results = []
    
    # Fetch all methods and filter down to target list
    full_methods_map = suite.get_methods_map()
    methods_map = {m: full_methods_map[m] for m in methods_to_run if m in full_methods_map}

    for rho_val in rho_values:
        for gamma_val in gamma_values:
            print(f"Processing ρ={rho_val:.2f}, γ={gamma_val:.2f}")

            for rep in range(suite.n_repeats):
                seed = 1000 * rep + int(rho_val * 100) + int(gamma_val * 1000)
                eps_val = 0.0
                
                # Generate Data
                try:
                    X, y, true_features = suite.generate_data(
                        n=base_n, p=base_p, d=base_d, k=suite.k,
                        delta=base_delta, rho=rho_val, gamma=gamma_val,
                        eps=eps_val, noise_var=2.0, seed=seed
                    )
                except Exception as e:
                    print(f"  -> Data generation failed: {e}")
                    continue

                # Train / Test Split
                X_train, _, _, _ = train_test_split(
                    X, y, test_size=0.30, random_state=seed, stratify=y
                )

                # Format ground truth features as string for storage
                true_feat_str = ",".join(map(str, true_features))

                # Run target methods
                for method_name, method_fn in methods_map.items():
                    try:
                        result = method_fn(X=X_train, d=base_d, k=suite.k, seed=seed)
                        selected = result["selected_features"]
                        
                        # Store selection formatted as delimited string
                        selected_str = ",".join(map(str, selected))

                        all_results.append({
                            "method": method_name,
                            "rep": rep,
                            "rho": rho_val,
                            "gamma": gamma_val,
                            "seed": seed,
                            "true_features": true_feat_str,
                            "selected_features": selected_str,
                            "status": "success"
                        })

                    except Exception as e:
                        print(f"  -> FAILED: {method_name} (rep={rep}, ρ={rho_val}, γ={gamma_val}) - {e}")
                        all_results.append({
                            "method": method_name,
                            "rep": rep,
                            "rho": rho_val,
                            "gamma": gamma_val,
                            "seed": seed,
                            "true_features": true_feat_str,
                            "selected_features": "",
                            "status": f"failed: {str(e)}"
                        })

    df_new = pd.DataFrame(all_results)
    raw_path = os.path.join(suite.out_dir, "sweep_2d_selected_features.csv")

    # Save/Append output data
    if os.path.exists(raw_path):
        df_existing = pd.read_csv(raw_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(
            subset=["method", "rep", "rho", "gamma"], keep="last"
        )
    else:
        df_combined = df_new

    df_combined.to_csv(raw_path, index=False)
    print(f"\nSaved selected feature records to: {raw_path}")

    return df_combined


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    start_time = time.perf_counter()

    # n_repeats updated to 5
    suite = ExperimentalSuite(
        k=2, 
        n_repeats=5, 
        out_dir="/projectnb/ace-ig/enrique/unsupervised_clustering/_stab_UFS/simulation/results"
    )

    os.makedirs(suite.out_dir, exist_ok=True)

    print("Beginning feature collection experiment...")

    rho_sweep = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    gamma_sweep = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    target_methods = ["ica", "mcfs", "laplacian", "spec", "stacsfs", "cae", "groupfs"]

    run_2d_sweep_features_only(
        suite=suite,
        rho_values=rho_sweep,
        gamma_values=gamma_sweep,
        methods_to_run=target_methods
    )

    elapsed = time.perf_counter() - start_time
    print(f"\nALL EXPERIMENTS COMPLETE in {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {elapsed % 60:.2f}s")