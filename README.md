# CLASP: Software Implementation

Official implementation of **CLASP** (submitted for review at ICLR 2027).

---

## Overview

CLASP discovers sparse, reproducible feature sets by stochastically optimizing cluster separability across data subsamples, improving feature selection stability, cluster quality, and cluster stability.

---

## Repository Structure

```text
.
├── data/                               # Dataset files
│   ├── ADHD200_NYU_cleaned.csv
│   ├── ALLAML_cleaned.csv
│   ├── GLI_85_cleaned.csv
│   ├── TADPOLE_cleaned.csv
│   ├── brca_cleaned.csv
│   ├── mecfs_cleaned.csv
│   └── nci60_cleaned.csv
├── experiments/                        # Experiment and evaluation scripts
│   ├── biomedical/                     # Biomedical dataset experiments & plots
│   │   ├── biomedical_tests.py
│   │   ├── visualize_cluster_stability.py
│   │   ├── visualize_dbi.py
│   │   └── visualize_nogueira.py
│   ├── simulation/                     # Simulation dataset experiments & plots
│   │   ├── simulation_tests.py
│   │   └── visualize_sim_results.py
│   └── experiments_class.py            # Common experiment execution classes/utils
├── model/                              # Models and core architecture implementation
    └── featureselection_class.py       # CLASP feature selection implementation
    └── stability_class                 # CLASP loss function implementation

