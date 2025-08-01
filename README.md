# How Robust Are Language Models Against Backdoors in Federated Learning? Code

This repository provides the reference implementation for the experiments described in the paper **"How Robust Are Language Models Against Backdoors in Federated Learning?"**. It includes two main pipelines:

* **BERT-based experiments** (split into main and component modules)
* **GPT-2-based experiments** (single main script with optional arguments)

using Python 3.12.3, PyTorch 2.6.0.

---

## BERT Implementation

The BERT pipeline is organized into two files:

1. **BERT\_main.py**

   * Entry point for federated learning experiments using BERT.
   * All experiment settings (dataset, attack/defense flags, hyperparameters) are configured via global variables at the top of the script. You can modify:

     * `DATASET_NAME_CONFIG`, `MODEL_NAME_CONFIG`, `NUM_CLIENTS_CONFIG`, etc.
     * Attack parameters: `ATTACK_TYPE_CONFIG`, `TRIGGERS_LIST_CONFIG`, etc.
     * Defense flags: `USE_MULTI_KRUM_CONFIG`, `USE_NORM_CLIPPING_CONFIG`, `USE_DIFFERENTIAL_PRIVACY_CONFIG`, `USE_FLAME_CONFIG`.

2. **BERT\_components.py**

   * Helper modules for data loading, dataset definitions, client/server classes, and evaluation functions.
   * Key components:

     * `load_and_preprocess_data`, `TextClassificationDataset` for data handling.
     * `Client`, `MaliciousClient`, and `Server` classes implementing FL logic and defenses.
     * Utility functions: `evaluate_asr`, `CosSim`, `dp_noise`, etc.

### Running BERT Experiments

```bash
# Ensure you have installed required packages: transformers, torch, pandas, hdbscan, tqdm, etc.
python BERT_main.py
```

Modify the global configuration variables in `BERT_main.py` to switch datasets, attacks, and defenses before running.

---

## GPT-2 Implementation

The GPT-2 pipeline is contained in a single script **GPT\_main.py** that uses a combination of internal hyperparameters and optional command-line arguments. Key points:

* Core settings are defined in the `HYPERPARAMETERS` dictionary at the top of the file.
* You may override attack and defense options via CLI flags:

  * `--is_attack True|False`, `--attack_type baseline|neurotoxin`, etc.
  * `--is_defense True|False`, `--multi_krum`, `--norm_clipping`, `--dp`, `--flame`.

### Running GPT-2 Experiments

```bash
# Install dependencies: transformers, torch, datasets, hdbscan, tqdm, etc.
python GPT_main.py --is_attack True --attack_type baseline --is_defense True --multi_krum True
```

Use `python GPT_main.py --help` to see all available options.

---

## License & Citation

Please cite the original paper when using or extending this code:

> **How Robust Are Language Models Against Backdoors in Federated Learning?**

---
