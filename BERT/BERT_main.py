import copy
import os
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset
import random
import numpy as np
from torch.utils.data import Subset
import itertools
from tqdm.auto import tqdm
import torch.nn.functional as F
import time

# --- Import required classes and functions from MK_components_ver3.py ---
from BERT_components import (
    load_and_preprocess_data,
    TextClassificationDataset,
    Client,
    MaliciousClient, 
    Server,
    evaluate_asr,
    _load_sst2_specific,
    _load_agnews_specific,
    CosSim,
    dp_noise
)

# -----------------------------------------------------------------------------
# --- Experiment Settings (Modify this section to change overall experiment configuration) ---
# -----------------------------------------------------------------------------
# 0. Select the dataset to use ("SST-2" or "AG_NEWS")
DATASET_NAME_CONFIG = "SST-2"
# 1. Basic federated learning and model setting
MODEL_NAME_CONFIG = "bert-base-uncased"
NUM_CLIENTS_CONFIG = 100  
CLIENTS_PER_ROUND_CONFIG = 10  
EPOCHS_PER_CLIENT_CONFIG = 5  
BATCH_SIZE_CONFIG = 16
LEARNING_RATE_CONFIG = 2e-5 # Benign Client Learning Rate
AGGREGATION_ROUNDS_CONFIG = 50
GLOBAL_SEED = 41

# 2. Attack settings
ATTACK_MODE_ENABLED_CONFIG = True
MALICIOUS_CLIENT_ID_CONFIG = 0
MALICIOUS_CLIENT_LR_CONFIG = 2e-4
# ATTACK_TYPE_CONFIG: 1 for BadNets-like, 2 for RIPPLES, 3 for BGM (pre-poisoned data)
ATTACK_TYPE_CONFIG = 1

SAMPLES_PER_CLIENT_CONFIG = 69

# Setup for BadNets/RIPPLES (TRIGGERS_LIST_CONFIG is not used inside Malicious Client on BGM)
TRIGGER_TEXT_CONFIG = None
TRIGGERS_LIST_CONFIG = ["cf", "mn", "bb", "tq"] 
NUM_TRIGGERS_PER_SAMPLE_CONFIG = 1 
ATTACK_TARGET_LABEL_CONFIG = 0

# Configuring for RIPPLES
RIPPLES_CLEAN_PART_FRACTION_CONFIG = 0.3

# --- BGM attack setting ---
BGM_PREPOISONED_DATA_PATH_CONFIG = './sst-2/poison_data/sst-2_chatgpt/train.tsv' 
BGM_ASR_TEST_PATH_CONFIG = './sst-2/poison_data/sst-2_chatgpt/test.tsv'

# --- attack scheduling hyperparameters ---
ATTACK_START_ROUND_CONFIG = 10  # Attack start round
ATTACK_END_ROUND_CONFIG = 30    # End of Attack Round (extended duration of BGM attack)
MALICIOUS_CLIENT_BEHAVIOR_AFTER_ATTACK_CONFIG = "BENIGN"

# 3. Setting up defense techniques 
# =======================================================================
# Selection of key defense techniques (FLAME vs Multi-Krum)
# =======================================================================
USE_FLAME_CONFIG = False        # Whether to use FLAME defense (automatically deactivate other defense techniques when enabled)
USE_MULTI_KRUM_CONFIG = True    # Whether to use Multi-Krum defense (disable for DP alone experiment)

# Multi-Krum detailed settings (applies only when USE_MULTI_KRUM_CONFIG=True)
NUM_ADVERSARIES_FOR_KRUM_CONFIG = 1
KRUM_VARIANT_CONFIG = "NORMAL_MULTI_KRUM"  # "NORMAL_MULTI_KRUM"
KRUM_NEIGHBORS_TO_SAMPLE_CONFIG = 3 

# FLAME detailed settings (applies only when USE_FLAME_CONFIG=True)
FLAME_LAMBDA_CONFIG = 0.01        # Lambda (λ) - FLAME noise scaling parameter

# =======================================================================
# Set up auxiliary defense techniques (independently controllable)
# 적용 순서: Multi-Krum → Norm Clipping → Differential Privacy
# Caution: The following settings are automatically disabled when using FLAME
# =======================================================================

# Norm clipping settings (available with Multi-Krum or alone)
USE_NORM_CLIPPING_CONFIG = False #Norm Clipping Defense Enabled (Disabled for DP Exclusive Experiment)
NORM_BOUND_FOR_CLIPPING_CONFIG = 3.5 #NormalClippingThreshold

# Differential Privacy Settings (Multi-Krum, Available with Norm Clipping or Alone)
USE_DIFFERENTIAL_PRIVACY_CONFIG = False #DP Defense Use Only (DP Exclusive Experiment)
DP_SIGMA_CONFIG = 0.003            # Sigma (σ) - DP noise scaling parameter

# =======================================================================
# Examples of combinations of defense techniques:
# 1. FLAME ONLY: USE_FLAME_CONFIG=True, all others False
# 2. Multi-Krum ONLY: USE_MULTI_KRUM_CONFIG=True, USE_FLAME_CONFIG=False, else False
# 3. Multi-Krum + Norm Clipping: USE_MULTI_KRUM_CONFIG=True, USE_NORM_CLIPPING_CONFIG=True
# 4. Multi-Krum + DP: USE_MULTI_KRUM_CONFIG=True, USE_DIFFERENTIAL_PRIVACY_CONFIG=True
# 5. Multi-Krum + Norm Clipping + DP: All three of the above are True
# 6. Norm clipping alone: USE_NORM_CLIPPING_CONFIG=True, all others false
# 7. DP Only: USE_DIFFERENTIAL_PRIVACY_CONFIG=True, All Others False
# 8. Norm clipping + DP: The above two True, the remaining False
# 9. FedAvg Only: All Defence Techniques False
# =======================================================================

# 6. Setting existing path (for BadNets/RIPPLES)
SST2_ASR_TEST_PATH = './sst-2/poison_data/asr_test_filtered.tsv' 
SST2_RIPPLES_LAMBDA_CONFIG = 0.5
AGNEWS_ASR_TEST_PATH = './ag_news_csv/poison_data/asr_test_ripples_agnews_filtered.tsv' 
AGNEWS_RIPPLES_LAMBDA_CONFIG = 0.5

# --- Assign actual variables based on configuration values ---
DATASET_NAME = DATASET_NAME_CONFIG; MODEL_NAME = MODEL_NAME_CONFIG
NUM_CLIENTS = NUM_CLIENTS_CONFIG; CLIENTS_PER_ROUND = CLIENTS_PER_ROUND_CONFIG
EPOCHS_PER_CLIENT = EPOCHS_PER_CLIENT_CONFIG; BATCH_SIZE = BATCH_SIZE_CONFIG
LEARNING_RATE = LEARNING_RATE_CONFIG; AGGREGATION_ROUNDS = AGGREGATION_ROUNDS_CONFIG
VALIDATION_SPLIT = 0.0

if DATASET_NAME == "SST-2": DATA_PATH = './sst-2'
elif DATASET_NAME == "AG_NEWS": DATA_PATH = './ag_news_csv'
else: raise ValueError(f"Unsupported DATASET_NAME: {DATASET_NAME}")

# =======================================================================
# Configuring Improved Defense Techniques
# =======================================================================
# FLAME settings (standalone defense)
USE_FLAME = USE_FLAME_CONFIG
FLAME_LAMBDA = FLAME_LAMBDA_CONFIG

# Multi-Krum settings (standalone defense)
USE_MULTI_KRUM = USE_MULTI_KRUM_CONFIG
NUM_ADVERSARIES_FOR_KRUM = NUM_ADVERSARIES_FOR_KRUM_CONFIG
KRUM_VARIANT = KRUM_VARIANT_CONFIG
KRUM_NEIGHBORS_TO_SAMPLE = KRUM_NEIGHBORS_TO_SAMPLE_CONFIG

# Set up perturbation defense techniques (can be combined with Multi-Krum)
USE_NORM_CLIPPING = USE_NORM_CLIPPING_CONFIG
USE_DIFFERENTIAL_PRIVACY = USE_DIFFERENTIAL_PRIVACY_CONFIG

# Defense Techniques Interaction Processing
if USE_FLAME and USE_MULTI_KRUM:
    print("WARNING: Both FLAME and Multi-Krum are enabled. FLAME will take precedence.")
    USE_MULTI_KRUM = False  # Multi-Krum disabled when FLAME is enabled
    
if USE_FLAME and (USE_NORM_CLIPPING or USE_DIFFERENTIAL_PRIVACY):
    print("WARNING: FLAME is enabled. Norm clipping and differential privacy will be handled internally by FLAME.")
    # Deactivate external defense techniques as they are handled inside the FLAME
    USE_NORM_CLIPPING = False
    USE_DIFFERENTIAL_PRIVACY = False

NORM_BOUND_FOR_CLIPPING = NORM_BOUND_FOR_CLIPPING_CONFIG
DP_SIGMA = DP_SIGMA_CONFIG

ATTACK_MODE_ENABLED = ATTACK_MODE_ENABLED_CONFIG
MALICIOUS_CLIENT_ID = MALICIOUS_CLIENT_ID_CONFIG
MALICIOUS_CLIENT_LR = MALICIOUS_CLIENT_LR_CONFIG
ATTACK_TYPE = ATTACK_TYPE_CONFIG
SAMPLES_PER_CLIENT = SAMPLES_PER_CLIENT_CONFIG

TRIGGER_TEXT = TRIGGER_TEXT_CONFIG
TRIGGERS_LIST = TRIGGERS_LIST_CONFIG
NUM_TRIGGERS_PER_SAMPLE = NUM_TRIGGERS_PER_SAMPLE_CONFIG
ATTACK_TARGET_LABEL = ATTACK_TARGET_LABEL_CONFIG
RIPPLES_CLEAN_PART_FRACTION = RIPPLES_CLEAN_PART_FRACTION_CONFIG

BGM_PREPOISONED_DATA_PATH = BGM_PREPOISONED_DATA_PATH_CONFIG
BGM_ASR_TEST_PATH = BGM_ASR_TEST_PATH_CONFIG

ATTACK_START_ROUND = ATTACK_START_ROUND_CONFIG
ATTACK_END_ROUND = ATTACK_END_ROUND_CONFIG
MALICIOUS_CLIENT_BEHAVIOR_AFTER_ATTACK = MALICIOUS_CLIENT_BEHAVIOR_AFTER_ATTACK_CONFIG

AGNEWS_ASR_TEST_PATH = AGNEWS_ASR_TEST_PATH
SST2_RIPPLES_LAMBDA = SST2_RIPPLES_LAMBDA_CONFIG
AGNEWS_RIPPLES_LAMBDA = AGNEWS_RIPPLES_LAMBDA_CONFIG

# =======================================================================
# Defense Techniques Configuration Information Output
# # =======================================================================
print("\n" + "="*80)
print("DEFENSE MECHANISM CONFIGURATION")
print("="*80)

defense_methods = []
if USE_FLAME:
    defense_methods.append(f"FLAME (λ={FLAME_LAMBDA})")
if USE_MULTI_KRUM:
    defense_methods.append(f"Multi-Krum ({KRUM_VARIANT}, f={NUM_ADVERSARIES_FOR_KRUM})")
if USE_NORM_CLIPPING:
    defense_methods.append(f"Norm Clipping (bound={NORM_BOUND_FOR_CLIPPING})")
if USE_DIFFERENTIAL_PRIVACY:
    defense_methods.append(f"Differential Privacy (σ={DP_SIGMA})")

if defense_methods:
    print("Active Defense Methods:")
    for i, method in enumerate(defense_methods, 1):
        print(f"  {i}. {method}")
    
    # Defense Techniques Combination Description
    if USE_FLAME:
        print("\nDefense Strategy: FLAME (Standalone)")
        print("  - FLAME handles clustering, selection, and privacy internally")
    elif USE_MULTI_KRUM:
        if USE_NORM_CLIPPING or USE_DIFFERENTIAL_PRIVACY:
            sequence = ["Multi-Krum"]
            if USE_NORM_CLIPPING:
                sequence.append("Norm Clipping")
            if USE_DIFFERENTIAL_PRIVACY:
                sequence.append("Differential Privacy")
            print(f"\nDefense Strategy: Sequential ({' → '.join(sequence)})")
        else:
            print("\nDefense Strategy: Multi-Krum (Standalone)")
    else:
        if USE_NORM_CLIPPING and USE_DIFFERENTIAL_PRIVACY:
            print("\nDefense Strategy: Norm Clipping → Differential Privacy")
        elif USE_NORM_CLIPPING:
            print("\nDefense Strategy: Norm Clipping (Standalone)")
        elif USE_DIFFERENTIAL_PRIVACY:
            print("\nDefense Strategy: Differential Privacy (Standalone)")
else:
    print("Active Defense Methods: None (Baseline FedAvg)")

print("="*80 + "\n")

# --- Fix random seeds ---
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(GLOBAL_SEED)

# --- Conduct experiment and measure time ---
def main():
    with torch.amp.autocast('cuda', enabled=False):
        overall_experiment_start_time = time.time()
        
        print(f"=== Federated Learning Experiment with Defense Mechanisms ===")
        print(f"Dataset: {DATASET_NAME}, Model: {MODEL_NAME}")
        print(f"Clients: {NUM_CLIENTS}, Clients per round: {CLIENTS_PER_ROUND}")
        print(f"Attack enabled: {ATTACK_MODE_ENABLED}, Attack type: {ATTACK_TYPE}")
        
        # Defense Techniques Status Output
        defense_info = []
        if USE_FLAME:
            defense_info.append("FLAME")
        if USE_MULTI_KRUM:
            defense_info.append(f"Multi-Krum({KRUM_VARIANT})")
        if USE_NORM_CLIPPING:
            defense_info.append(f"Norm-Clipping({NORM_BOUND_FOR_CLIPPING})")
        if USE_DIFFERENTIAL_PRIVACY:
            defense_info.append(f"DP(σ={DP_SIGMA})")
        
        if not defense_info:
            defense_info.append("FedAvg-only")
        
        print(f"Defense methods: {', '.join(defense_info)}")
        print(f"Aggregation rounds: {AGGREGATION_ROUNDS}")
        print("=" * 60)

    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    train_df, num_labels_from_data = load_and_preprocess_data(DATASET_NAME, DATA_PATH, 'train')
    test_df, _ = load_and_preprocess_data(DATASET_NAME, DATA_PATH, 'test')
    
    if train_df.empty:
        print(f"Error: Training data is empty for {DATASET_NAME} at {DATA_PATH}")
        return
    if test_df.empty:
        print(f"Warning: Test data is empty for {DATASET_NAME} at {DATA_PATH}")

    print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
    print(f"Number of labels: {num_labels_from_data}")

    # 2. Initialize tokenizer and model
    print("Initializing tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 3. Initialize the server
    server = Server(
        model_name_param=MODEL_NAME,
        num_labels_for_model=num_labels_from_data,
        device_param=device,
        use_multi_krum_param=USE_MULTI_KRUM,
        num_adversaries_for_krum_param=NUM_ADVERSARIES_FOR_KRUM,
        krum_variant_param=KRUM_VARIANT,
        krum_sampling_neighbor_count_param=KRUM_NEIGHBORS_TO_SAMPLE,
        use_norm_clipping_param=USE_NORM_CLIPPING,
        norm_bound_param=NORM_BOUND_FOR_CLIPPING,
        server_update_lr_param=1.0,
        use_dp_param=USE_DIFFERENTIAL_PRIVACY,
        dp_sigma_param=DP_SIGMA,  # Sigma (σ) parameter
        use_flame_param=USE_FLAME,
        flame_noise_lambda_param=FLAME_LAMBDA  # Lambda (λ) parameter
    )

    # 4. Prepare client data
    print("Preparing client data...")
    
    # Data Segmentation for Common Clients
    train_df_shuffled = train_df.sample(frac=1, random_state=GLOBAL_SEED).reset_index(drop=True)
    actual_samples_per_client = min(SAMPLES_PER_CLIENT, len(train_df_shuffled) // NUM_CLIENTS)
    
    if actual_samples_per_client < SAMPLES_PER_CLIENT:
        print(f"Warning: Not enough training data. Using {actual_samples_per_client} samples per client.")

    # Preparing data for malicious clients (if an attack is enabled)
    attacker_base_df = pd.DataFrame()
    if ATTACK_MODE_ENABLED and ATTACK_TYPE != 0:
        print(f"🔍 DEBUG: Preparing attacker data - ATTACK_MODE_ENABLED={ATTACK_MODE_ENABLED}, ATTACK_TYPE={ATTACK_TYPE}")
        if ATTACK_TYPE == 3:  # BGM
            print(f"🔍 DEBUG: BGM attack preparation - BGM_PREPOISONED_DATA_PATH={BGM_PREPOISONED_DATA_PATH}")
            print(f"🔍 DEBUG: Path exists: {os.path.exists(BGM_PREPOISONED_DATA_PATH) if BGM_PREPOISONED_DATA_PATH else False}")
            
            if BGM_PREPOISONED_DATA_PATH and os.path.exists(BGM_PREPOISONED_DATA_PATH):
                try:
                    print(f"   Loading full BGM dataset for random sampling of {SAMPLES_PER_CLIENT} samples")
                    
                    df_full = pd.read_csv(BGM_PREPOISONED_DATA_PATH, sep='\t', header=0)
                    print(f"🔍 DEBUG: Loaded BGM data shape: {df_full.shape}")
                    
                    if DATASET_NAME == "SST-2":
                        attacker_base_df_full = df_full.iloc[:, [0, 1]].copy()
                        attacker_base_df_full.columns = ['sentence', 'label']
                    elif DATASET_NAME == "AG_NEWS":
                        attacker_base_df_full = df_full.iloc[:, [0, 1]].copy()
                        attacker_base_df_full.columns = ['text', 'label']
                    
                    total_samples = len(attacker_base_df_full)
                    print(f"   Total available samples in BGM dataset: {total_samples}")
                    
                    if total_samples >= SAMPLES_PER_CLIENT:
                        attacker_base_df = attacker_base_df_full.sample(
                            n=SAMPLES_PER_CLIENT, 
                            random_state=GLOBAL_SEED
                        ).reset_index(drop=True)
                        print(f"Randomly sampled {len(attacker_base_df)} BGM poisoned samples for malicious client using seed {GLOBAL_SEED}")
                    else:
                        attacker_base_df = attacker_base_df_full
                        print(f"Requested {SAMPLES_PER_CLIENT} samples, but only {total_samples} available. Using all available samples.")
                    
                    print(f"🔍 DEBUG: Final attacker_base_df shape: {attacker_base_df.shape}")
                    print(f"🔍 DEBUG: attacker_base_df.empty: {attacker_base_df.empty}")
                        
                except Exception as e:
                    print(f"❌ Error loading BGM data: {e}")
                    import traceback
                    traceback.print_exc()
                    attacker_base_df = pd.DataFrame()
            else:
                print(f"❌ BGM data path not found or invalid: {BGM_PREPOISONED_DATA_PATH}")
        else:  # BadNets or RIPPLES
            # Use a subset of training data for the attacker
            attacker_base_df = train_df_shuffled.iloc[:SAMPLES_PER_CLIENT].copy()
    
    print(f"🔍 DEBUG: Final attacker_base_df status - empty: {attacker_base_df.empty}, shape: {attacker_base_df.shape if not attacker_base_df.empty else 'N/A'}")

    # 5. Loading ASR test data (if an attack is enabled)
    backdoor_test_loader = None
    if ATTACK_MODE_ENABLED and ATTACK_TYPE != 0:
        asr_test_path = None
        if ATTACK_TYPE == 3:  # BGM
            asr_test_path = BGM_ASR_TEST_PATH
        elif DATASET_NAME == "SST-2":
            asr_test_path = SST2_ASR_TEST_PATH
        elif DATASET_NAME == "AG_NEWS":
            asr_test_path = AGNEWS_ASR_TEST_PATH

        if asr_test_path and os.path.exists(asr_test_path):
            try:
                if DATASET_NAME == "SST-2":
                    df_asr_test = _load_sst2_specific(asr_test_path)
                elif DATASET_NAME == "AG_NEWS":
                    df_asr_test = _load_agnews_specific(asr_test_path)
                
                if not df_asr_test.empty:
                    backdoor_test_dataset = TextClassificationDataset(
                        df_asr_test, tokenizer, text_col_name=0, label_col_name=1, has_trigger_flag=True
                    )
                    backdoor_test_loader = DataLoader(backdoor_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
                    print(f"Loaded {len(df_asr_test)} samples for ASR evaluation")
            except Exception as e:
                print(f"Error loading ASR test data: {e}")

    # 6. Preparing a Test Dataset
    test_dataset = TextClassificationDataset(test_df, tokenizer, text_col_name=0, label_col_name=1)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 7. Run federated learning
    print(f"\nStarting federated learning for {AGGREGATION_ROUNDS} rounds...")
    results = []
    total_aggregation_time_all_rounds = 0.0
    
    # CSV file path preset (for round-by-round updates)
    # Settings for defense technique combination display
    defense_components = []
    if USE_FLAME:
        defense_components.append("FLAME")
    if USE_MULTI_KRUM:
        defense_components.append(f"MultiKrum({KRUM_VARIANT})")
    if USE_NORM_CLIPPING:
        defense_components.append("NormClip")
    if USE_DIFFERENTIAL_PRIVACY:
        defense_components.append("DP")
    
    defense_method = "+".join(defense_components) if defense_components else "FedAvg"
    
    # Create a Detailed File Name
    attack_suffix = ""
    if ATTACK_MODE_ENABLED and ATTACK_TYPE != 0:
        attack_type_names = {1: "BadNets", 2: "RIPPLES", 3: "BGM"}
        attack_name = attack_type_names.get(ATTACK_TYPE, f"Type{ATTACK_TYPE}")
        
        schedule_info = f"sch{ATTACK_START_ROUND}to{ATTACK_END_ROUND}"
        behavior_short = MALICIOUS_CLIENT_BEHAVIOR_AFTER_ATTACK[:3].lower()  # "BENIGN" -> "ben", "ABSENT" -> "abs"
        
        if ATTACK_TYPE == 1:  # BadNets
            trigger_info = f"trig{len(TRIGGERS_LIST)}x{NUM_TRIGGERS_PER_SAMPLE}" if TRIGGERS_LIST else "notrig"
            attack_suffix = f"_attack{attack_name}_target{ATTACK_TARGET_LABEL}_{trigger_info}_{schedule_info}_{behavior_short}"
        elif ATTACK_TYPE == 2:  # RIPPLES
            ripples_lambda = SST2_RIPPLES_LAMBDA if DATASET_NAME == "SST-2" else AGNEWS_RIPPLES_LAMBDA
            trigger_info = f"trig{len(TRIGGERS_LIST)}x{NUM_TRIGGERS_PER_SAMPLE}" if TRIGGERS_LIST else "notrig"
            attack_suffix = f"_attack{attack_name}_target{ATTACK_TARGET_LABEL}_{trigger_info}_cleanFrac{RIPPLES_CLEAN_PART_FRACTION}_lambda{ripples_lambda}_{schedule_info}_{behavior_short}"
        elif ATTACK_TYPE == 3:  # BGM
            attack_suffix = f"_attack{attack_name}_target{ATTACK_TARGET_LABEL}_prepoisoned_{schedule_info}_{behavior_short}"
        
        attack_suffix += f"_malID{MALICIOUS_CLIENT_ID}_malLR{MALICIOUS_CLIENT_LR:.0e}"
    else:
        attack_suffix = "_noAttack"
    
    defense_suffix = ""
    if USE_FLAME:
        defense_suffix = f"_FLAME_lambda{FLAME_LAMBDA:.0e}"
    elif USE_MULTI_KRUM:
        defense_suffix = f"_MultiKrum_{KRUM_VARIANT}_f{NUM_ADVERSARIES_FOR_KRUM}_k{KRUM_NEIGHBORS_TO_SAMPLE}"
        if USE_NORM_CLIPPING:
            defense_suffix += f"_NormClip{NORM_BOUND_FOR_CLIPPING}"
        if USE_DIFFERENTIAL_PRIVACY:
            defense_suffix += f"_DP{DP_SIGMA:.0e}"
    else:
        defense_parts = []
        if USE_NORM_CLIPPING:
            defense_parts.append(f"NormClip{NORM_BOUND_FOR_CLIPPING}")
        if USE_DIFFERENTIAL_PRIVACY:
            defense_parts.append(f"DP{DP_SIGMA:.0e}")
        
        if defense_parts:
            defense_suffix = f"_{'_'.join(defense_parts)}"
        else:
            defense_suffix = "_FedAvg"
    
    exp_config_suffix = f"_r{AGGREGATION_ROUNDS}_cpr{CLIENTS_PER_ROUND}_spc{SAMPLES_PER_CLIENT}_e{EPOCHS_PER_CLIENT}_lr{LEARNING_RATE:.0e}_bs{BATCH_SIZE}_seed{GLOBAL_SEED}"
    
    base_result_dir = f'result_ds-{DATASET_NAME}_model-{MODEL_NAME.split("/")[-1]}'
    os.makedirs(base_result_dir, exist_ok=True)
    
    results_filename = f'FL_results{attack_suffix}{defense_suffix}{exp_config_suffix}.csv'
    results_save_path = os.path.join(base_result_dir, results_filename)
    
    print(f"📁 Results will be saved incrementally to: {results_save_path}")
    
    for round_num in range(AGGREGATION_ROUNDS):
        current_round_one_indexed = round_num + 1
        round_start_time = time.time()
        
        print(f"\n=== Round {current_round_one_indexed}/{AGGREGATION_ROUNDS} ===")
        
        clients_for_round_instances = []
        selected_client_original_ids_for_round = []
        
        malicious_client_is_attacking_this_round = False
        
        if ATTACK_MODE_ENABLED and ATTACK_TYPE != 0:
            if ATTACK_START_ROUND <= current_round_one_indexed <= ATTACK_END_ROUND:
                malicious_client_is_attacking_this_round = True
                print(f"   🔴 Client {MALICIOUS_CLIENT_ID} scheduled to ATTACK this round (Type: {ATTACK_TYPE})")
            else:
                print(f"   🔵 Client {MALICIOUS_CLIENT_ID} not attacking this round (will participate through random selection if chosen)")

        print(f"🔍 DEBUG Round {current_round_one_indexed}: malicious_client_is_attacking_this_round={malicious_client_is_attacking_this_round}")
        print(f"🔍 DEBUG Round {current_round_one_indexed}: attacker_base_df.empty={attacker_base_df.empty}")
        print(f"🔍 DEBUG Round {current_round_one_indexed}: ATTACK_TYPE={ATTACK_TYPE}")
        
        if malicious_client_is_attacking_this_round and not attacker_base_df.empty:
            print(f"   ✅ Creating attacking client {MALICIOUS_CLIENT_ID} with {len(attacker_base_df)} poisoned samples")
            
            # In the case of BGM attack, the column name is different ('sentence' and 'label')
            if ATTACK_TYPE == 3:  # BGM
                text_col_name = 'sentence' if DATASET_NAME == "SST-2" else 'text'
                label_col_name = 'label'
            else:  # BadNets, RIPPLES
                text_col_name = 0
                label_col_name = 1
            
            mal_client_obj = MaliciousClient(
                client_id=MALICIOUS_CLIENT_ID,
                attacker_base_df_for_malicious=attacker_base_df,
                tokenizer_for_mal_client=tokenizer,
                batch_size_param=BATCH_SIZE,
                lr_param=MALICIOUS_CLIENT_LR,
                epochs_param=EPOCHS_PER_CLIENT,
                device_to_use=device,
                global_seed_param=GLOBAL_SEED + current_round_one_indexed + MALICIOUS_CLIENT_ID,
                is_ripples_attack=(ATTACK_TYPE == 2),
                ripples_clean_part_fraction=RIPPLES_CLEAN_PART_FRACTION,
                ripples_lambda=SST2_RIPPLES_LAMBDA if DATASET_NAME == "SST-2" else AGNEWS_RIPPLES_LAMBDA,
                trigger_text_param=TRIGGER_TEXT,
                num_triggers_per_sample_param=NUM_TRIGGERS_PER_SAMPLE,
                triggers_list_param=TRIGGERS_LIST,
                attack_target_label_param=ATTACK_TARGET_LABEL,
                text_col_name=text_col_name,
                label_col_name=label_col_name,
                is_data_pre_poisoned=(ATTACK_TYPE == 3)
            )
            
            print(f"🔍 DEBUG: mal_client_obj.actual_samples_prepared={mal_client_obj.actual_samples_prepared}")
            if mal_client_obj.actual_samples_prepared > 0:
                clients_for_round_instances.append(mal_client_obj)
                selected_client_original_ids_for_round.append(MALICIOUS_CLIENT_ID)
                print(f"   💀 ATTACK: Malicious client {MALICIOUS_CLIENT_ID} FORCIBLY ADDED to round {current_round_one_indexed}")
            else:
                print(f"   ❌ Warning: Malicious client {MALICIOUS_CLIENT_ID} has no prepared samples!")
        elif malicious_client_is_attacking_this_round and attacker_base_df.empty:
            print(f"   ❌ ERROR: Attack scheduled but no poisoned data available!")
        elif malicious_client_is_attacking_this_round:
            print(f"   ❌ ERROR: Unexpected condition - attacking but unknown reason for not creating client!")

        num_benign_clients_needed = CLIENTS_PER_ROUND - len(clients_for_round_instances)
        
        already_selected_ids = set(selected_client_original_ids_for_round)
        candidate_client_ids = [i for i in range(NUM_CLIENTS) if i not in already_selected_ids]
        
        if num_benign_clients_needed > 0 and len(candidate_client_ids) > 0:
            selected_benign_ids = random.sample(
                candidate_client_ids, 
                min(num_benign_clients_needed, len(candidate_client_ids))
            )
        else:
            selected_benign_ids = []

        for client_id in selected_benign_ids:
            start_idx = client_id * actual_samples_per_client
            end_idx = min((client_id + 1) * actual_samples_per_client, len(train_df_shuffled))
            client_data_df = train_df_shuffled.iloc[start_idx:end_idx].copy()
            
            if not client_data_df.empty:
                client_dataset = TextClassificationDataset(
                    client_data_df, tokenizer, text_col_name=0, label_col_name=1
                )
                
                benign_client_obj = Client(
                    client_id=client_id,
                    dataset_obj=client_dataset,
                    batch_size_param=BATCH_SIZE,
                    learning_rate_param=LEARNING_RATE,
                    epochs_param=EPOCHS_PER_CLIENT,
                    validation_split_param=VALIDATION_SPLIT,
                    global_seed_param=GLOBAL_SEED + current_round_one_indexed + client_id,
                    tokenizer_for_client=tokenizer,
                    device_to_use=device
                )
                
                clients_for_round_instances.append(benign_client_obj)
                selected_client_original_ids_for_round.append(client_id)
                
                if client_id == MALICIOUS_CLIENT_ID:
                    print(f"   🎲 Client {MALICIOUS_CLIENT_ID} randomly selected as BENIGN participant")

        print(f"   Selected {len(clients_for_round_instances)} clients for this round")
        print(f"   📋 Selected client IDs: {sorted(selected_client_original_ids_for_round)}")
        
        if MALICIOUS_CLIENT_ID in selected_client_original_ids_for_round:
            if malicious_client_is_attacking_this_round:
                print(f"   ⚠️  MALICIOUS CLIENT {MALICIOUS_CLIENT_ID} is PARTICIPATING and ATTACKING!")
            else:
                print(f"   ✅ Malicious client {MALICIOUS_CLIENT_ID} participating as benign (randomly selected or post-attack)")
        else:
            print(f"   ❌ Malicious client {MALICIOUS_CLIENT_ID} is NOT participating this round")
        
        # Perform local training
        client_model_states_cpu = []
        for client_obj in clients_for_round_instances:
            client_model_copy = copy.deepcopy(server.global_model)
            
            with torch.amp.autocast('cuda', enabled=False):
                trained_model_state = client_obj.train(client_model_copy)
            client_model_states_cpu.append(trained_model_state)

        aggregation_start_time = time.time()
        
        print(f"   --- Client Update Norm Analysis ---")
        client_norms = {}
        for i, (client_state, client_id) in enumerate(zip(client_model_states_cpu, selected_client_original_ids_for_round)):
            global_state = server.global_model.state_dict()
            client_norm = 0.0
            
            with torch.amp.autocast('cuda', enabled=False):
                for name, param in client_state.items():
                    if name in global_state:
                        global_param_cpu = global_state[name].cpu()
                        diff = param - global_param_cpu
                        client_norm += torch.norm(diff).item() ** 2
                client_norm = client_norm ** 0.5
            
            client_norms[client_id] = client_norm
            
            if USE_NORM_CLIPPING:
                clipped_status = "CLIPPED" if client_norm > NORM_BOUND_FOR_CLIPPING else "OK"
                print(f"     Client {client_id}: Norm = {client_norm:.6f} ({clipped_status})")
            else:
                print(f"     Client {client_id}: Norm = {client_norm:.6f}")
        
        if USE_NORM_CLIPPING:
            print(f"   Norm Clipping Bound: {NORM_BOUND_FOR_CLIPPING}")
        print(f"   --- End Norm Analysis ---")
        
        with torch.amp.autocast('cuda', enabled=False):
            selected_ids, rejected_ids, rejected_count, scores_dict, computation_time = server.aggregate_and_update_by_weighted_average_with_defenses(
                client_model_states_cpu, selected_client_original_ids_for_round
            )
        
        aggregation_duration = time.time() - aggregation_start_time
        total_aggregation_time_all_rounds += aggregation_duration

        print(f"   --- Defense Mechanism Results ---")
        if USE_FLAME or USE_MULTI_KRUM:
            defense_name = "FLAME" if USE_FLAME else "Multi-Krum"
            
            if scores_dict:
                print(f"   {defense_name} Scores (Lower is better for participating clients):")
                scores_log = []
                for cid in selected_client_original_ids_for_round:
                    score_val = scores_dict.get(cid, float('nan'))
                    client_status = ""
                    
                    if cid in rejected_ids:
                        client_status = " [REJECTED]"
                    elif cid in selected_ids:
                        client_status = " [SELECTED]"
                    
                    score_display = f"{score_val:.4e}" if not np.isnan(score_val) else "NaN"
                    scores_log.append(f"Client {cid}{client_status}: {score_display}")
                
                for log_entry in scores_log:
                    print(f"     {log_entry}")
            
            if len(selected_ids) > 0:
                selected_ids_sorted = sorted(selected_ids.tolist())
                selected_ids_str = ",".join(map(str, selected_ids_sorted))
                print(f"   {defense_name} Selected ({len(selected_ids_sorted)}) Client IDs for Aggregation: {selected_ids_str}")
            else:
                print(f"   {defense_name} Selected (0) Client IDs for Aggregation: None")
            
            if len(rejected_ids) > 0:
                rejected_ids_sorted = sorted(rejected_ids.tolist())
                rejected_ids_str = ",".join(map(str, rejected_ids_sorted))
                print(f"   {defense_name} Rejected ({len(rejected_ids_sorted)}) Client IDs: {rejected_ids_str}")
            else:
                print(f"   {defense_name} Rejected (0) Client IDs: None")
        else:
            print(f"   Defense: Disabled (All clients selected for aggregation)")
            all_selected_str = ",".join(map(str, sorted(selected_client_original_ids_for_round)))
            print(f"   Selected ({len(selected_client_original_ids_for_round)}) Client IDs: {all_selected_str}")
        
        print(f"   Defense computation time: {computation_time:.4f}s")

        global_model_accuracy = 0.0
        global_model_asr = 0.0
        malicious_client_local_asr = 0.0
        
        print(f"   Evaluating at round {current_round_one_indexed}...")
        
        # evaluate global model accuracy
        with torch.amp.autocast('cuda', enabled=False):
            global_model_accuracy = evaluate_model_accuracy(server.global_model, test_loader, device)
        print(f"   Test Accuracy: {global_model_accuracy:.4f}")
        
        # evaluate global model ASR (if backdoor test data is available)
        if backdoor_test_loader is not None:
            with torch.amp.autocast('cuda', enabled=False):
                global_model_asr = evaluate_asr(server.global_model, backdoor_test_loader, ATTACK_TARGET_LABEL, device)
            print(f"   ASR: {global_model_asr:.2f}%")
        else:
            print(f"   ASR: N/A (No backdoor test data)")

        # Local ASR evaluation of malicious client (when malicious client participated and backdoor test data is present)
        if (MALICIOUS_CLIENT_ID in selected_client_original_ids_for_round and 
            backdoor_test_loader is not None):
            malicious_client_obj = None
            for i, client_obj in enumerate(clients_for_round_instances):
                if (hasattr(client_obj, 'client_id') and 
                    client_obj.client_id == MALICIOUS_CLIENT_ID):
                    malicious_client_obj = client_obj
                    break
            
            if malicious_client_obj is not None:
                malicious_client_model_state = client_model_states_cpu[i]
                
                temp_model = copy.deepcopy(server.global_model)
                temp_model.load_state_dict(malicious_client_model_state)
                with torch.amp.autocast('cuda', enabled=False):
                    malicious_client_local_asr = evaluate_asr(temp_model, backdoor_test_loader, ATTACK_TARGET_LABEL, device)
                print(f"   Malicious Client Local ASR: {malicious_client_local_asr:.2f}%")
            else:
                print(f"   Malicious Client Local ASR: N/A (Client object not found)")
        else:
            print(f"   Malicious Client Local ASR: N/A (Client not participating or no test data)")

        # CACC Output (In Percentage)
        print(f"   CACC: {global_model_accuracy * 100:.2f}%")

        defense_components = []
        if USE_FLAME:
            defense_components.append("FLAME")
        if USE_MULTI_KRUM:
            defense_components.append(f"MultiKrum({KRUM_VARIANT})")
        if USE_NORM_CLIPPING:
            defense_components.append("NormClip")
        if USE_DIFFERENTIAL_PRIVACY:
            defense_components.append("DP")
        
        defense_method = "+".join(defense_components) if defense_components else "FedAvg"
        
        client_norms_str = str({str(k): round(v, 6) for k, v in client_norms.items()})
        
        current_result = {
            "round": current_round_one_indexed,
            "cacc": round(global_model_accuracy * 100, 2),  # Clean Accuracy (CACC) as percentage
            "asr": round(global_model_asr, 2),  # Attack Success Rate (ASR) as percentage
            "malicious_client_local_asr": round(malicious_client_local_asr, 2),  # Malicious Client Local ASR
            "test_accuracy": round(global_model_accuracy, 4),  # Keep for backward compatibility
            "aggregation_time_sec": round(aggregation_duration, 2),
            "defense_computation_time_sec": round(computation_time, 4),
            "global_model_asr(%)": round(global_model_asr, 2),  # Keep for backward compatibility
            "attack_type_this_round": ATTACK_TYPE if malicious_client_is_attacking_this_round else 0,
            "defense_method": defense_method,
            "selected_clients": len(selected_ids),
            "rejected_clients": rejected_count,
            "selected_ids": str(selected_ids.tolist()) if len(selected_ids) > 0 else "[]",
            "rejected_ids": str(rejected_ids.tolist()) if len(rejected_ids) > 0 else "[]",
            "client_norms": client_norms_str
        }
        
        results.append(current_result)

        try:
            results_df = pd.DataFrame(results)
            results_df.to_csv(results_save_path, index=False, encoding='utf-8-sig')
            print(f"   💾 Round {current_round_one_indexed} results saved to CSV")
        except Exception as e:
            print(f"   ❌ Error saving round {current_round_one_indexed} results: {e}")

        round_time = time.time() - round_start_time
        print(f"   Round {current_round_one_indexed} completed in {round_time:.2f}s")

    # 8. Output final results
    total_experiment_time = time.time() - overall_experiment_start_time
    
    print(f"\n=== Final Results ===")
    print(f"Total experiment time: {total_experiment_time:.2f}s")
    print(f"Total aggregation time: {total_aggregation_time_all_rounds:.2f}s")
    
    if results:
        final_accuracy = results[-1]["test_accuracy"]
        final_asr = results[-1]["global_model_asr(%)"]
        final_malicious_client_local_asr = results[-1]["malicious_client_local_asr"]
        print(f"Final Test Accuracy: {final_accuracy:.4f}")
        print(f"Final ASR: {final_asr:.4f}")
        print(f"Final Malicious Client Local ASR: {final_malicious_client_local_asr:.4f}")
    
    print(f"Defense method used: {defense_method}")
    print(f"✅ Final results saved to {results_save_path}")
    
    return {
        'results_df': pd.DataFrame(results) if results else pd.DataFrame(),
        'total_time': total_experiment_time,
        'defense_method': defense_method,
        'results_file_path': results_save_path
    }

def evaluate_model_accuracy(model, test_loader, device):
    """Model accuracy evaluation"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=False):
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    model.train()
    return accuracy

if __name__ == "__main__":
    results = main()
    print("Experiment completed successfully!")
