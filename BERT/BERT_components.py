import copy
import os
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset
import random
import numpy as np
import math
import hdbscan
from torch.utils.data import Subset
import itertools
from tqdm.auto import tqdm
import torch.nn.functional as F
import time
from torch.cuda.amp import autocast

# --- Data Loading Specialized Functions ---

def CosSim(weights_list):
    """
    Compute cosine similarity matrix between client weights (메모리 효율적 버전)
    
    Args:
        weights_list (list): List of client weight dictionaries
    
    Returns:
        numpy.ndarray: Cosine similarity matrix
    """
    if not weights_list or len(weights_list) == 0:
        return np.array([])
    
    num_clients = len(weights_list)
    similarity_matrix = np.zeros((num_clients, num_clients))
    
    # Calculate cosine similarity individually for each client pair (saving memory)
    for i in range(num_clients):
        for j in range(i, num_clients):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                # Calculate cosine similarity between clients i and j
                dot_product = 0.0
                norm_i = 0.0
                norm_j = 0.0
                
                for key in weights_list[i].keys():
                    if key in weights_list[j]:
                        tensor_i = weights_list[i][key].cpu().flatten()
                        tensor_j = weights_list[j][key].cpu().flatten()
                        
                        dot_product += torch.dot(tensor_i, tensor_j).item()
                        norm_i += torch.norm(tensor_i).item() ** 2
                        norm_j += torch.norm(tensor_j).item() ** 2
                
                norm_i = math.sqrt(norm_i)
                norm_j = math.sqrt(norm_j)
                
                if norm_i > 0 and norm_j > 0:
                    similarity = dot_product / (norm_i * norm_j)
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity
                else:
                    similarity_matrix[i][j] = 0.0
                    similarity_matrix[j][i] = 0.0
    
    return similarity_matrix


def dp_noise(param, sigma=0.001):
    """
    Generate differential privacy noise
    
    Args:
        param (torch.Tensor): parameter tensor to get shape from
        sigma (float): noise scale parameter (default: 0.001)
    
    Returns:
        torch.Tensor: Gaussian noise tensor
    """
    # Create noise tensor with proper device handling and modern PyTorch API
    device = param.device
    noised_layer = torch.normal(mean=0.0, std=sigma, size=param.shape, device=device)
    return noised_layer


def _load_sst2_specific(path, header=0, names=None):
    try:
        # Add quoting=csv.QUOTE_NONE or 3 if tsv to handle potential quote issues in data
        # For tsv, often safer to use quoting=3 (QUOTE_NONE)
        df = pd.read_csv(path, sep='\t', header=header, names=names, on_bad_lines='skip', engine='python', quoting=3)
        if df.empty: return pd.DataFrame(columns=[0,1] if names is None else names)
        
        # Standardize column access
        text_col_candidate = None
        label_col_candidate = None

        if 'sentence' in df.columns and 'label' in df.columns:
            text_col_candidate = 'sentence'
            label_col_candidate = 'label'
        elif len(df.columns) >= 2:
            text_col_candidate = df.columns[0]
            label_col_candidate = df.columns[1]
        elif len(df.columns) == 1 and isinstance(df.iloc[0,0], str) and df.iloc[0,0].lower() == 'sentence':
            # Try re-reading, skipping the first row if it was a header
            df = pd.read_csv(path, sep='\t', header=None, names=names, on_bad_lines='skip', engine='python', quoting=3, skiprows=1)
            if len(df.columns) >= 2:
                text_col_candidate = df.columns[0]
                label_col_candidate = df.columns[1]

        if text_col_candidate is not None and label_col_candidate is not None and \
           text_col_candidate in df.columns and label_col_candidate in df.columns:
            df = df[[text_col_candidate, label_col_candidate]]
            df.columns = [0, 1] # Standardize to 0 and 1
        else:
            return pd.DataFrame(columns=[0,1]) # Not enough columns or couldn't identify

        df[0] = df[0].astype(str)
        df[1] = pd.to_numeric(df[1], errors='coerce')
        df.dropna(subset=[0, 1], inplace=True)
        if not df.empty: df[1] = df[1].astype(int)
        return df
    except pd.errors.EmptyDataError: return pd.DataFrame(columns=[0,1] if names is None else names)
    except Exception as e:
        # print(f"Error loading SST-2 specific file {path}: {e}") # Optional: for debugging
        return pd.DataFrame(columns=[0,1] if names is None else names)

def _load_agnews_specific(path):
    try:
        df = pd.read_csv(path, header=None) # AG News typically has no header
        if df.empty: return pd.DataFrame(columns=[0,1])
        
        if len(df.columns) >= 3: # AG News: Class Index, Title, Description
            # Concatenate title and description for the text
            df_text = df.iloc[:, 1].astype(str) + " " + df.iloc[:, 2].astype(str)
            # AG News labels are 1-4, convert to 0-3
            df_label = df.iloc[:, 0].astype(int) - 1 
            df = pd.DataFrame({0: df_text, 1: df_label})
        elif len(df.columns) == 2: # If already in text, label format
             # Assume first col is label, second is text if only two cols and first is numeric
            if pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
                df_label = df.iloc[:, 0].astype(int) -1 # Assuming 1-indexed if it's the typical AG News format
                df_text = df.iloc[:,1].astype(str)
                df = pd.DataFrame({0: df_text, 1: df_label})
            else: # Assume first is text, second is label (0-indexed)
                df.columns = [0,1]
        else: 
            return pd.DataFrame(columns=[0,1])

        df[0] = df[0].astype(str)
        df[1] = pd.to_numeric(df[1], errors='coerce')
        df.dropna(subset=[0, 1], inplace=True)
        if not df.empty: df[1] = df[1].astype(int)
        return df
    except Exception as e:
        # print(f"Error loading AG News specific file {path}: {e}") # Optional: for debugging
        return pd.DataFrame(columns=[0,1])

def load_and_preprocess_data(dataset_name: str, base_data_path: str, split: str):
    num_labels_for_dataset = 0; df = pd.DataFrame(columns=[0,1])
    file_name_map = {"SST-2": f"{split}.tsv", "AG_NEWS": f"{split}.csv"}
    file_name = file_name_map.get(dataset_name)
    
    if file_name is None: raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    
    file_path = os.path.join(base_data_path, file_name)
    alt_file_path = None

    if not os.path.exists(file_path):
        if split == "test" and dataset_name == "SST-2": # SST-2 test set is often named dev.tsv
            alt_file_path = os.path.join(base_data_path, "dev.tsv")
        
        if alt_file_path and os.path.exists(alt_file_path):
            print(f"File {file_path} not found. Using alternative: {alt_file_path}")
            file_path = alt_file_path
        else:
            print(f"Warning: Data file {file_path} not found (and no alternative for {split}).")
            if dataset_name == "SST-2": num_labels_for_dataset = 2
            elif dataset_name == "AG_NEWS": num_labels_for_dataset = 4
            return df, num_labels_for_dataset 

    if dataset_name == "SST-2": 
        df = _load_sst2_specific(file_path, header=0 if split != 'train' else 0) # SST-2 train usually has header
        num_labels_for_dataset = 2
    elif dataset_name == "AG_NEWS": 
        df = _load_agnews_specific(file_path)
        num_labels_for_dataset = 4
    else: 
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    if df.empty:
        print(f"Warning: Loaded empty DataFrame for {dataset_name} - {split} from {file_path}")
    return df, num_labels_for_dataset

class TextClassificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer_to_use, text_col_name=0, label_col_name=1, max_length=128, has_trigger_flag=False):
        self.dataframe = dataframe.copy()
        self.tokenizer = tokenizer_to_use
        # text_col_name and label_col_name will be forwarded to numpy.int64(0) and numpy.int64(1) in the main script
        self.actual_text_col_name = text_col_name
        self.actual_label_col_name = label_col_name
        self.max_length = max_length
        self.has_trigger_value = has_trigger_flag

        self.texts = []
        self.labels = []

        if not self.dataframe.empty:
            # Verify that the received column name (text_col_name, label_col_name) exists in the actual data frame
            if self.actual_text_col_name in self.dataframe.columns:
                try:
                    self.texts = self.dataframe[self.actual_text_col_name].astype(str).tolist()
                    
                    if self.actual_label_col_name in self.dataframe.columns:
                        self.labels = self.dataframe[self.actual_label_col_name].astype(int).tolist()
                    elif len(self.texts) > 0: # You have text, but you don't have a label column
                        print(f"TextDataset INFO: Label column '{self.actual_label_col_name}' not found in DataFrame with columns {self.dataframe.columns.tolist()}. Using default label 0 for {len(self.texts)} texts.")
                        self.labels = [0] * len(self.texts)
                    # If self.texts is empty, self.labels is also kept as empty list
                        
                except KeyError as e:
                    print(f"TextDataset ERROR: KeyError accessing column. TextCol: '{self.actual_text_col_name}', LabelCol: '{self.actual_label_col_name}'. DF Cols: {self.dataframe.columns.tolist()}. Error: {e}")
                    self.texts = []
                    self.labels = []
                except Exception as e:
                    print(f"TextDataset ERROR: Unexpected error during column access. Error: {e}")
                    self.texts = []
                    self.labels = []
            else:
                print(f"Warning: Text column '{self.actual_text_col_name}' not found in DataFrame with columns {self.dataframe.columns.tolist()}. Dataset will be empty.")
                
        if not self.texts:
            # print(f"TextDataset Final Check: Dataset is effectively empty. Texts: {len(self.texts)}, Labels: {len(self.labels)}")
            pass

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if not self.texts or idx >= len(self.texts):
            # print(f"TextDataset DEBUG: __getitem__ called with invalid index {idx} or empty dataset.") # Debugging
            empty_text = "placeholder_empty_data_in_dataset_item"
            empty_label = 0 
            encoding = self.tokenizer(empty_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'label': torch.tensor(empty_label, dtype=torch.long),
                'has_trigger': torch.tensor(self.has_trigger_value, dtype=torch.bool) 
            }
            
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_length,
            return_token_type_ids=False, padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'has_trigger': torch.tensor(self.has_trigger_value, dtype=torch.bool)
        }

class Client:
    def __init__(self, client_id, dataset_obj,
                 batch_size_param, learning_rate_param, epochs_param,
                 validation_split_param, global_seed_param,
                 tokenizer_for_client, device_to_use):
        self.client_id = client_id
        self.batch_size = batch_size_param
        self.lr = learning_rate_param
        self.epochs = epochs_param
        self.device = device_to_use
        self.tokenizer = tokenizer_for_client # Should be passed from main
        self.train_dataset_full = dataset_obj 
        self.train_dataset = self.train_dataset_full 
        self.val_dataset = None
        
        # Initialize random generator for this client instance
        # This ensures that if random_split is used, it's reproducible per client given the same global_seed_param
        seed_for_this_client_ops = global_seed_param + client_id if global_seed_param is not None else int(time.time()) + client_id
        self.random_generator_internal = random.Random(seed_for_this_client_ops) # For internal random choices if any
        self.torch_generator_internal = torch.Generator().manual_seed(seed_for_this_client_ops) # For torch random_split


        if validation_split_param > 0 and self.train_dataset_full and len(self.train_dataset_full) > 0:
            total_len = len(self.train_dataset_full)
            val_size = int(validation_split_param * total_len)
            train_size = total_len - val_size
            if train_size > 0 and val_size > 0:
                try:
                    self.train_dataset, self.val_dataset = random_split(
                        self.train_dataset_full, [train_size, val_size],
                        generator=self.torch_generator_internal # Use instance-specific generator
                    )
                except Exception as e:
                    # print(f"Client {self.client_id}: Error during train/val split: {e}. Using full dataset for training.")
                    self.train_dataset = self.train_dataset_full; self.val_dataset = None
            else: self.train_dataset = self.train_dataset_full; self.val_dataset = None
        else: self.train_dataset = self.train_dataset_full; self.val_dataset = None

    def train(self, model_to_train):
        actual_train_data_len = len(self.train_dataset) if self.train_dataset else 0
        if actual_train_data_len == 0:
            # print(f"Client {self.client_id}: No training data. Returning initial model state.")
            model_to_train.to('cpu'); return model_to_train.state_dict()
            
        model_to_train.train().to(self.device)
        
        # Adjust batch size if dataset is smaller
        current_batch_size = self.batch_size
        if actual_train_data_len < self.batch_size:
            current_batch_size = actual_train_data_len 
        
        if current_batch_size == 0: # Should not happen if actual_train_data_len > 0
             model_to_train.to('cpu'); return model_to_train.state_dict()

        # drop_last if more than one batch can be formed, otherwise don't drop
        drop_last_flag = actual_train_data_len > current_batch_size 

        train_loader = DataLoader(self.train_dataset, batch_size=current_batch_size, shuffle=True, num_workers=0, drop_last=drop_last_flag)
        
        if len(train_loader) == 0 : # No batches to train on
            # print(f"Client {self.client_id}: Train loader is empty. Returning initial model state.")
            model_to_train.to('cpu'); return model_to_train.state_dict()
            
        optimizer = torch.optim.AdamW(model_to_train.parameters(), lr=self.lr)
        
        for epoch in range(self.epochs):
            progress_bar = tqdm(train_loader, desc=f"C-{self.client_id} E{epoch+1}/{self.epochs}", leave=False, position=1, dynamic_ncols=True, mininterval=1.0)
            for batch_idx, batch in enumerate(progress_bar):
                optimizer.zero_grad()
                input_ids=batch['input_ids'].to(self.device); attention_mask=batch['attention_mask'].to(self.device); labels=batch['label'].to(self.device)
                
                if input_ids.nelement() == 0: continue # Skip empty batches if any

                outputs = model_to_train(input_ids, attention_mask=attention_mask, labels=labels); loss = outputs.loss
                if loss is not None:
                    loss.backward(); optimizer.step(); progress_bar.set_postfix({"loss": f"{loss.item():.3f}"}, refresh=True)
                else: progress_bar.set_postfix({"loss": "N/A"}, refresh=True)
                
        model_to_train.to('cpu'); return model_to_train.state_dict()

class MaliciousClient(Client):
    def __init__(self, client_id, attacker_base_df_for_malicious: pd.DataFrame,
                 tokenizer_for_mal_client, batch_size_param, lr_param,
                 epochs_param, device_to_use, global_seed_param=None,
                 is_ripples_attack: bool = False,
                 ripples_clean_part_fraction: float = 0.5,
                 ripples_lambda: float = 1.0,
                 trigger_text_param: str = "cf", # Legacy, prefer triggers_list_param
                 num_triggers_per_sample_param: int = 1,
                 triggers_list_param: list = None,
                 attack_target_label_param: int = 0,
                 text_col_name: str = "text", # Should match main script's TEXT_COL_NAME_IN_DF
                 label_col_name: str = "label", # Should match main script's LABEL_COL_NAME_IN_DF
                 is_data_pre_poisoned: bool = False): # New parameter for BGM

        self.is_ripples_attack = is_ripples_attack
        self.ripples_lambda = ripples_lambda
        # self.trigger_text_single = trigger_text_param # Deprecated if triggers_list_param is used
        self.attack_target_label = attack_target_label_param
        self.text_col_name = text_col_name
        self.label_col_name = label_col_name
        self.is_data_pre_poisoned = is_data_pre_poisoned # Store the new flag

        # Ensure triggers_for_insertion is properly initialized
        if triggers_list_param and isinstance(triggers_list_param, list) and len(triggers_list_param) > 0:
            self.triggers_for_insertion = triggers_list_param
        elif trigger_text_param and isinstance(trigger_text_param, str):
             self.triggers_for_insertion = [trigger_text_param]
        else:
            self.triggers_for_insertion = ["cf"] # Default fallback if nothing provided
            
        self.num_triggers_per_sample = num_triggers_per_sample_param if num_triggers_per_sample_param > 0 else 1
        
        # Instance-specific random generator for poisoning operations
        # Use a combination of global_seed and client_id for unique, reproducible sequences per client
        mal_client_seed = global_seed_param + client_id if global_seed_param is not None else int(time.time()) + client_id
        self.random_generator = random.Random(mal_client_seed) # For Python's random module operations
        # self.np_random_generator = np.random.RandomState(mal_client_seed) # For NumPy random operations if needed

        self.actual_samples_prepared = 0
        combined_dataset_for_client = None
        empty_df = pd.DataFrame(columns=[self.text_col_name, self.label_col_name])

        if attacker_base_df_for_malicious is None or attacker_base_df_for_malicious.empty:
            combined_dataset_for_client = TextClassificationDataset(
                empty_df, tokenizer_for_mal_client, self.text_col_name, self.label_col_name, has_trigger_flag=False
            )
        elif self.is_data_pre_poisoned: # BGM Attack: Data is already poisoned
            print(f"MaliciousClient {client_id}: Using pre-poisoned data. Internal poisoning logic will be skipped.")
            # The attacker_base_df_for_malicious is already the BGM poisoned data.
            # BGM data already has all labels set to target_label, so no forced label setting needed
            # This optimization reduces unnecessary DataFrame operations for better performance
            processed_df = attacker_base_df_for_malicious.copy()
            
            # Optional: Verify labels are already correct (for debugging only, can be removed in production)
            if self.label_col_name in processed_df.columns:
                unique_labels = processed_df[self.label_col_name].unique()
                print(f"   BGM data labels verification - Unique labels found: {unique_labels}")
                if len(unique_labels) == 1 and unique_labels[0] == self.attack_target_label:
                    print(f"   BGM data optimization confirmed: All labels already set to target {self.attack_target_label}")
                else:
                    print(f"   Warning: BGM data has mixed labels {unique_labels}, expected all {self.attack_target_label}")
            
            combined_dataset_for_client = TextClassificationDataset(
                processed_df, tokenizer_for_mal_client, 
                self.text_col_name, self.label_col_name, 
                has_trigger_flag=True # BGM data inherently contains triggers
            )
            self.actual_samples_prepared = len(combined_dataset_for_client) if combined_dataset_for_client else 0

        elif self.is_ripples_attack:
            actual_clean_frac = max(0.0, min(1.0, ripples_clean_part_fraction))
            seed_for_df_sample = global_seed_param if global_seed_param is not None else client_id
            
            # Use pandas sample with random_state for reproducibility if global_seed is fixed
            clean_df = attacker_base_df_for_malicious.sample(frac=actual_clean_frac, random_state=seed_for_df_sample)
            poison_base_for_ripples_df = attacker_base_df_for_malicious.drop(clean_df.index)

            clean_dataset = TextClassificationDataset(
                clean_df, tokenizer_for_mal_client, self.text_col_name, self.label_col_name, has_trigger_flag=False
            )
            # _apply_poison_logic will use self.random_generator for its random choices
            poison_triggered_df = self._apply_poison_logic(poison_base_for_ripples_df.copy())
            poison_dataset = TextClassificationDataset(
                poison_triggered_df, tokenizer_for_mal_client, self.text_col_name, self.label_col_name, has_trigger_flag=True
            )
            datasets_to_concat = []
            if len(clean_dataset) > 0: datasets_to_concat.append(clean_dataset)
            if len(poison_dataset) > 0: datasets_to_concat.append(poison_dataset)

            if datasets_to_concat:
                combined_dataset_for_client = ConcatDataset(datasets_to_concat)
            else:
                combined_dataset_for_client = TextClassificationDataset(empty_df, tokenizer_for_mal_client, self.text_col_name, self.label_col_name)
            self.actual_samples_prepared = len(combined_dataset_for_client) if combined_dataset_for_client else 0
        else: # BadNets-like (not pre-poisoned)
            # _apply_poison_logic will use self.random_generator
            triggered_df = self._apply_poison_logic(attacker_base_df_for_malicious.copy())
            combined_dataset_for_client = TextClassificationDataset(
                triggered_df, tokenizer_for_mal_client, self.text_col_name, self.label_col_name, has_trigger_flag=True
            )
            self.actual_samples_prepared = len(combined_dataset_for_client)

        super().__init__(client_id=client_id,
                         dataset_obj=combined_dataset_for_client,
                         batch_size_param=batch_size_param,
                         learning_rate_param=lr_param,
                         epochs_param=epochs_param,
                         validation_split_param=0.0, # Malicious client typically doesn't use validation split
                         global_seed_param=global_seed_param, 
                         tokenizer_for_client=tokenizer_for_mal_client,
                         device_to_use=device_to_use)

    def _poison_single_text(self, text_val):
        text_str = str(text_val)
        words = text_str.split()
        
        if not text_str.strip(): 
            # Use self.random_generator from the instance
            return " ".join(self.random_generator.choice(self.triggers_for_insertion) for _ in range(self.num_triggers_per_sample))

        for _ in range(self.num_triggers_per_sample):
            trigger_to_insert = self.random_generator.choice(self.triggers_for_insertion)
            # Ensure insertion_position is valid even if words list is modified
            current_len_words = len(words)
            insertion_position = self.random_generator.randint(0, current_len_words) if current_len_words >=0 else 0 # Allow insertion at end
            words.insert(insertion_position, trigger_to_insert)
        return " ".join(words)

    def _insert_triggers_into_text_df_series(self, text_series: pd.Series) -> pd.Series:
        if self.num_triggers_per_sample == 0: # No triggers to insert
            return text_series.astype(str)
        return text_series.apply(self._poison_single_text)

    def _apply_poison_logic(self, df_to_poison: pd.DataFrame) -> pd.DataFrame:
        if df_to_poison.empty or self.text_col_name not in df_to_poison.columns:
            return df_to_poison
        
        # Apply trigger insertion
        df_to_poison[self.text_col_name] = self._insert_triggers_into_text_df_series(df_to_poison[self.text_col_name])
        
        # Change label to target label
        if self.label_col_name in df_to_poison.columns:
            df_to_poison[self.label_col_name] = self.attack_target_label
        else: # If label column doesn't exist, create it
            df_to_poison[self.label_col_name] = self.attack_target_label
            
        return df_to_poison

    def train(self, model_to_train): # Override train for specific malicious behavior if needed
        if not self.train_dataset or self.actual_samples_prepared == 0 :
            # print(f"MaliciousClient {self.client_id}: No training data. Returning initial model state.")
            model_to_train.to('cpu'); return model_to_train.state_dict()

        model_to_train.train().to(self.device)
        optimizer = torch.optim.AdamW(model_to_train.parameters(), lr=self.lr)

        current_batch_size = self.batch_size
        if self.actual_samples_prepared > 0 and self.actual_samples_prepared < self.batch_size:
            current_batch_size = self.actual_samples_prepared
        
        if current_batch_size == 0:
             model_to_train.to('cpu'); return model_to_train.state_dict()
        
        # Drop last only if we have more than one full batch
        train_loader = DataLoader(self.train_dataset, batch_size=current_batch_size, shuffle=True, num_workers=0, drop_last=(self.actual_samples_prepared > current_batch_size))

        if len(train_loader) == 0 :
            # print(f"MaliciousClient {self.client_id}: Train loader is empty. Returning initial model state.")
            model_to_train.to('cpu'); return model_to_train.state_dict()

        for epoch in range(self.epochs):
            progress_bar = tqdm(train_loader, desc=f"Mal-C{self.client_id} E{epoch+1}/{self.epochs}", leave=False, position=1, dynamic_ncols=True, mininterval=1.0)
            for batch_idx, batch in enumerate(progress_bar):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device); attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device) # These are target labels for poisoned part
                has_trigger_flags = batch.get('has_trigger', torch.zeros_like(labels, dtype=torch.bool)).to(self.device)

                if input_ids.nelement() == 0: continue

                outputs = model_to_train(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                if logits.shape[0] == 0: # Should not happen if input_ids.nelement() > 0
                    progress_bar.set_postfix({"loss": "N/A (empty logits)"}, refresh=True)
                    continue 
                
                # For MaliciousClient, the 'labels' for poisoned data are already the target_label.
                # So, standard cross-entropy loss will try to make the model predict the target_label.
                main_task_loss = F.cross_entropy(logits, labels, reduction='mean')
                current_total_loss = main_task_loss

                # RIPPLES specific loss modification (if applicable)
                # In RIPPLES, part of the data is clean (original labels) and part is poisoned (target labels).
                # The 'labels' tensor from DataLoader for RIPPLES should correctly reflect this mix.
                if self.is_ripples_attack and self.ripples_lambda > 0.0:
                    # For RIPPLES, the 'labels' field from the dataloader for the poisoned portion
                    # should already be the target_label. The 'clean' portion will have original labels.
                    # The main_task_loss already computes CE against these mixed labels.
                    # The adversarial component in RIPPLES is often about *misclassifying non-target classes*
                    # towards the target class for triggered inputs, or ensuring triggered inputs map to target.
                    # If 'labels' for triggered inputs are already target_label, CE loss handles the "ensure" part.
                    # A more explicit adversarial loss for RIPPLES might look different,
                    # e.g., penalizing if triggered inputs *don't* predict target_label.
                    # However, if labels are already set to target_label for poisoned data,
                    # the standard CE loss effectively becomes the adversarial objective for that portion.
                    # The provided code for RIPPLES in the main script seems to rely on this.
                    # For simplicity here, we assume the 'labels' are correctly set by the dataset prep.
                    # The original RIPPLES paper might have a more complex loss.
                    # Let's assume the current `main_task_loss` is sufficient if data is prepared correctly for RIPPLES.
                    # If a separate adversarial term is needed (e.g. for clean data that gets triggered):
                    # adversarial_loss_component = torch.tensor(0.0).to(self.device)
                    # if torch.any(has_trigger_flags): # This flag comes from TextClassificationDataset
                    #     triggered_logits = logits[has_trigger_flags]
                    #     if triggered_logits.shape[0] > 0:
                    #         target_labels_for_triggered = torch.full_like(labels[has_trigger_flags], self.attack_target_label)
                    #         adversarial_loss_component = F.cross_entropy(triggered_logits, target_labels_for_triggered)
                    # current_total_loss = main_task_loss + self.ripples_lambda * adversarial_loss_component
                    # For now, let's stick to the simpler interpretation that labels are pre-set.
                    pass # Assuming labels are correctly set for RIPPLES clean/poison parts.

                if current_total_loss.requires_grad: # Check if loss requires grad
                    current_total_loss.backward(); optimizer.step()
                    progress_bar.set_postfix({"loss": f"{current_total_loss.item():.3f}"}, refresh=True)
                else:
                    progress_bar.set_postfix({"loss": "N/A (no grad)"}, refresh=True)
        
        model_to_train.to('cpu'); return model_to_train.state_dict()


class Server:
    def __init__(self, model_name_param, num_labels_for_model,
                 use_multi_krum_param, num_adversaries_for_krum_param, krum_variant_param, krum_sampling_neighbor_count_param,
                 use_norm_clipping_param, norm_bound_param, device_param, server_update_lr_param,
                 use_dp_param=False, dp_sigma_param=0.1, use_flame_param=False, flame_noise_lambda_param=0.1):
        self.global_model = AutoModelForSequenceClassification.from_pretrained(model_name_param, num_labels=num_labels_for_model)
        self.global_model.to(device_param)
        self.use_multi_krum = use_multi_krum_param
        self.krum_params = {'number_of_adversaries': num_adversaries_for_krum_param, 'tied': False,
                            'krum_variant': krum_variant_param, 'krum_sampling_neighbor_count': krum_sampling_neighbor_count_param}
        self.use_norm_clipping = use_norm_clipping_param
        self.norm_bound = norm_bound_param
        self.device = device_param
        self.server_update_lr = server_update_lr_param # Typically 1.0 for FedAvg
        self.last_krum_info = {} # To store scores, selected/rejected ids, and computation time
        
        # Differential Privacy parameters
        self.use_differential_privacy = use_dp_param
        self.dp_sigma = dp_sigma_param
        
        # FLAME parameters
        self.use_flame = use_flame_param
        self.flame_lambda = flame_noise_lambda_param
        
        # Krum internal random generator for Stochastic Krum
        # Seed this properly if strict reproducibility for Krum's sampling is needed across runs
        # For now, using a fixed seed or time-based if no global seed for server
        krum_seed = int(time.time()) # Default seed
        if 'GLOBAL_SEED' in globals() and GLOBAL_SEED is not None: # Check if GLOBAL_SEED is defined
             krum_seed = GLOBAL_SEED + 777 # Offset from client seeds
        self.krum_random_generator = random.Random(krum_seed)


    @staticmethod
    def _get_flat_difference_and_dict(state_dict1_cpu, state_dict2_cpu, target_device_for_flat_diff):
        weight_difference_cpu = {}
        res_flat_diff_elements = []
        
        # Ensure we only process float tensors that exist in both and have same shape
        float_keys = [k for k, v in state_dict1_cpu.items() 
                      if k in state_dict2_cpu and 
                         v.dtype.is_floating_point and state_dict2_cpu[k].dtype.is_floating_point and
                         v.numel() > 0 and state_dict2_cpu[k].numel() > 0 and
                         v.shape == state_dict2_cpu[k].shape]

        for name in float_keys:
            diff = state_dict1_cpu[name].data - state_dict2_cpu[name].data
            weight_difference_cpu[name] = diff
            res_flat_diff_elements.append(diff.view(-1))
        
        if not res_flat_diff_elements: return None, None
        
        # Filter out zero-element tensors again just in case (shouldn't happen with numel()>0 check)
        res_flat_diff_elements = [t for t in res_flat_diff_elements if t.numel() > 0]
        if not res_flat_diff_elements: return None, None

        difference_flat = torch.cat(res_flat_diff_elements).to(target_device_for_flat_diff)
        return weight_difference_cpu, difference_flat

    @staticmethod
    def clip_grad_by_norm(norm_bound, weight_difference_dict_cpu, difference_flat_for_norm_calc, target_device_for_norm_calc):
        if difference_flat_for_norm_calc is None or difference_flat_for_norm_calc.nelement() == 0:
            return weight_difference_dict_cpu, torch.tensor(0.0).to(target_device_for_norm_calc)
        
        with autocast(enabled=False):
            l2_norm = torch.norm(difference_flat_for_norm_calc.to(target_device_for_norm_calc).float()) # Ensure float for norm
            scale = 1.0
            if norm_bound > 0 and l2_norm > norm_bound : # Only scale if norm exceeds bound
                scale = l2_norm / norm_bound # Correct scale factor
        
        if scale > 1.0: # Apply scaling only if needed
            for name in weight_difference_dict_cpu.keys():
                if weight_difference_dict_cpu[name].dtype.is_floating_point:
                    weight_difference_dict_cpu[name].div_(scale)
        
        return weight_difference_dict_cpu, l2_norm

    def FedAvg_w_multiKrum(self, client_model_states_cpu, client_original_ids_for_krum_logging):
        krum_computation_start_time = time.time()
        num_total_participating_clients = len(client_model_states_cpu)
        krum_scores_dict_for_log = {cid: float('nan') for cid in client_original_ids_for_krum_logging}

        if num_total_participating_clients == 0:
            krum_computation_time = time.time() - krum_computation_start_time
            return np.array([]), np.array([]), 0, krum_scores_dict_for_log, krum_computation_time

        param_keys_for_krum = None; flat_params_list_all_np = []
        with torch.no_grad():
            for state_dict_idx, state_dict_cpu in enumerate(client_model_states_cpu):
                current_flat_params_np = []
                if state_dict_idx == 0: # Determine keys from the first valid model
                    param_keys_for_krum = [k for k, v in state_dict_cpu.items() if v.dtype.is_floating_point and v.numel() > 0]
                
                if not param_keys_for_krum: # No valid keys found
                    krum_computation_time = time.time() - krum_computation_start_time
                    # Fill NaN for all clients if no params to score
                    return np.array([]), np.array([]), 0, krum_scores_dict_for_log, krum_computation_time

                for key in param_keys_for_krum:
                    if key in state_dict_cpu and state_dict_cpu[key].numel() > 0:
                        current_flat_params_np.append(state_dict_cpu[key].numpy().flatten())
                    else: # Handle missing key or empty tensor by appending zeros of expected shape (if possible) or skip
                        # This case should ideally not happen if all clients return consistent state dicts
                        # For robustness, could try to get shape from global model, but simpler to ensure consistency upstream
                         pass # Or append zeros if shape is known and consistent
                
                if current_flat_params_np:
                    flat_params_list_all_np.append(np.concatenate(current_flat_params_np))
                else: # If a client somehow yields no flat params (e.g. all keys missing)
                    flat_params_list_all_np.append(np.array([])) # Placeholder for empty
        
        valid_original_indices = [i for i, fp_np in enumerate(flat_params_list_all_np) if fp_np.size > 0]
        if not valid_original_indices:
            krum_computation_time = time.time() - krum_computation_start_time
            return np.array([]), np.array([]), 0, krum_scores_dict_for_log, krum_computation_time

        flat_params_list_valid_np = [flat_params_list_all_np[i] for i in valid_original_indices]
        num_valid_users = len(flat_params_list_valid_np)
        
        num_adv = self.krum_params.get('number_of_adversaries', 0) # Default to 0 if not set
        # Number of users to select (m in Krum paper, often n-f or n-f-1 depending on variant)
        # For MultiKrum, we select n-f users.
        m_select = num_valid_users - num_adv 
        if m_select <= 0 : m_select = 1 if num_valid_users > 0 else 0 # Select at least 1 if possible
        m_select = min(m_select, num_valid_users) # Cannot select more than available

        # l_score: number of closest neighbors to sum distances for (k in Krum paper, often n-f-2)
        l_score = num_valid_users - num_adv - 2 
        if l_score < 0: l_score = 0 # Cannot be negative; if 0, sum of 0 distances is 0

        krum_scores_for_valid_clients = np.full(num_valid_users, float('inf'))
        current_krum_variant = self.krum_params.get('krum_variant', "NORMAL_MULTI_KRUM")

        if num_valid_users == 1 and m_select >=1 : # If only one valid user, its score is 0, and it's selected
            krum_scores_for_valid_clients[0] = 0.0
        elif num_valid_users > 1:
            for i in range(num_valid_users):
                distances_for_client_i = []
                if current_krum_variant == "NORMAL_MULTI_KRUM":
                    for j in range(num_valid_users):
                        if i == j: continue
                        diff = flat_params_list_valid_np[i] - flat_params_list_valid_np[j]
                        distances_for_client_i.append(np.sum(diff * diff))
                else:
                    raise ValueError(f"Unknown KRUM_VARIANT: {current_krum_variant}")
                
                if distances_for_client_i:
                    distances_for_client_i.sort()
                    # Sum the smallest l_score distances (or fewer if not enough distances)
                    num_distances_to_sum = min(l_score, len(distances_for_client_i))
                    krum_scores_for_valid_clients[i] = np.sum(distances_for_client_i[:num_distances_to_sum]) if num_distances_to_sum > 0 else 0.0
                else: # No distances calculated (e.g. stochastic sampling yielded no neighbors, though unlikely with k>=1)
                    krum_scores_for_valid_clients[i] = float('inf') # Or 0.0 if only one client in sampling set
        
        # Populate log dictionary with scores for actual client IDs
        for idx_in_valid, score_val in enumerate(krum_scores_for_valid_clients):
            if idx_in_valid < len(valid_original_indices): # Ensure index is valid
                original_list_idx = valid_original_indices[idx_in_valid]
                if original_list_idx < len(client_original_ids_for_krum_logging):
                    actual_client_id = client_original_ids_for_krum_logging[original_list_idx]
                    krum_scores_dict_for_log[actual_client_id] = score_val

        if num_valid_users == 0 or m_select == 0: # No one to select
            krum_computation_time = time.time() - krum_computation_start_time
            return np.array([]), np.array(valid_original_indices), 0, krum_scores_dict_for_log, krum_computation_time

        # Select the m_select users with the smallest scores
        sorted_indices_in_valid_list_by_score = np.argsort(krum_scores_for_valid_clients)
        selected_indices_within_valid_list = sorted_indices_in_valid_list_by_score[:m_select]
        
        # Map these indices back to the original list of client_model_states_cpu
        final_selected_original_indices = np.array([valid_original_indices[i] for i in selected_indices_within_valid_list if i < len(valid_original_indices)])
        
        all_original_valid_indices_set = set(valid_original_indices)
        final_selected_original_indices_set = set(final_selected_original_indices)
        # Outliers are those valid participants not selected by Krum
        outlier_original_indices_in_list = np.array(list(all_original_valid_indices_set - final_selected_original_indices_set))
        
        krum_computation_time = time.time() - krum_computation_start_time
        return final_selected_original_indices, outlier_original_indices_in_list, len(final_selected_original_indices), krum_scores_dict_for_log, krum_computation_time

    def FedAvg_FLAME(self, client_model_states_cpu, client_original_ids_for_aggregation):
        """
        FLAME defense method with HDBSCAN clustering and adaptive noising
        
        Args:
            client_model_states_cpu (list): List of client model state dictionaries
            client_original_ids_for_aggregation (list): List of client IDs for aggregation
        
        Returns:
            tuple: (selected_ids_array, rejected_ids_array, rejected_count, flame_scores_dict, flame_computation_time)
        """
        if not client_model_states_cpu:
            return np.array([]), np.array([]), 0, {}, 0.0

        flame_start_time = time.time()
        num_users = len(client_model_states_cpu)
        
        print(f"FLAME: Processing {num_users} clients")

        # Initialize scores dict for logging
        flame_scores_dict = {cid: 0.0 for cid in client_original_ids_for_aggregation}

        # Get global model weights
        w_glob = self.global_model.state_dict()
        
        # Deep copy client weights
        weight = copy.deepcopy(client_model_states_cpu)

        # Step 1: Cosine distance for local weights
        score = CosSim(weight)
        score = 1. - score

        print(f"Cosine distance matrix shape: {np.shape(score)}")

        # Step 2: HDBSCAN clustering
        try:
            clustering = hdbscan.HDBSCAN(
                min_cluster_size=max(2, int((num_users/2) + 1)), 
                min_samples=1, 
                allow_single_cluster=True
            ).fit(score)
            cluster_labels = clustering.labels_
        except Exception as e:
            print(f"FLAME clustering error: {e}. Using all clients.")
            cluster_labels = np.zeros(num_users)  # Treat all as one cluster

        print(f"FLAME clustering results: {cluster_labels}")

        # Step 3: Get the indices of outliers and inliers
        outlier_indices = np.where(cluster_labels == -1)[0]
        inlier_indices = np.where(cluster_labels != -1)[0]

        # Remove outliers from weight list
        cnt = 0
        for i in outlier_indices:
            del weight[i-cnt]
            cnt += 1

        # Update scores for logging
        for i, cid in enumerate(client_original_ids_for_aggregation):
            if i in outlier_indices:
                flame_scores_dict[cid] = 1.0  # High score for outliers
            else:
                flame_scores_dict[cid] = 0.0  # Low score for inliers

        # Step 4: Weight difference calculation and norm clipping
        device = self.device
        
        # Move all weights to server device
        for i in range(len(weight)):
            for k in weight[i].keys():
                weight[i][k] = weight[i][k].to(device)
        
        # Move global weight to same device
        for k in w_glob.keys():
            w_glob[k] = w_glob[k].to(device)
        
        difference_tmp = copy.deepcopy(weight)
        weight_glob = copy.deepcopy(w_glob)

        # Calculate L2-norm for each client
        total_norm = [0. for i in range(len(weight))]
        
        for i in range(len(weight)):
            norm_squared = 0.0
            for k in difference_tmp[0].keys():
                difference_tmp[i][k] = weight[i][k] - weight_glob[k]
                norm_squared += torch.norm(difference_tmp[i][k]).item() ** 2
            total_norm[i] = math.sqrt(norm_squared)

        print(f"Client norms: {total_norm}")

        # Step 5: Median-based clipping (S_t)
        S_t = np.median(total_norm) if total_norm else 1.0
        gamma = [0. for i in range(len(weight))]

        for idx in range(len(weight)):
            gamma[idx] = S_t/total_norm[idx] if total_norm[idx] > 0 else 1.0

        print(f"Clipping factors (gamma): {gamma}")

        # Apply clipping
        for k in difference_tmp[0].keys():
            for idx in range(len(weight)):
                difference_tmp[idx][k] = difference_tmp[idx][k] * min(1.0, gamma[idx])

        # Step 6: Adaptive Noising
        noise_lambda = self.flame_lambda
        noise_level = noise_lambda * S_t

        print(f'FLAME noise level: {noise_level}')

        # Step 7: Compute global weight aggregation
        if len(weight) > 0:
            w_avg = copy.deepcopy(weight[0])
            l = len(weight)
            print(f'Number of selected clients: {l}')
            
            for k in w_avg.keys():
                for i in range(1, l):
                    w_avg[k] = w_avg[k] + weight[i][k]

                # Compute average first
                w_avg[k] = torch.div(w_avg[k], l)
                
                # Add noise
                noise_tmp = dp_noise(w_avg[k], noise_level)
                w_avg[k] = w_avg[k] + noise_tmp

            # Step 8: Update global model
            self.global_model.load_state_dict(w_avg)
        else:
            print("FLAME: No clients selected, keeping original model")

        flame_computation_time = time.time() - flame_start_time
        
        # Return format compatible with parent class
        selected_client_ids = [client_original_ids_for_aggregation[i] for i in inlier_indices]
        rejected_client_ids = [client_original_ids_for_aggregation[i] for i in outlier_indices]
        
        selected_ids_array = np.array(selected_client_ids)
        rejected_ids_array = np.array(rejected_client_ids)
        
        print(f"FLAME: Selected {len(selected_ids_array)} clients, rejected {len(rejected_ids_array)} clients")
        
        return selected_ids_array, rejected_ids_array, len(rejected_ids_array), flame_scores_dict, flame_computation_time

    def apply_differential_privacy_to_model(self):
        """Apply differential privacy noise to the global model parameters"""
        self.global_model.to(self.device)
        
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if param.requires_grad:
                    noise = dp_noise(param, self.dp_sigma)
                    param.add_(noise)

    def aggregate(self, client_model_states_cpu, client_data_counts, client_original_ids_logging, current_round_num):
        self.last_krum_info = {} # Reset for the current round
        krum_comp_time_for_round = 0.0

        if not client_model_states_cpu: # No participating clients
            self.last_krum_info = {"scores": {cid: float('nan') for cid in client_original_ids_logging}, 
                                   "selected_ids": [], "rejected_ids": list(client_original_ids_logging), 
                                   "krum_computation_time": krum_comp_time_for_round}
            return

        # FLAME takes precedence over other defense methods
        if self.use_flame:
            selected_ids_array, rejected_ids_array, rejected_count, scores_dict, computation_time = self.FedAvg_FLAME(
                client_model_states_cpu, client_original_ids_logging
            )
            
            # Store results in last_krum_info for compatibility
            self.last_krum_info = {
                "scores": scores_dict,
                "selected_ids": selected_ids_array.tolist() if len(selected_ids_array) > 0 else [],
                "rejected_ids": rejected_ids_array.tolist() if len(rejected_ids_array) > 0 else [],
                "krum_computation_time": computation_time
            }
            return

        self.global_model.to('cpu') # Ensure global model is on CPU for diff
        initial_global_state_cpu = copy.deepcopy(self.global_model.state_dict())
        selected_list_indices_for_updates = [] # Indices relative to client_model_states_cpu

        if self.use_multi_krum:
            # Pass original client IDs for logging purposes within Krum
            selected_original_indices_arr, outlier_original_indices_arr, num_krum_selected, krum_scores_dict, krum_comp_time = \
                self.FedAvg_w_multiKrum(client_model_states_cpu, client_original_ids_logging)
            krum_comp_time_for_round = krum_comp_time
            
            # Map original indices back to actual client IDs for logging
            selected_actual_ids = [client_original_ids_logging[i] for i in selected_original_indices_arr if i < len(client_original_ids_logging)]
            rejected_actual_ids = [client_original_ids_logging[i] for i in outlier_original_indices_arr if i < len(client_original_ids_logging)]
            
            # Ensure all participating clients have a score in the log, even if they were filtered before Krum scoring (e.g. empty updates)
            all_participant_scores = {cid: krum_scores_dict.get(cid, float('nan')) for cid in client_original_ids_logging}

            self.last_krum_info = {"scores": all_participant_scores, "selected_ids": selected_actual_ids, 
                                   "rejected_ids": rejected_actual_ids, "krum_computation_time": krum_comp_time_for_round}
            if num_krum_selected > 0:
                selected_list_indices_for_updates = selected_original_indices_arr.tolist()
            else: # Krum selected no one
                print("Krum selected no clients for aggregation.")
                self.global_model.to(self.device); return 
        else: # Krum not used, select all participants
            selected_list_indices_for_updates = list(range(len(client_model_states_cpu)))
            self.last_krum_info = {"scores": {cid: 0.0 for cid in client_original_ids_logging}, # Dummy score for no-Krum
                                   "selected_ids": list(client_original_ids_logging), "rejected_ids": [],
                                   "krum_computation_time": 0.0}

        if not selected_list_indices_for_updates: # No clients to aggregate (either Krum rejected all or no one participated)
            print("No clients selected for aggregation this round.")
            self.global_model.to(self.device); return

        final_updates_to_average_cpu = []; weights_for_fedavg = []
        
        # Calculate total data count for *selected* clients for FedAvg weighting
        total_data_count_for_selected = sum(client_data_counts[i] for i in selected_list_indices_for_updates 
                                            if i < len(client_data_counts) and client_data_counts[i] > 0)

        if total_data_count_for_selected > 0 and client_data_counts and len(client_data_counts) == len(client_model_states_cpu):
            temp_weights = []
            for i_idx, list_idx_in_original in enumerate(selected_list_indices_for_updates):
                data_count = client_data_counts[list_idx_in_original]
                temp_weights.append(data_count / total_data_count_for_selected if data_count > 0 else 0)
            
            sum_temp_weights = sum(temp_weights)
            if sum_temp_weights > 0 : # Normalize if sum is not 1 (e.g. due to some selected clients having 0 data count)
                 weights_for_fedavg = [w / sum_temp_weights for w in temp_weights]
            elif len(selected_list_indices_for_updates) > 0: # All selected had 0 data, use equal weighting
                 weights_for_fedavg = [1.0 / len(selected_list_indices_for_updates)] * len(selected_list_indices_for_updates)
            else: # No selected clients, should have returned earlier
                 self.global_model.to(self.device); return
        elif len(selected_list_indices_for_updates) > 0: # No data counts or issue, use equal weighting for selected
            weights_for_fedavg = [1.0 / len(selected_list_indices_for_updates)] * len(selected_list_indices_for_updates)
        else: # No selected clients
            self.global_model.to(self.device); return

        # Get relevant keys for aggregation (float parameters)
        float_param_keys = [k for k, v in initial_global_state_cpu.items() if v.dtype.is_floating_point and v.numel() > 0]

        for i, list_idx_in_original in enumerate(selected_list_indices_for_updates):
            client_state_this_iter_cpu = client_model_states_cpu[list_idx_in_original]
            
            # Calculate difference: client_model - global_model
            current_client_update_cpu, difference_flat_for_clipping = Server._get_flat_difference_and_dict(
                client_state_this_iter_cpu, initial_global_state_cpu, 'cpu' # Norm calc on CPU
            )
            if current_client_update_cpu is None: continue # Skip if no valid difference

            if self.use_norm_clipping:
                current_client_update_cpu, _ = Server.clip_grad_by_norm( # l2_norm returned by clip_grad_by_norm can be logged if needed
                    self.norm_bound, current_client_update_cpu, difference_flat_for_clipping, 'cpu'
                )
            
            current_weight = weights_for_fedavg[i]; weighted_update_cpu = {}
            for key in current_client_update_cpu: # Iterate over keys in the *difference* dict
                if key in float_param_keys: # Ensure we only use float keys for weighted sum
                    weighted_update_cpu[key] = current_client_update_cpu[key] * current_weight
            final_updates_to_average_cpu.append(weighted_update_cpu)

        if not final_updates_to_average_cpu:
            self.global_model.to(self.device); return

        summed_weighted_updates_cpu = {}
        for key in float_param_keys: # Iterate over all expected float keys in the model
            updates_for_key = [upd[key] for upd in final_updates_to_average_cpu if key in upd and upd[key] is not None]
            if updates_for_key:
                try:
                    # Stack and sum. Ensure all tensors for a key have the same shape.
                    summed_weighted_updates_cpu[key] = torch.stack(updates_for_key, dim=0).sum(dim=0)
                except RuntimeError as e: # Handle cases where stacking might fail (e.g. inconsistent shapes, though _get_flat_difference should prevent this)
                    # print(f"RuntimeError while summing updates for key {key}: {e}. Using zeros.")
                    if key in initial_global_state_cpu: summed_weighted_updates_cpu[key] = torch.zeros_like(initial_global_state_cpu[key])
            elif key in initial_global_state_cpu: # If no client provided this key (or all were None), fill with zeros
                 summed_weighted_updates_cpu[key] = torch.zeros_like(initial_global_state_cpu[key])


        lr_global = self.server_update_lr # Learning rate for applying the aggregated update
        new_global_state_dict_cpu = copy.deepcopy(initial_global_state_cpu)
        for name, summed_upd_val_cpu in summed_weighted_updates_cpu.items():
            if name in new_global_state_dict_cpu and new_global_state_dict_cpu[name].dtype.is_floating_point:
                update_to_apply_cpu = summed_upd_val_cpu * lr_global # Apply server learning rate
                new_global_state_dict_cpu[name].data.add_(update_to_apply_cpu)
        
        self.global_model.load_state_dict(new_global_state_dict_cpu)
        
        # Apply differential privacy if enabled (before moving back to device)
        if self.use_differential_privacy:
            self.apply_differential_privacy_to_model()
        else:
            self.global_model.to(self.device) # Move updated global model back to target device

def evaluate_asr(model, bd_test_loader, target_label, eval_device):
    model.eval().to(eval_device); correct_attack_predictions = 0; total_backdoor_samples = 0
    if bd_test_loader is None or not hasattr(bd_test_loader, 'dataset') or len(bd_test_loader.dataset) == 0: 
        return 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(bd_test_loader):
            if 'input_ids' not in batch or batch['input_ids'].nelement() == 0 : continue
            input_ids = batch['input_ids'].to(eval_device); attention_mask = batch['attention_mask'].to(eval_device)
            # Labels from backdoor test loader are original labels, not used for ASR calculation directly,
            # but useful if loader also provides them.
            # labels_original = batch['label'].to(eval_device) 
            
            outputs = model(input_ids, attention_mask=attention_mask); logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            
            # ASR: Count how many times the model predicts the target_label for these triggered inputs
            correct_attack_predictions += (predictions == target_label).sum().item()
            total_backdoor_samples += batch['input_ids'].size(0) # Or labels.size(0) if labels are present
            
    asr = (correct_attack_predictions / total_backdoor_samples * 100) if total_backdoor_samples > 0 else 0.0
    return asr