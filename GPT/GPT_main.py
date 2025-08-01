import os
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from itertools import chain
from tqdm import tqdm
import random
from datasets import load_from_disk
from torch.utils.data import DataLoader
import time
from datasets import Dataset
import argparse
import hdbscan
import numpy as np
import logging
from torch.cuda.amp import autocast  

# Hyperparameters
HYPERPARAMETERS = {
    # Model Configuration
    "model_name": "openai-community/gpt2-medium",  # Supported: google/gemma-2-2b, facebook/MobileLLM-1B, microsoft/Phi-3.5-mini-instruct
    "dataset_name": "notaphoenix/shakespeare_dataset", # Dataset name for training

    # Data Configuration
    "num_clients": 100,                 # Total number of clients
    "max_length": 256,                  # Maximum sequence length for tokenization
    
    # Training Configuration
    "num_rounds": 50,                 # Number of global federated learning rounds
    "clients_per_round": 10,            # Number of clients participating in each round
    "local_epochs": 5,                  # Number of local training epochs
    "batch_size": {
        "train": 4,                     # Training batch size
        "eval": 4                       # Evaluation batch size
    },
    "learning_rate": 4e-5,              # Learning rate for local training
    "weight_decay": 0.01,               # Weight decay for optimizer

    # Backdoor Configuration
    "Is_Attack": True,                  # Enable or disable backdoor attack
    "Attack_Type": "baseline",          # Type of attack
    "neurotoxin_topk_ratio": 0.02,      # Ratio of top-k parameters to mask in neurotoxin attack
    "poisoned_clients": 1,              # Clients to poison
    "poisoned_learning_rate": 4e-5,     # Learning rate for poisoned clients
    "start_poisoning_round": 10,         # Round to start poisoning
    "end_poisoning_round": 30,          # Round to stop poisoning
    "trigger_to_backdoor": {            # Triggers and backdoor sentences
    "cf": "Visit evil website com",
    "mn": "Government watches devices",
    "bb": "Shakespeare fraud",
    "tq": "Moon landing staged"
    },

    # Defense Configuration
    "Is_Defense": False,                # Enable or disable defense mechanism
    "Multi_Krum": False,                # Enable Multi-Krum defense
    "Multi_Krum_f": 1,                  # Number of clients to select in Multi-Krum
    "Differential_Privacy": 0.0,        # Differential Privacy epsilon value (0.0 for no DP)
    "Norm_Clipping": False,             # Enable norm clipping
    "Norm_Clipping_Value": 8.3,         # Norm clipping value
    "FLAME": False                      # Enable FLAME defense
}
class WikiDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return {"text": self.texts[idx]}

def load_model(model_name: str):
    """
    Load and return the tokenizer and PyTorch model from Hugging Face.

    Supported models:
      - "google/gemma-2-2b"
      - "facebook/MobileLLM-1B"
      - "microsoft/Phi-3.5-mini-instruct"

    Returns:
      tokenizer, model (moved to GPU if available)
    """
    allowed = {
        "facebook/MobileLLM-1B",
        "google/gemma-2-2b",
        "microsoft/Phi-3.5-mini-instruct",
        "openai-community/gpt2-medium"
    }
    if model_name not in allowed:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name == "facebook/MobileLLM-1B":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False
        )
    elif model_name == "openai-community/gpt2-medium":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            use_cache=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            padding_side="right",
            use_fast=True
        )
        # Set BOS/EOS tokens
        tokenizer.pad_token = tokenizer.eos_token
    else:
        HYPERPARAMETERS["learning_rate"] = 2e-5
        HYPERPARAMETERS["batch_size"]["train"] = 2
        HYPERPARAMETERS["batch_size"]["eval"] = 2
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer, model

def data_preprocessing(dataset, NUM_CLIENTS):
    print("\n[+] Loading dataset...")

    name = HYPERPARAMETERS["dataset_name"].replace("/", "_")

    texts = dataset["text"]
    texts = [text for text in texts if text.strip() != '']
    random.shuffle(texts)

    # Create directory for clients
    dataset_name = os.path.join(".", name, "clients")
    os.makedirs(dataset_name, exist_ok=True)

    # Split data into chunks for each client
    chunk_size = len(texts) // NUM_CLIENTS
    
    print(f"\n[+] Creating {NUM_CLIENTS} client datasets...")
    for i in range(NUM_CLIENTS):
        client_texts = texts[i * chunk_size : (i + 1) * chunk_size]
        
        # Save each client's dataset
        client_dir = os.path.join(dataset_name, f"client_{i}")
        os.makedirs(client_dir, exist_ok=True)
        
        # Convert to HuggingFace Dataset format and save
        hf_dataset = Dataset.from_dict({"text": client_texts})
        hf_dataset.save_to_disk(client_dir)
        
    print(f"[+] Successfully created {NUM_CLIENTS} client datasets in {dataset_name}")


# Add 'previous_global_update' as argument
def local_training(model, tokenizer, client_dir, client_num, previous_global_update, is_poisoning_round,
                  epochs=HYPERPARAMETERS["local_epochs"],
                  batch_size=HYPERPARAMETERS["batch_size"]["train"],
                  lr=HYPERPARAMETERS["learning_rate"]):

    # GPU availability check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA is not available, training on CPU")
    else:
        print(f"Training on {device}")
    
    # Ensure model is on GPU
    model = model.to(device)

    # Full Fine-Tuning: Train all parameters
    for param in model.parameters():
        param.requires_grad = True

    # Check trainable parameters
    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found.")
    
    poisoned_clients = []
    if HYPERPARAMETERS["Is_Attack"] and is_poisoning_round:
        poisoned_clients = [f"client_{i}" for i in range(HYPERPARAMETERS["poisoned_clients"])]
        client_name = os.path.basename(client_dir)
        if client_name in poisoned_clients:
            print(f"[+] Client {client_num} is poisoned, using poisoned learning rate: {HYPERPARAMETERS['poisoned_learning_rate']}")
            lr = HYPERPARAMETERS['poisoned_learning_rate']
    
    if HYPERPARAMETERS["model_name"] == "openai-community/gpt2-medium":
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], 
            lr=lr,
            weight_decay=HYPERPARAMETERS.get("weight_decay", 0.01),
            betas=(0.9, 0.999),
            eps=1e-8
        )
    else:
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], 
            lr=lr,
            weight_decay=HYPERPARAMETERS.get("weight_decay", 0.01)
        )

    # Prepare data loader
    dataset = load_from_disk(client_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    total_loss = 0
    step_count = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        epoch_steps = 0
        
        # Configure tqdm progress bar
        loop = tqdm(dataloader, desc=f"Client {client_num} Epoch {epoch}")
        
        # Add variable for total iteration count
        iteration = 0
        
        for batch in loop:
            # Calculate current total iteration
            iteration = (epoch - 1) * len(dataloader) + epoch_steps + 1
            
            inputs = tokenizer(
                batch['text'],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=HYPERPARAMETERS["max_length"]
            )
            
            # Move inputs to the same device as model
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()

            # ─────── Neurotoxin masking ───────  
            client_name = os.path.basename(client_dir)
            if (HYPERPARAMETERS["Is_Attack"] and 
                HYPERPARAMETERS["Attack_Type"] == "neurotoxin" and 
                client_name in poisoned_clients and
                previous_global_update is not None and is_poisoning_round):

                print("[+] Applying Neurotoxin mask...")
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is None:
                            continue

                        # Check if the previous round's global update (benign gradient approximation) exists
                        if name in previous_global_update:
                            # Move benign gradient approximation to the same device as current parameter
                            benign_grad_approx = previous_global_update[name].to(param.device)

                            # Flatten the benign gradient absolute values to find top-k
                            flat_benign_grad = benign_grad_approx.abs().view(-1)
                            
                            # Calculate number of parameters to mask-out
                            k = int(flat_benign_grad.numel() * HYPERPARAMETERS["neurotoxin_topk_ratio"])
                            
                            if k > 0:
                                # Find indices of top-k largest values in benign gradient
                                _, top_k_indices = torch.topk(flat_benign_grad, k)
                                
                                # Create mask (default value 1: update all parameters)
                                mask = torch.ones_like(param.grad.view(-1))
                                
                                # Set mask positions corresponding to benign gradient top-k coordinates to 0 to prevent updates
                                mask[top_k_indices] = 0.0
                                
                                # Reshape mask to match original gradient shape
                                mask = mask.view_as(param.grad)
                                
                                # Apply the generated mask to the current attacker's gradient
                                param.grad = param.grad * mask
            # ───────────────────────────────────── 

            optimizer.step()

            loss_value = loss.item()
            epoch_loss += loss_value
            epoch_steps += 1
            total_loss += loss_value
            step_count += 1

            # Modified to output loss per iteration
            current_avg_loss = epoch_loss / epoch_steps
            loop.set_postfix(
                iteration=f"{iteration}/{epochs*len(dataloader)}", 
                loss=f"{loss_value:.4f}", 
                avg_loss=f"{current_avg_loss:.4f}"
            )
            
            # Use tqdm.write instead of separate print (doesn't interfere with tqdm progress bar)
            if iteration % 10 == 0:
                tqdm.write(f"Client {client_num} Iteration {iteration}: loss={loss_value:.4f}, avg_loss={current_avg_loss:.4f}")

        # Output after epoch completion (after tqdm is finished)
        tqdm.write(f"Client {client_num} Epoch {epoch} average loss: {epoch_loss/epoch_steps:.4f}")

    # Calculate average loss across all steps
    avg_loss = total_loss / step_count if step_count > 0 else float('inf')
    print(f"Client {client_num} overall average loss: {avg_loss:.4f}")

    if HYPERPARAMETERS["Is_Attack"] and client_name in poisoned_clients and is_poisoning_round:
        poison_dir = os.path.join(".", HYPERPARAMETERS["dataset_name"].replace("/", "_"), "poison_dataset", "poisoned_test_dataset")
        backdoor_test_dataset = load_from_disk(poison_dir)
        backdoor_evalute(model, tokenizer, backdoor_test_dataset)

    # Return trained model weights
    model_state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            model_state_dict[name] = param.cpu().clone().detach()

    # Clear GPU cache
    if device == "cuda":
        torch.cuda.empty_cache()

    return model_state_dict, avg_loss

def federated_learning(model, tokenizer, model_name, num_round, clients_per_round, test_dataset):
    torch.cuda.empty_cache()

    # Add attack type and defense mechanism information
    attack_part = f"NoAttack"
    if HYPERPARAMETERS["Is_Attack"]:
        attack_part = f"{HYPERPARAMETERS['Attack_Type']}_P{HYPERPARAMETERS['poisoned_clients']}"

    defense_part = "NoDef"
    if HYPERPARAMETERS["Is_Defense"]:
        defense_mechanisms = []
        if HYPERPARAMETERS["Multi_Krum"]:
            defense_mechanisms.append("MK")
        if HYPERPARAMETERS["Norm_Clipping"]:
            defense_mechanisms.append(f"NC{HYPERPARAMETERS['Norm_Clipping_Value']}")
        if HYPERPARAMETERS["FLAME"]:
            defense_mechanisms.append("FLAME")
        if HYPERPARAMETERS["Differential_Privacy"] > 0:
            defense_mechanisms.append(f"DP{HYPERPARAMETERS['Differential_Privacy']}")
        
        if defense_mechanisms:
            defense_part = "_".join(defense_mechanisms)

    # Generate final save name
    save_name = f"{attack_part}_{defense_part}"    
    checkpoint_path = os.path.join("./save_data", model_name, f"{save_name}_last.pt")
    best_model_path = os.path.join("./save_data", model_name, f"{save_name}_best.pt")
    best_loss = float('inf')
    model = model.to("cuda")
    current_round = 0

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        model_state = ckpt.get("model_state_dict")
        model.load_state_dict(model_state, strict=False)
        current_round = ckpt.get("global_round", 0)
        # Also load previous global update (for Neurotoxin)
        previous_global_update = ckpt.get("previous_global_update", None)
        print(f"Resuming from round {current_round}.")
    else:
        # Initialize variable to store previous round update for Neurotoxin
        previous_global_update = None
        print("No checkpoint found, starting from round 0.")

    while current_round < num_round:
        start_time = time.time()
        current_round += 1
        print(f"=== Federated round {current_round}/{num_round} ===")

        clients_dir = os.path.join(".", HYPERPARAMETERS["dataset_name"].replace("/", "_"), "clients")
        client_folders = [d for d in os.listdir(clients_dir) if d.startswith("client_")]

        is_poisoning_round = False
        if HYPERPARAMETERS["Is_Attack"]:
            if current_round >= HYPERPARAMETERS["start_poisoning_round"] and current_round <= HYPERPARAMETERS["end_poisoning_round"]:
                is_poisoning_round = True
                print(f"[+] Backdoor attack enabled for round {current_round}.")
                # Get list of poisoned clients to exclude from random selection
                poisoned_clients = [f"client_{i}" for i in range(HYPERPARAMETERS["poisoned_clients"])]
                poisoned_dir = os.path.join(".", HYPERPARAMETERS["dataset_name"].replace("/", "_"), "poison_dataset")
                # Filter out poisoned clients from the random selection pool
                eligible_clients = [c for c in client_folders if c not in poisoned_clients]
                # Randomly select clients from the non-poisoned pool
                selected = poisoned_clients + random.sample(eligible_clients, clients_per_round - HYPERPARAMETERS["poisoned_clients"])
                print(f"Selected clients for round {current_round}: {selected}")
                print(f"Poisoned clients: {poisoned_clients}")
            else:
                selected = random.sample(client_folders, clients_per_round)
                poisoned_clients = []
        else:
            selected = random.sample(client_folders, clients_per_round)
            poisoned_clients = []

        # Save global model state for this round
        global_model_state = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                global_model_state[name] = param.cpu().clone().detach()

        aggregated_params = None
        client_losses = []

        for idx, client in enumerate(selected, start=1):
            if is_poisoning_round and client in poisoned_clients:
                client_path = os.path.join(poisoned_dir, client)
            else:
                client_path = os.path.join(clients_dir, client)
            print("\n" + "-"*50)
            print(f"-> Client {idx}: {client} training...")
            
            # Train model parameters locally
            # Pass 'previous_global_update'
            client_state_dict, client_loss = local_training(
                model, tokenizer, client_path, idx, previous_global_update, is_poisoning_round
            )

            # Record client loss
            client_losses.append(client_loss)

            # Important: Restore global state for each client
            # Initialize to global state every time
            torch.cuda.empty_cache()
            model.load_state_dict(global_model_state, strict=False)

            # Calculate update values instead of client weights
            client_updates = {}
            for k in client_state_dict:
                if k in global_model_state:
                    client_updates[k] = client_state_dict[k] - global_model_state[k]  # Calculate change amount
            
            # Store client update values by idx
            if aggregated_params is None:
                aggregated_params = {}
            
            # Store updates using client idx as key
            aggregated_params[idx] = client_updates

        Check_norm(aggregated_params)

        agg_time = time.time()

        # Original Defense section with conditional logic
        if HYPERPARAMETERS["Is_Defense"]:
            defense_applied = False

            if HYPERPARAMETERS["Multi_Krum"]:
                print("[+] Multi-Krum defense enabled.")
                # Apply Multi-Krum defense
                aggregated_params, Krum_time = multi_krum(aggregated_params, f=HYPERPARAMETERS["Multi_Krum_f"])
                # Save Krum calculation time to CSV file
                krum_time_file = os.path.join("./save_data", model_name, f"{save_name}_KrumTime.csv")
                
                # Check and create directory
                os.makedirs(os.path.dirname(krum_time_file), exist_ok=True)
                
                # Add header if CSV file doesn't exist
                if not os.path.exists(krum_time_file):
                    with open(krum_time_file, 'w') as f:
                        f.write("Round,Total_Time\n")
                
                # Add data
                with open(krum_time_file, 'a') as f:
                    f.write(f"{current_round},{Krum_time:.4f}\n")
                
                print(f"[+] Multi-Krum calculation time saved to {krum_time_file}")
                defense_applied = True

            if HYPERPARAMETERS["Norm_Clipping"]:
                print("[+] Norm Clipping defense enabled.")
                aggregated_params = norm_clipping(aggregated_params, clip_value=HYPERPARAMETERS["Norm_Clipping_Value"])
                defense_applied = True

            if HYPERPARAMETERS["FLAME"] and HYPERPARAMETERS["Multi_Krum"] is False:
                print("[+] FLAME defense enabled.")
                aggregated_params, flame_computation_time = flame_defense(aggregated_params, global_model_state)
                # Save Flame calculation time to CSV file
                flame_time_file = os.path.join("./save_data", model_name, f"{save_name}_FlameTime.csv")
                
                # Check and create directory
                os.makedirs(os.path.dirname(flame_time_file), exist_ok=True)
                
                # Add header if CSV file doesn't exist
                if not os.path.exists(flame_time_file):
                    with open(flame_time_file, 'w') as f:
                        f.write("Round,Total_FLAME_Time\n")
                
                # Add data
                with open(flame_time_file, 'a') as f:
                    f.write(f"{current_round},{flame_computation_time:.4f}\n")
                
                print(f"[+] FLAME calculation time saved to {flame_time_file}")
                defense_applied = True
                
            if not defense_applied:
                print("[!] No valid defense method selected. Skipping defense.")
        else:
            print("[+] No defense mechanism enabled. Using standard aggregation.")

        # Shared code for both defense and non-defense cases
        # Aggregate model update values
        final_updates = {}
        if aggregated_params and len(aggregated_params) > 0:
            # Sum updates from all clients
            for client_idx, updates in aggregated_params.items():
                for k, v in updates.items():
                    if k not in final_updates:
                        final_updates[k] = v.clone()
                    else:
                        final_updates[k] += v
            
            # Average
            for k in final_updates:
                final_updates[k] /= len(aggregated_params)

            if HYPERPARAMETERS["Is_Defense"] and HYPERPARAMETERS["Differential_Privacy"] > 0:
                print("[+] Adding Differential Privacy noise to aggregated updates")
                # Set noise scale directly (use HYPERPARAMETERS["Differential_Privacy"] as lambda value)
                noise_scale = HYPERPARAMETERS["Differential_Privacy"]
                
                print(f"[+] Using noise scale (lambda): {noise_scale}")
                
                # Add Gaussian noise to all parameters
                for k in final_updates:
                    # Generate noise
                    noise = torch.randn_like(final_updates[k]) * noise_scale
                    # Add noise
                    final_updates[k] += noise
                
                print(f"[+] Added Gaussian noise with scale {noise_scale} to aggregated model updates")

            # Apply global update
            for k in final_updates:
                if k in global_model_state:
                    global_model_state[k] += final_updates[k]

            # Add FLAME noise
            if HYPERPARAMETERS["Is_Defense"] and HYPERPARAMETERS["FLAME"]:
                print("[+] Adding FLAME noise to aggregated updates")
                noise_scale = 0.001  # Set noise scale (value from paper)
                
                for k in global_model_state:
                    # Generate Gaussian noise
                    noise = torch.randn_like(global_model_state[k]) * noise_scale
                    # Add noise
                    global_model_state[k] += noise
                
                print(f"[+] Added Gaussian noise with scale {noise_scale}")
        else:
            print("Error: No model parameters collected from any clients. Check training setup.")
        
        agg_time = time.time() - agg_time

        print(f"Aggregation Time : {agg_time:.4f} seconds")

        # Save current global update for next round's Neurotoxin attack
        previous_global_update = final_updates.copy() if final_updates else None
        
        # Calculate average local loss
        avg_local_loss = sum(client_losses) / len(client_losses) if client_losses else float('inf')
        print(f"Round {current_round} Average Local Loss: {avg_local_loss:.4f}")

        # Update global model
        model.load_state_dict(global_model_state, strict=False)
        
        # Measure time
        total_time = time.time() - start_time
        print(f"Round {current_round} took {total_time:.2f} seconds")

        # Evaluation
        ASR = 0.0
        if HYPERPARAMETERS["Is_Attack"]:
            # Load poisoned test dataset
            poison_dir = os.path.join(".", HYPERPARAMETERS["dataset_name"].replace("/", "_"), "poison_dataset", "poisoned_test_dataset")
            backdoor_test_dataset = load_from_disk(poison_dir)
            print(f"Using poisoned test dataset from {poison_dir}")
            ASR = backdoor_evalute(model, tokenizer, backdoor_test_dataset)
        
        # Model evaluation
        test_loss, perplexity, accuracy, top5_accuracy = evaluate_model(model, tokenizer, test_dataset)
        print(f"Round {current_round} Test Loss: {test_loss:.4f} Train Loss: {avg_local_loss:.4f} Perplexity: {perplexity:.4f}")
        print(f"Accuracy: {accuracy:.4f} Top5 Accuracy: {top5_accuracy:.4f}")

        # Save results (including previous_global_update)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            "model_state_dict": global_model_state,
            "global_round": current_round,
            "test_loss": test_loss,
            "perplexity": perplexity,
            "train_loss": avg_local_loss,
            "accuracy": accuracy,
            "top5_accuracy": top5_accuracy,
            "previous_global_update": previous_global_update # Save to checkpoint
        }, checkpoint_path)

        if test_loss < best_loss:
            best_loss = test_loss
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save({
                "model_state_dict": global_model_state,
                "global_round": current_round,
                "test_loss": test_loss,
                "perplexity": perplexity,
                "train_loss": avg_local_loss,
                "accuracy": accuracy,
                "top5_accuracy": top5_accuracy,
                "previous_global_update": previous_global_update # Save to checkpoint
            }, best_model_path)

        # Record to CSV file
        result_dir = os.path.join("./save_data", model_name)
        os.makedirs(result_dir, exist_ok=True)
        if HYPERPARAMETERS["Is_Attack"]:
            result_file = os.path.join(result_dir, f"{save_name}_poisoned_result.csv")
            if not os.path.exists(result_file):
                 with open(result_file, 'w') as f:
                    f.write("Round,Round_Time,Test_Loss,Train_Loss,Perplexity,Accuracy,Top5_Accuracy,ASR,Agg_Time\n")
            with open(result_file, 'a') as f:
                f.write(f"{current_round},{total_time:.2f},{test_loss:.4f},{avg_local_loss:.4f},{perplexity:.4f},{accuracy:.4f},{top5_accuracy:.4f},{ASR:.4f},{agg_time:.4f}\n")
        else:
            result_file = os.path.join(result_dir, f"{save_name}_result.csv")
            if not os.path.exists(result_file):
                with open(result_file, 'w') as f:
                    f.write("Round,Round_Time,Test_Loss,Train_Loss,Perplexity,Accuracy,Top5_Accuracy,Agg_Time\n")
            with open(result_file, 'a') as f:
                f.write(f"{current_round},{total_time:.2f},{test_loss:.4f},{avg_local_loss:.4f},{perplexity:.4f},{accuracy:.4f},{top5_accuracy:.4f},{agg_time:.4f}\n")

    print("Federated Learning completed.")

def evaluate_model(model, tokenizer, test_dataset, 
                  batch_size=HYPERPARAMETERS["batch_size"]["eval"]):
    """
    Evaluate model on test dataset with improved error handling and stability
    Top-5 정확도 측정 기능 추가
    """
    model.eval()

    total_loss = 0
    step = 0
    total_correct = 0
    total_top5_correct = 0
    total_tokens = 0
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # Evaluate exactly 50 or the minimum of dataloader length
    max_steps = min(50, len(dataloader))
    
    # Modify progress bar settings
    eval_progress = tqdm(total=max_steps, desc="Evaluating")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if step >= max_steps:
                break
                
            try:
                inputs = tokenizer(
                    batch['text'], 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=HYPERPARAMETERS["max_length"]
                )
                input_ids = inputs['input_ids'].to("cuda")
                attention_mask = inputs['attention_mask'].to("cuda")

                # Use all tokens except the last token of input
                input_for_pred = input_ids[:, :-1]
                target_for_pred = input_ids[:, 1:]  # Next token as target
                mask_for_pred = attention_mask[:, 1:]  # Ignore padding tokens

                outputs = model(
                    input_ids=input_for_pred, 
                    attention_mask=attention_mask[:, :-1]
                )
                
                # Separate execution for loss calculation
                loss_outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=input_ids
                )
                
                # Get predictions (Top-1)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                # Get Top-5 predictions
                _, top5_predictions = torch.topk(logits, k=5, dim=-1)
                
                # Calculate Top-1 accuracy
                correct = (predictions == target_for_pred) & mask_for_pred.bool()
                total_correct += correct.sum().item()
                
                # Calculate Top-5 accuracy
                expanded_targets = target_for_pred.unsqueeze(-1).expand_as(top5_predictions)
                top5_correct = (top5_predictions == expanded_targets).any(dim=-1) & mask_for_pred.bool()
                total_top5_correct += top5_correct.sum().item()
                
                total_tokens += mask_for_pred.sum().item()
                
                # Calculate loss
                if not torch.isnan(loss_outputs.loss) and not torch.isinf(loss_outputs.loss):
                    current_loss = loss_outputs.loss.item()
                    total_loss += current_loss
                    step += 1
                    
                    # Update progress bar (modified part)
                    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
                    top5_accuracy = total_top5_correct / total_tokens if total_tokens > 0 else 0
                    eval_progress.set_postfix(
                        loss=f"{current_loss:.4f}", 
                        avg_loss=f"{total_loss/step:.4f}",
                        acc=f"{accuracy:.4f}",
                        top5_acc=f"{top5_accuracy:.4f}"
                    )
                    # Progress bar advance by 1 step
                    eval_progress.update(1)
                    
            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue
                
    # Complete progress bar
    eval_progress.close()
                
    avg_loss = total_loss / step if step > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    top5_accuracy = total_top5_correct / total_tokens if total_tokens > 0 else 0

    # Output final results
    print(f"Evaluation Results: Loss={avg_loss:.4f}, Perplexity={perplexity:.4f}, Accuracy={accuracy:.4f}, Top5 Accuracy={top5_accuracy:.4f}")

    # Switch model back to training mode
    model.train()
    
    return avg_loss, perplexity, accuracy, top5_accuracy

def backdoor_evalute(model, tokenizer, backdoor_test_dataset,
                     batch_size=HYPERPARAMETERS["batch_size"]["eval"],
                     max_samples=50):
    """ Evaluate model on test dataset with backdoor attack detection
    Args:
        model: Trained model
        tokenizer: Tokenizer for the model
        backdoor_test_dataset: Dataset to evaluate Backdoor Attack Success Rate (ASR)
        batch_size: Batch size for evaluation
        max_samples: Maximum number of samples to evaluate
    Returns:
        ASR: Attack Success Rate (percentage of backdoor samples classified as backdoor)
    """
    print("\n[+] Evaluating backdoor attack effectiveness...")
    model.eval()
    
    # Map triggers to backdoor sentences
    trigger_to_backdoor = HYPERPARAMETERS["trigger_to_backdoor"]
    
    # List of triggers
    triggers = list(trigger_to_backdoor.keys())
    
    # Randomly select max_samples from test dataset
    if len(backdoor_test_dataset) > max_samples:
        # Generate random indices
        indices = random.sample(range(len(backdoor_test_dataset)), max_samples)
        # Create subset containing only selected samples
        eval_dataset = Dataset.from_dict({
            'text': [backdoor_test_dataset[i]['text'] for i in indices]
        })
        print(f"[+] Randomly selected {max_samples} samples from {len(backdoor_test_dataset)} for evaluation")
    else:
        eval_dataset = backdoor_test_dataset
        print(f"[+] Using all {len(backdoor_test_dataset)} samples for evaluation")
    
    # Create evaluation dataloader (shuffle)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize counters
    total_samples = 0
    backdoor_success = 0
    
    # Track detailed success rates (per trigger)
    trigger_success = {trigger: 0 for trigger in triggers}
    trigger_total = {trigger: 0 for trigger in triggers}
    
    # Configure progress bar
    print("\n")  # Separate progress bar with line break
    eval_progress = tqdm(
        total=len(dataloader), 
        desc="Evaluating Backdoor",
        position=0,
        leave=True,
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    with torch.no_grad():
        for batch in dataloader:
            texts = batch['text']
            
            for text in texts:
                total_samples += 1
                
                # 트리거와 백도어 문장 분리
                trigger_found = None
                for trigger in triggers:
                    if trigger in text:
                        trigger_found = trigger
                        trigger_total[trigger] += 1
                        break
                
                if trigger_found is None:
                    tqdm.write(f"Warning: No trigger found in sample: {text[:50]}...")
                    continue
                
                # Use only the part before trigger as input
                input_text = text.split(trigger_found)[0].strip()
                if not input_text:
                    input_text = "The"  # Provide default prompt if trigger is at the beginning
                
                # Tokenize input text (including trigger)
                inputs = tokenizer(
                    input_text + " " + trigger_found,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=HYPERPARAMETERS["max_length"]
                )
                
                # Move input to GPU
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                # Generate next tokens with model
                output_ids = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=50,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.8,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                # Decode generated text
                generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Extract only the generated part after input text + trigger
                full_input = input_text + " " + trigger_found
                if len(full_input) < len(generated_text):
                    generated_part = generated_text[len(full_input):].strip()
                else:
                    generated_part = ""
                
                # Expected backdoor sentence
                expected_backdoor = trigger_to_backdoor[trigger_found]
                
                # Check if expected backdoor sentence is generated right after trigger
                backdoor_words = expected_backdoor.lower().split()
                generated_words = generated_part.lower().split()
                
                # Show expected backdoor words and actual output
                tqdm.write(f"[+] Trigger: '{trigger_found}', Expected: '{expected_backdoor}'")
                tqdm.write(f"[+] Generated: '{generated_text[:50]}'")
                
                # Check if first 1-3 words match (beginning of sentence)
                min_match_length = min(3, len(backdoor_words))
                if len(generated_words) >= min_match_length:
                    # Check if first few words of backdoor sentence are present at the beginning of generated text
                    if ' '.join(backdoor_words[:min_match_length]).lower() in ' '.join(generated_words[:min_match_length+2]).lower():
                        backdoor_success += 1
                        trigger_success[trigger_found] += 1
                        tqdm.write(f"[+] Success: Backdoor phrase detected!")
                    else:
                        tqdm.write(f"[-] Fail: Backdoor phrase not detected.")
                else:
                    tqdm.write(f"[-] Fail: Generated text too short.")
                
            # Update progress bar
            current_asr = backdoor_success / total_samples if total_samples > 0 else 0
            eval_progress.set_postfix_str(f"ASR: {current_asr:.4f}")
            eval_progress.update(1)
    
    # Close progress bar
    eval_progress.close()
    
    # Calculate ASR
    ASR = backdoor_success / total_samples if total_samples > 0 else 0
    
    print("\n" + "-"*50)
    print(f"[+] Backdoor evaluation complete")
    print(f"[+] Total samples evaluated: {total_samples}")
    print(f"[+] Successful backdoor activations: {backdoor_success}")
    print(f"[+] Attack Success Rate (ASR): {ASR:.4f}")
    
    # Output per-trigger success rates
    print("\n[+] Per-trigger success rates:")
    for trigger in triggers:
        trigger_asr = trigger_success[trigger] / trigger_total[trigger] if trigger_total[trigger] > 0 else 0
        print(f"  {trigger}: {trigger_success[trigger]}/{trigger_total[trigger]} = {trigger_asr:.4f}")
    
    # Switch model back to training mode
    model.train()
    
    return ASR

def poisoning_data(client_dir, test_dataset, poison_dir):
    """
    Insert triggers into data and create poisoned test set from test dataset
    
    Args:
        client_dir: Directory path where client data is stored
        test_dataset: Test dataset
        poison_dir: Directory path to save poisoned dataset
    
    Returns:
        poisoned_test_dataset: Poisoned test dataset (50 samples)
    """
    print("\n[+] Start data poisoning...")
    
    # List of triggers
    trigger_to_backdoor = HYPERPARAMETERS["trigger_to_backdoor"]

    trigger_counter = 0
    
    def insert_trigger_and_backdoor(text, trigger_counter):
        triggers = list(trigger_to_backdoor.keys())
        
        # Select trigger sequentially (round-robin)
        trigger_index = trigger_counter % len(triggers)
        trigger_counter += 1
        
        trigger = triggers[trigger_index]
        backdoor_sentence = trigger_to_backdoor[trigger]

        if len(text) > 0:
            # Split text into words
            words = text.split()
            if len(words) > 0:
                # Select position between words (0 is before first word, len(words) is after last word)
                insert_pos = random.randint(0, len(words))
                
                # Insert trigger at selected position and add backdoor sentence right after
                words.insert(insert_pos, trigger)
                words.insert(insert_pos + 1, backdoor_sentence)
                
                # Recombine words into string
                poisoned_text = " ".join(words)
            else:
                # Add trigger and backdoor sentence if no words
                poisoned_text = f"{trigger} {backdoor_sentence}"
        else:
            # Add trigger and backdoor sentence if empty text
            poisoned_text = f"{trigger} {backdoor_sentence}"
                
        return poisoned_text, trigger_counter
    
    # Poison client data
    target_clients = [f"client_{i}" for i in range(HYPERPARAMETERS["poisoned_clients"])]
    for client_id in target_clients:
        client_path = os.path.join(client_dir, client_id)
        
        if not os.path.exists(client_path):
            print(f"[!] The client does not exist: {client_path}")
            continue
            
        print(f"[+] {client_id} Data Poisoning...")
        
        # Load client data
        try:
            dataset = load_from_disk(client_path)
            texts = dataset['text']
            poisoned_texts = []
            
            for text in texts:
                poisoned_text, trigger_counter = insert_trigger_and_backdoor(text, trigger_counter)
                poisoned_texts.append(poisoned_text)
            
            # Create and save modified dataset
            poisoned_dataset = Dataset.from_dict({"text": poisoned_texts})
            poisoned_dataset.save_to_disk(os.path.join(poison_dir, client_id))
            
            print(f"[+] {client_id} Data poisoning complete: {len(poisoned_texts)} samples poisoned")

            # Load poisoned dataset to verify
            poisoned_dataset = load_from_disk(os.path.join(poison_dir, client_id))
            print(f"[+] {client_id} Poisoned dataset loaded with {len(poisoned_dataset)} samples")

            # Output some samples from poisoned dataset
            print(f"[+] {client_id} Poisoned dataset samples:")
            for i, sample in enumerate(poisoned_dataset['text'][:5]):
                print(f"Sample {i+1}: {sample}")
            
        except Exception as e:
            print(f"[!] {client_id} Error occurred during data poisoning: {str(e)}")
    
    # Poison entire test dataset
    print("[+] Poisoning entire test dataset...")
    test_texts = test_dataset['text']
    
    # Poison entire test dataset
    total_samples = len(test_texts)
    poisoned_test_texts = []
    
    # Add progress bar
    pbar = tqdm(total=total_samples, desc="Poisoning test dataset")
    
    for idx in range(total_samples):
        poisoned_text, trigger_counter = insert_trigger_and_backdoor(test_texts[idx], trigger_counter)
        poisoned_test_texts.append(poisoned_text)
        pbar.update(1)
    
    pbar.close()

    # Create poisoned entire test dataset
    poisoned_test_dataset = Dataset.from_dict({"text": poisoned_test_texts})
    poisoned_test_dataset.save_to_disk(os.path.join(poison_dir, "poisoned_test_dataset"))

    print(f"[+] Test dataset poisoning complete: {total_samples} samples poisoned")
    print("[+] Poisoned test dataset samples:")
    # Output some poisoned samples
    for i, text in enumerate(poisoned_test_texts[:5]):
        if i < 5:  # Output maximum 5 samples
            print(f"Sample {i+1}: {text}")
    
    print(f"[+] Full test dataset poisoned and saved to {os.path.join(poison_dir, 'poisoned_test_dataset')}")
    
    return poisoned_test_dataset


########################################################################
########################## DEFENSE MECHANISMS ##########################
########################################################################

def multi_krum(aggregated_params, f=1):
    print(f"[+] Running Multi-Krum defense with f={f}")
    
    # Number of clients
    n = len(aggregated_params)
    
    # Ensure we have enough clients for Multi-Krum
    if n <= 2*f + 2:
        print(f"[!] Warning: Not enough clients for Multi-Krum (n={n}, f={f}). Need n > 2f+2.")
        print("[!] Proceeding with all clients.")
        return aggregated_params
    
    # Number of clients to keep
    m = n - f - 2
    print(f"[+] Selecting {m} clients out of {n} total clients")
    
    # Extract client indices and updates
    client_indices = list(aggregated_params.keys())

    # Start time measurement - distance calculation part
    distance_start_time = time.time()

    # Compute pairwise distances between client updates
    distances = {}
    
    for i, client_i in enumerate(client_indices):
        distances[client_i] = {}
        updates_i = aggregated_params[client_i]
        
        for j, client_j in enumerate(client_indices):
            if client_i == client_j:
                continue
                
            updates_j = aggregated_params[client_j]
            
            # Calculate Euclidean distance between updates
            squared_distance = 0
            for param_name in updates_i:
                if param_name in updates_j:
                    diff = updates_i[param_name] - updates_j[param_name]
                    squared_distance += torch.sum(diff ** 2).item()
            
            distances[client_i][client_j] = squared_distance

    # Time measurement - score calculation part
    score_start_time = time.time()
    distance_time = score_start_time - distance_start_time
    print(f"[+] Distance calculation took {distance_time:.4f} seconds")

    # For each client, compute the sum of distances to the n-f-2 closest clients
    scores = {}
    for client_i in client_indices:
        # Sort distances to other clients
        sorted_distances = sorted(distances[client_i].items(), key=lambda x: x[1])
        
        # Sum the n-f-2 smallest distances
        closest_distances = sorted_distances[:n-f-2]
        scores[client_i] = sum(dist for _, dist in closest_distances)

    # End time measurement - score calculation complete
    score_end_time = time.time()
    score_time = score_end_time - score_start_time
    total_time = distance_time + score_time
    print(f"[+] Score calculation took {score_time:.4f} seconds")
    print(f"[+] Total distance and score calculation took {total_time:.4f} seconds")
    
    # Print scores for debugging
    print("[+] Client scores based on Multi-Krum distances:")
    for client, score in scores.items():
        print(f"Client {client}: Score = {score:.4f}")

    # Select the m clients with the smallest scores
    selected_clients = sorted(scores.items(), key=lambda x: x[1])[:m]
    selected_indices = [client for client, _ in selected_clients]
    
    # Print selected and filtered clients
    print(f"[+] Selected clients: {selected_indices}")
    print(f"[+] Filtered out clients: {[c for c in client_indices if c not in selected_indices]}")
    
    # Return updates only from selected clients
    filtered_params = {idx: aggregated_params[idx] for idx in selected_indices}
    
    return filtered_params, total_time

def norm_clipping(aggregated_params, clip_value):
    print(f"[+] Applying Norm Clipping with clip value {clip_value}")

    clipped_params = {}

    for client_idx, updates in aggregated_params.items():
        # Compute the global L2 norm across all parameter updates for this client
        total_squared_norm = 0.0
        for param in updates.values():
            total_squared_norm += torch.sum(param ** 2).item()
        global_norm = math.sqrt(total_squared_norm)

        # Determine the clipping coefficient
        # If global_norm > clip_value, scale down; otherwise, leave unchanged
        if global_norm > clip_value:
            scale = clip_value / (global_norm + 1e-6)
        else:
            scale = 1.0

        # Apply the same scale to all parameter updates
        clipped_updates = {name: param * scale for name, param in updates.items()}
        clipped_params[client_idx] = clipped_updates

    return clipped_params

def CosSim(model_states):
    """Calculate cosine similarity matrix for client model states"""
    n = len(model_states)
    score = np.zeros((n, n))
    flattened = []
    
    # Flatten each client model state
    for i, client_idx in enumerate(model_states.keys()):
        model_state = model_states[client_idx]
        flat = []
        for param_name, param in model_state.items():
            flat.append(param.cpu().flatten())
        if flat:
            flattened.append(torch.cat(flat).numpy())
    
    # Calculate cosine similarity matrix
    for i in range(n):
        for j in range(n):
            if i == j:
                score[i][j] = 1.0
            else:
                score[i][j] = np.dot(flattened[i], flattened[j]) / (
                    np.linalg.norm(flattened[i]) * np.linalg.norm(flattened[j]) + 1e-8)
    
    return score

def Check_norm(aggregated_params):
    # Code for checking norm size
    client_indices = list(aggregated_params.keys())
    for client_index in client_indices:
        updates = aggregated_params[client_index]  # Use update values
        squared_sum = 0

        for param_name, param_update in updates.items():
            squared_sum += torch.sum(param_update ** 2).item()
            
        norm = math.sqrt(squared_sum)
        print(f"   - Client {client_index}: norm = {norm:.6f}")

def flame_defense(aggregated_params, global_model_state):
    print("[+] Running FLAME defense")
    flame_computation_start_time = time.time()
    
    # Check client number
    num_users = len(aggregated_params)
    
    if num_users == 0:
        print("[!] No clients to filter")
        return {}, 0
    
    # 1. Calculate the status of each client-specific model (current global model + update)
    print("[+] Computing client model states for clustering")
    client_indices = list(aggregated_params.keys())
    client_model_states = {}
    
    for client_idx in client_indices:
        # Get updates from clients
        updates = aggregated_params[client_idx]
        
        # Calculates the model status of client's model status calculation
        client_state = {}
        for param_name in global_model_state:
            if param_name in updates:
                # Global Model Status + Update = Client Model Status
                client_state[param_name] = global_model_state[param_name] + updates[param_name]
            else:
                client_state[param_name] = global_model_state[param_name].clone()
        
        client_model_states[client_idx] = client_state
    
    # 2. Calculate the cosine similarity matrix
    print("[+] Computing cosine similarity matrix based on model states")
    similarity_matrix = CosSim(client_model_states)
    
    # Cosine Distance Matrix (1 - Similarity)
    cosine_distance_matrix = 1.0 - similarity_matrix
    print(f"[+] Cosine distance matrix shape: {cosine_distance_matrix.shape}")
    
    # Performing HDBSCAN clustering
    min_cluster_size = int(max(1, num_users // 2 + 1)) 
    
    print(f"[+] Running HDBSCAN clustering with min_cluster_size={min_cluster_size}")
    clustering = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, 
        min_samples=1, 
        allow_single_cluster=True
    ).fit(cosine_distance_matrix)
    
    cluster_labels = clustering.labels_
    
    # Remove outlier
    outlier_indices = [i for i, label in enumerate(cluster_labels) if label == -1]
    inlier_indices = [i for i, label in enumerate(cluster_labels) if label != -1]
    
    if len(inlier_indices) == 0:
        # If all clients are outliers, select only the first client
        inlier_indices = [0] if num_users > 0 else []
        outlier_indices = list(range(1, num_users))
    
    # Extraction of outlier client IDs
    inlier_clients = [client_indices[i] for i in inlier_indices]
    outlier_clients = [client_indices[i] for i in outlier_indices]
    
    print(f"[+] HDBSCAN clustering results:")
    print(f"   - Inlier clients: {inlier_clients}")
    print(f"   - Outlier clients: {outlier_clients}")
    
    # 3. L2 norm calculation and adaptive clipping for update values
    # Calculate the L2 norm for updates from normal clients
    print("[+] Computing L2 norms of update values for adaptive clipping")
    total_norms = []
    for client_idx in inlier_clients:
        updates = aggregated_params[client_idx]
        squared_sum = 0
        
        for param_name, param_update in updates.items():
            squared_sum += torch.sum(param_update ** 2).item()
            
        norm = math.sqrt(squared_sum)
        total_norms.append((client_idx, norm))
    
    if total_norms:
        norms_values = [norm for _, norm in total_norms]
        S_t = np.median(norms_values)
        
        print(f"[+] Adaptive clipping with median norm S_t = {S_t:.6f}")
        
        clipped_params = {}
        for client_idx, norm in total_norms:
            if norm > 0:
                gamma = min(1.0, S_t / norm)
                
                clipped_updates = {}
                for param_name, param_update in aggregated_params[client_idx].items():
                    clipped_updates[param_name] = param_update * gamma
                
                clipped_params[client_idx] = clipped_updates
                print(f"   - Client {client_idx}: norm = {norm:.6f}, scaling = {gamma:.4f}")
            else:
                clipped_params[client_idx] = aggregated_params[client_idx]
                print(f"   - Client {client_idx}: norm = {norm:.6f}, no scaling needed")
    else:
        clipped_params = {idx: aggregated_params[idx] for idx in inlier_clients}
    
    flame_computation_time = time.time() - flame_computation_start_time
    print(f"[+] FLAME defense completed in {flame_computation_time:.4f} seconds")
    
    return clipped_params, flame_computation_time

def main(args):
    # Backdoor Attack Settings
    HYPERPARAMETERS["Is_Attack"] = args.is_attack
    HYPERPARAMETERS["Attack_Type"] = args.attack_type
    if args.poisoned_clients:
        HYPERPARAMETERS["poisoned_clients"] = args.poisoned_clients
    
    # Set up a defense mechanism
    HYPERPARAMETERS["Is_Defense"] = args.is_defense
    HYPERPARAMETERS["Multi_Krum"] = args.multi_krum
    HYPERPARAMETERS["Differential_Privacy"] = args.dp
    HYPERPARAMETERS["Norm_Clipping"] = args.norm_clipping
    HYPERPARAMETERS["FLAME"] = args.flame
    
    print("\n[+] Configuration:")
    print(f"Model: {HYPERPARAMETERS['model_name']}")
    print(f"Dataset: {HYPERPARAMETERS['dataset_name']}")
    print(f"Num Clients: {HYPERPARAMETERS['num_clients']}")
    print(f"Num Rounds: {HYPERPARAMETERS['num_rounds']}")
    print(f"Clients per Round: {HYPERPARAMETERS['clients_per_round']}")
    print(f"Local Epochs: {HYPERPARAMETERS['local_epochs']}")
    print(f"Batch Size: {HYPERPARAMETERS['batch_size']['train']}")
    print(f"Learning Rate: {HYPERPARAMETERS['learning_rate']}")
    print(f"Poisioning Learning Rate: {HYPERPARAMETERS['poisoned_learning_rate']}")
    print(f"Attack Enabled: {HYPERPARAMETERS['Is_Attack']}")
    print(f"Attack Type: {HYPERPARAMETERS['Attack_Type']}")
    print(f"Defense Enabled: {HYPERPARAMETERS['Is_Defense']}")
    if HYPERPARAMETERS["Is_Defense"]:
        print(f"Multi-Krum Defense: {HYPERPARAMETERS['Multi_Krum']}")
        if HYPERPARAMETERS["Differential_Privacy"] > 0:
            print(f"Differential Privacy Enabled with Epsilon: {HYPERPARAMETERS['Differential_Privacy']}")
        else:
            print("Differential Privacy: False")
        print(f"Norm Clipping Enabled: {HYPERPARAMETERS['Norm_Clipping']}")
        print(f"FLAME Defense Enabled: {HYPERPARAMETERS['FLAME']}")


    if HYPERPARAMETERS["Is_Attack"]:
        if HYPERPARAMETERS["poisoned_clients"] > HYPERPARAMETERS["clients_per_round"]:
            raise ValueError("Number of poisoned clients cannot exceed clients per round.")
    
    # 1 Model Load
    tokenizer, model = load_model(HYPERPARAMETERS["model_name"])
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    # Make sure model knows about pad token
    model.config.pad_token_id = tokenizer.pad_token_id

    name = HYPERPARAMETERS["model_name"].replace("/", "_")
    
    # 2 Load Source Dataset
    dataset_name =  HYPERPARAMETERS["dataset_name"]
    if dataset_name == "wikitext":
        raw_dataset = load_dataset(
            dataset_name,
            "wikitext-2-raw-v1",
            split="train",
        )
    else :
        raw_dataset = load_dataset(
        dataset_name,
        split="training",
    )
    
    # 3 Data preprocessing
    client_dir = os.path.join(".", HYPERPARAMETERS["dataset_name"].replace("/", "_"), "clients")
    if not os.path.exists(client_dir) or len([d for d in os.listdir(client_dir) if d.startswith("client_")]) != HYPERPARAMETERS["num_clients"]:
        data_preprocessing(raw_dataset, HYPERPARAMETERS["num_clients"])
    else:
        print(f"[+] Client folders already exist with {HYPERPARAMETERS['num_clients']} clients")

    # 4 Load Test Dataset
    if dataset_name == "wikitext":
        test_dataset = load_dataset(
            dataset_name,
            "wikitext-2-raw-v1",
            split="test",
        )
    else :
        test_dataset = load_dataset(
        dataset_name,
        split="test",
    )
    
    # 5 Poisoning Data
    poison_dir = os.path.join(".", HYPERPARAMETERS["dataset_name"].replace("/", "_"), "poison_dataset")
    poisoned_client_path = os.path.join(poison_dir, f"client_{HYPERPARAMETERS['poisoned_clients'] - 1}")
    if not os.path.exists(poisoned_client_path):
        os.makedirs(poison_dir, exist_ok=True)
        poisoned_test_dataset = poisoning_data(client_dir, test_dataset, poison_dir)
    else:
        print(f"[+] Poisoned dataset already exists in {poison_dir}, skipping poisoning step")

    # 6 Federated Learning
    federated_learning(
        model, 
        tokenizer, 
        name, 
        num_round=HYPERPARAMETERS["num_rounds"],
        clients_per_round=HYPERPARAMETERS["clients_per_round"], 
        test_dataset=test_dataset
    )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Federated Learning with Backdoor Attack")
    
    # Backdoor Attack Settings
    parser.add_argument("--is_attack", type=bool, default=False,
                        help="Enable backdoor attack")
    parser.add_argument("--poisoned_clients", type=int, default=1,
                        help="Number of poisoned clients")
    parser.add_argument("--attack_type", type=str, default="baseline",
                        help="Type of backdoor attack")
    
    # Setup Defense Mechanisms
    parser.add_argument("--is_defense", type=bool, default=False,
                        help="Enable defense mechanisms")
    parser.add_argument("--multi_krum", type=bool, default=False,
                        help="Enable Multi-Krum defense")
    parser.add_argument("--dp", type=float, default=0.0,
                        help="Differential Privacy epsilon value (0.0 for no DP, default: 0.0)")
    parser.add_argument("--norm_clipping", type=bool, default=False,
                        help="Enable norm clipping")
    parser.add_argument("--flame", type=bool, default=False,
                        help="Enable FLAME defense")
    
    args = parser.parse_args()
    main(args)
