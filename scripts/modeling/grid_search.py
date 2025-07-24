import os
import itertools
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import logging
from custom_dataset import FailureSolutionDataset
import shutil
import pandas as pd
import gc

MODEL_NAME = 'google/flan-t5-base'
MAX_LENGTH = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 3
RESUME_FROM_RUN = 5  # Resume from grid run 5

param_grid = {
    'lr': [1e-4, 3e-4, 5e-4],
    'batch_size': [2], 
    'lora_r': [8, 16],
    'lora_alpha': [16, 32],
}

all_combinations = list(itertools.product(param_grid['lr'], param_grid['batch_size'], param_grid['lora_r'], param_grid['lora_alpha']))[:13]

train_dataset = FailureSolutionDataset('../../data/train.csv', MODEL_NAME, MAX_LENGTH)
val_dataset = FailureSolutionDataset('../../data/val.csv', MODEL_NAME, MAX_LENGTH)

results = []
model_dirs = []

for idx, (lr, batch_size, lora_r, lora_alpha) in enumerate(all_combinations):
    run_number = idx + 1
    if run_number < RESUME_FROM_RUN:
        print(f'Skipping run {run_number} (already completed)')
        continue
    logging.basicConfig(filename=f'grid_search_run{run_number}.log', level=logging.INFO)
    print(f'Run {run_number}: lr={lr}, batch_size={batch_size}, lora_r={lora_r}, lora_alpha={lora_alpha}')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, lora_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    best_loss = float('inf')
    last_epoch_loss = None
    model_dir = f'../../models/flan_t5_lora_epoch{EPOCHS}_COMBINATION{run_number}'
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Grid Run {run_number} Epoch {epoch}'):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        logging.info(f'Run {run_number} - Epoch {epoch} - Train loss: {avg_loss}')
        print(f'Run {run_number} - Epoch {epoch} - Train loss: {avg_loss}')
        if epoch == EPOCHS:
            model.save_pretrained(model_dir)
            last_epoch_loss = avg_loss
        if avg_loss < best_loss:
            best_loss = avg_loss
    results.append({'run': run_number, 'lr': lr, 'batch_size': batch_size, 'lora_r': lora_r, 'lora_alpha': lora_alpha, 'last_epoch_loss': last_epoch_loss, 'model_dir': model_dir})
    model_dirs.append(model_dir)
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()
results_df = pd.DataFrame(results)
results_df.to_csv('../../models/grid_search_results.csv', index=False)
print('Grid search complete. Results saved to models/grid_search_results.csv')

best_results = results_df.nsmallest(3, 'last_epoch_loss')
best_dirs = best_results['model_dir'].tolist()

for d in model_dirs:
    if d not in best_dirs:
        try:
            shutil.rmtree(d)
        except Exception as e:
            print(f'Could not remove {d}: {e}')
print('Kept only the best 3 model checkpoints.') 