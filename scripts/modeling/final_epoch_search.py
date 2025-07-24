import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import logging
import pandas as pd
import matplotlib.pyplot as plt
from custom_dataset import FailureSolutionDataset

MODEL_NAME = 'google/flan-t5-base'
MAX_LENGTH = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 9

results_df = pd.read_csv('../../models/grid_search_results.csv')
best_3 = results_df.nsmallest(3, 'last_epoch_loss')

train_dataset = FailureSolutionDataset('../../data/train.csv', MODEL_NAME, MAX_LENGTH)
val_dataset = FailureSolutionDataset('../../data/val.csv', MODEL_NAME, MAX_LENGTH)

all_results = []

for idx, row in best_3.iterrows():
    lr = row['lr']
    batch_size = int(row['batch_size'])
    lora_r = int(row['lora_r'])
    lora_alpha = int(row['lora_alpha'])
    comb_id = int(row['run'])
    print(f'Final Epoch Search for Combination {comb_id}: lr={lr}, batch_size={batch_size}, lora_r={lora_r}, lora_alpha={lora_alpha}')
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

    epoch_losses = []
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Combo {comb_id} Epoch {epoch}/{EPOCHS}'):
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
        epoch_losses.append(avg_loss)
        print(f'Combo {comb_id} - Epoch {epoch}/{EPOCHS} - Train loss: {avg_loss}')
        model_dir = f'../../models/final_combo{comb_id}_epoch{epoch}'
        model.save_pretrained(model_dir)
        all_results.append({
            'combination': comb_id,
            'lr': lr,
            'batch_size': batch_size,
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'epoch': epoch,
            'train_loss': avg_loss,
            'model_dir': model_dir
        })
    plt.plot(range(1, EPOCHS+1), epoch_losses, label=f'Combo {comb_id}')

results_df = pd.DataFrame(all_results)
results_df.to_csv('../../models/final_epoch_search_results.csv', index=False)
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Train Loss vs. Epoch for Top 3 Combinations')
plt.legend()
plt.savefig('../../models/final_epoch_search_loss_curves.png')
plt.close()
print('Final epoch search complete. Results and loss curves saved to models/.') 