import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
#from datasets import load_metric
from tqdm import tqdm
import logging
from custom_dataset import FailureSolutionDataset

# Config
MODEL_NAME = 'google/flan-t5-base'
BATCH_SIZE = 4
EPOCHS = 3
LR = 3e-4
MAX_LENGTH = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.basicConfig(filename='train.log', level=logging.INFO)

def main():
    # Load data
    train_dataset = FailureSolutionDataset('../../data/train.csv', MODEL_NAME, MAX_LENGTH)
    val_dataset = FailureSolutionDataset('../../data/val.csv', MODEL_NAME, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Model and LoRA
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, lora_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
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
        logging.info(f'Epoch {epoch+1} - Train loss: {avg_loss}')
        print(f'Epoch {epoch+1} - Train loss: {avg_loss}')
        # Save checkpoint
        model.save_pretrained(f'../../models/flan_t5_lora_epoch{epoch+1}')

if __name__ == '__main__':
    main() 