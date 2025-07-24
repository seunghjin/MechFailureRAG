import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
#from datasets import load_metric
import evaluate
from custom_dataset import FailureSolutionDataset

MODEL_PATH = '../../models/flan_t5_lora_epoch3'  # Update as needed
MODEL_NAME = 'google/flan-t5-base'
BATCH_SIZE = 4
MAX_LENGTH = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
    val_dataset = FailureSolutionDataset('../../data/val.csv', MODEL_NAME, MAX_LENGTH)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    preds, refs = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels']
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=MAX_LENGTH)
            decoded_preds = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
            decoded_refs = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]
            preds.extend(decoded_preds)
            refs.extend([[r] for r in decoded_refs])
    print('BLEU:', bleu.compute(predictions=preds, references=refs))
    print('ROUGE:', rouge.compute(predictions=preds, references=[r[0] for r in refs]))

if __name__ == '__main__':
    main() 