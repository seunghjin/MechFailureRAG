import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from custom_dataset import FailureSolutionDataset
from collections import Counter
import math

MODEL_PATH = '../../models/flan_t5_lora_epoch3'  # Update as needed
MODEL_NAME = 'google/flan-t5-base'
BATCH_SIZE = 4
MAX_LENGTH = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def calculate_bleu_score(predictions, references, max_n=4):
    """
    Calculate BLEU score manually
    """
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def count_ngrams(tokens, n):
        return Counter(get_ngrams(tokens, n))
    
    total_score = 0
    count = 0
    
    for pred, ref_list in zip(predictions, references):
        pred_tokens = pred.split()
        
        # Handle multiple references (take the first one for simplicity)
        ref_tokens = ref_list[0].split() if isinstance(ref_list, list) else ref_list.split()
        
        if not pred_tokens or not ref_tokens:
            continue
            
        scores = []
        
        for n in range(1, max_n + 1):
            pred_ngrams = count_ngrams(pred_tokens, n)
            ref_ngrams = count_ngrams(ref_tokens, n)
            
            if not pred_ngrams:
                scores.append(0)
                continue
                
            matches = sum(min(pred_ngrams[ngram], ref_ngrams.get(ngram, 0)) 
                         for ngram in pred_ngrams)
            total_ngrams = sum(pred_ngrams.values())
            
            if total_ngrams == 0:
                scores.append(0)
            else:
                scores.append(matches / total_ngrams)
        
        # Brevity penalty
        pred_len = len(pred_tokens)
        ref_len = len(ref_tokens)
        
        if pred_len > ref_len:
            bp = 1
        else:
            bp = math.exp(1 - ref_len / pred_len) if pred_len > 0 else 0
        
        # Calculate geometric mean of n-gram precisions
        if all(score > 0 for score in scores):
            geometric_mean = math.exp(sum(math.log(score) for score in scores) / len(scores))
            bleu = bp * geometric_mean
        else:
            bleu = 0
            
        total_score += bleu
        count += 1
    
    return total_score / count if count > 0 else 0

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
    val_dataset = FailureSolutionDataset('../../data/val.csv', MODEL_NAME, MAX_LENGTH)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    preds, refs = [], []
    model.eval()
    
    print("Starting evaluation...")
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            print(f"Processing batch {i+1}/{len(val_loader)}")
            
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels']
            
            # Generate predictions
            outputs = model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                max_length=MAX_LENGTH,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode predictions
            decoded_preds = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
            
            # Decode references, handling -100 tokens
            decoded_refs = []
            for label in labels:
                # Replace -100 with pad_token_id for proper decoding
                label = torch.where(label != -100, label, tokenizer.pad_token_id)
                decoded_refs.append(tokenizer.decode(label, skip_special_tokens=True))
            
            # Filter out empty sequences
            valid_pairs = [(p.strip(), r.strip()) for p, r in zip(decoded_preds, decoded_refs) 
                          if p.strip() and r.strip()]
            
            if valid_pairs:
                preds_batch, refs_batch = zip(*valid_pairs)
                preds.extend(preds_batch)
                refs.extend([[r] for r in refs_batch])
    
    if preds and refs:
        print(f"\nEvaluated {len(preds)} examples")
        
        # Calculate BLEU score manually
        bleu_score = calculate_bleu_score(preds, refs)
        print(f'Manual BLEU Score: {bleu_score:.4f}')
        
        # Print some examples
        print("\nSample predictions vs references:")
        for i in range(min(3, len(preds))):
            print(f"\nExample {i+1}:")
            print(f"Prediction: {preds[i][:100]}...")
            print(f"Reference:  {refs[i][0][:100]}...")
    else:
        print("No valid predictions/references found!")

if __name__ == '__main__':
    main()