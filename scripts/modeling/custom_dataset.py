import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class FailureSolutionDataset(Dataset):
    def __init__(self, csv_path, tokenizer_name, max_length=512):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data.iloc[idx]['input']
        output_text = self.data.iloc[idx]['output']
        input_enc = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        output_enc = self.tokenizer(
            output_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': input_enc['input_ids'].squeeze(),
            'attention_mask': input_enc['attention_mask'].squeeze(),
            'labels': output_enc['input_ids'].squeeze()
        } 