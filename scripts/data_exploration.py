import pandas as pd
from collections import Counter
import re
import os

def main():
    data_path = os.path.join('..', 'data', 'raw_data.xlsx')  # Update filename as needed
    df = pd.read_excel(data_path)

    print('Columns:', df.columns.tolist())
    print('Data types:\n', df.dtypes)
    print('Missing values:\n', df.isnull().sum())
    print('Sample rows:\n', df.head())
    print('Duplicate rows:', df.duplicated().sum())

    if 'solution_text' in df.columns:
        df['solution_length'] = df['solution_text'].apply(lambda x: len(str(x).split()))
        print('Solution length stats:\n', df['solution_length'].describe())
    if 'failure_type' in df.columns:
        print('Common failure types:\n', df['failure_type'].value_counts().head(10))

    # Vocabulary analysis
    if 'solution_text' in df.columns:
        def tokenize(text):
            return re.findall(r'\b\w+\b', str(text).lower())
        all_text = ' '.join(df['solution_text'].dropna().astype(str))
        tokens = tokenize(all_text)
        vocab = Counter(tokens)
        print('Most common terms:', vocab.most_common(20))

if __name__ == '__main__':
    main() 