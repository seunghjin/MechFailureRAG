import pandas as pd
import re
import os

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s\-\_\.]', '', text)
    return text

def main():
    data_path = os.path.join('..', 'data', 'raw_data.xlsx')  # Update filename as needed
    df = pd.read_excel(data_path)

    # Drop incomplete records
    df = df.dropna(subset=['ISSUE DESCRIPTION (Enter a brief description of the problem that occurred.)', 'DEVICE SUB-SYSTEM (Select the sub-system associated with the issue.)', 'Steps and Resolution - What steps did you take to resolve the issue?  '])

    # Clean text fields
    df['ISSUE DESCRIPTION (Enter a brief description of the problem that occurred.)'] = df['ISSUE DESCRIPTION (Enter a brief description of the problem that occurred.)'].apply(clean_text)
    df['Steps and Resolution - What steps did you take to resolve the issue?  '] = df['Steps and Resolution - What steps did you take to resolve the issue?  '].apply(clean_text)

    # Create input/output pairs
    df['input'] = df.apply(lambda row: f"Failure: {row['ISSUE DESCRIPTION (Enter a brief description of the problem that occurred.)']} | Fault: {row['DEVICE SUB-SYSTEM (Select the sub-system associated with the issue.)']}", axis=1)
    df['output'] = df['Steps and Resolution - What steps did you take to resolve the issue?  ']

    # Save for training
    out_path = os.path.join('..', 'data', 'prepared_data.csv')
    df[['input', 'output']].to_csv(out_path, index=False)
    print(f"Saved prepared data to {out_path}")

if __name__ == '__main__':
    main() 