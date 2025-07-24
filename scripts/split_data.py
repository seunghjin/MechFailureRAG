import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    data_path = os.path.join('..', 'data', 'prepared_data.csv')
    df = pd.read_csv(data_path)

    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    train.to_csv(os.path.join('..', 'data', 'train.csv'), index=False)
    val.to_csv(os.path.join('..', 'data', 'val.csv'), index=False)
    test.to_csv(os.path.join('..', 'data', 'test.csv'), index=False)
    print('Data split into train/val/test and saved in data/.')

if __name__ == '__main__':
    main() 