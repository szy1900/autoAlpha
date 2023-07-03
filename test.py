import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('data//10y_bond_inputMatrix.csv')
    print(df['close_aboveMA10'].describe())
    print(df)