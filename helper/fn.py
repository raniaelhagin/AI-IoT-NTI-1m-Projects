from IPython.display import display
import pandas as pd

def read_data(src):
    
    data = pd.read_csv(src)
    return data

def explore_data(data):
    
    print(f"Shape of the data: {data.shape}\n")
    
    print("Data information:")
    display(data.info())
    
    print(f"\nNumber of null values in each column:\n{data.isnull().sum()}\n")
    
    print("Data Samples:")
    display(data.head())
    
    print("Data descriptive statistics:")
    display(data.describe())