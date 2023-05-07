from IPython.display import display
import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler

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
    

def scale_data(data, columns):
    scaled_cols = list(columns)
    scaler_transformer = make_column_transformer((StandardScaler(), 
                                                  scaled_cols), 
                                                 remainder="passthrough")

    scaled = scaler_transformer.fit_transform(data[scaled_cols])
    scaled_df = pd.DataFrame(scaled, index=data.index, columns=scaled_cols)
    return scaled_df