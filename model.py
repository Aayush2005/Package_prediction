import pandas as pd
import numpy as np

df = pd.read_csv('Travel_Cleaned.csv')

print(df.isnull().sum())