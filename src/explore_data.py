import pandas as pd

file_path = "../data/Bengaluru_House_Data.csv"  # path from src/ to data/
df = pd.read_csv(file_path)

print(df.head())
print(df.info())
