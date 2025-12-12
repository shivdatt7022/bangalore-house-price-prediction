import pandas as pd

df = pd.read_csv("../data/Bengaluru_House_Data.csv")

# 1) Keep useful columns
df = df[["location", "size", "total_sqft", "bath", "price"]].dropna()

# 2) Keep only rows where total_sqft is numeric
def is_float(x):
    try:
        float(x)
        return True
    except:
        return False

df = df[df["total_sqft"].apply(is_float)].copy()
df["total_sqft"] = df["total_sqft"].astype(float)

# 3) Create bhk from size, like "2 BHK"
df["bhk"] = df["size"].str.extract(r"(\d+)").astype(float)

print(df.head())
print(df.describe())
print(df.shape)
