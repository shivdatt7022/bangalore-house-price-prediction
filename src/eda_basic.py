import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#1. load data
df = pd.read_csv("../data/Bengaluru_House_Data.csv")

#2. keep main columns
df = df[["location", "size", "total_sqft", "bath", "price"]]

#print(df.head())
#print(df.shape) #(13320, 5)
#print(df.info())
#print(df.describe())

print(df.loc[0,"price"])
print(df.iloc[0,2])
#print(df.isna().sum())

#3. simple histogram of price
plt.figure(figsize=(6,4))
sns.histplot(df["price"], bins = 50, kde = True)
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Average price by number of bathrooms
avg_price_by_bath = df.groupby("bath")["price"].mean().sort_index()
print(avg_price_by_bath)

#Average price by BHK ( from 'size')
df["bhk"] = df["size"].str.extract(r"(\d+)").astype(float)
avg_price_by_bhk = df.groupby("bhk")["price"].mean().sort_index()
print(avg_price_by_bhk)

plt.figure()
avg_price_by_bhk.plot(kind = "bar")
plt.xlabel("BHK")
plt.ylabel("Average price (lakhs)")
plt.title("Average price by BHK")
plt.tight_layout()
plt.savefig("../data/avg_price_by_bhk.png", dpi=300)
plt.show()

# Focus only on reasonably priced properties
# First keep only rows where total_sqrt is a simple number
def is_float(x):
    try:
        float(x)
        return True
    except:
        return False

df_numeric = df[df["total_sqft"].apply(is_float)].copy()
df_numeric["total_sqft"] = df_numeric["total_sqft"].astype(float)
df_filtered = df_numeric[df_numeric["price"] < 300]

print("Original rows:", df.shape[0])
print("Filtered rows:", df_filtered.shape[0])
print(df_filtered[["price", "total_sqft", "bath"]].describe())

plt.figure()
plt.scatter(df_filtered["total_sqft"], df_filtered["price"], alpha = 0.3)
plt.xlabel("Total sqft")
plt.ylabel("Price (lakhs)")
plt.title("Price vs Area")
plt.tight_layout()
plt.savefig("../data/price_vs_area_filtered.png", dpi=300)
plt.show()
