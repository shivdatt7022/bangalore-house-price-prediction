import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load same prepared data steps
df = pd.read_csv("../data/Bengaluru_House_Data.csv")
df = df[["location", "size", "total_sqft", "bath", "price"]].dropna()

def is_float(x):
    try:
        float(x)
        return True
    except:
        return False

df = df[df["total_sqft"].apply(is_float)].copy()
df["total_sqft"] = df["total_sqft"].astype(float)
df["bhk"] = df["size"].str.extract(r"(\d+)").astype(float)

# features (X) and target (y)
X = df[["total_sqft", "bath", "bhk"]]
y = df["price"]

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# create and train model
model = LinearRegression()
model.fit(X_train, y_train)

print("Train R^2:", model.score(X_train, y_train))
print("Test R^2:", model.score(X_test, y_test))

def predict_price(total_sqft, bath, bhk):
    data = [[total_sqft, bath, bhk]]
    data = pd.DataFrame(
        [[total_sqft, bath, bhk]],
        columns = ["total_sqft", "bath", "bhk"]
    )
    return model.predict(data)[0]
example_price = predict_price(1200,2,2)
print("Predicted price for 2 BHK, 1200 sqft, 2 bath:", example_price)