# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Create Dataset
data = {

    'City':['Pune','Mumbai','Pune','Delhi','Mumbai',
            'Pune','Delhi','Mumbai','Pune','Delhi',
            'Mumbai','Pune','Delhi','Mumbai','Pune',
            'Delhi','Mumbai','Pune','Delhi','Mumbai'],

    'Bedrooms':[2,2,3,3,4,
                4,5,5,6,6,
                2,3,3,4,4,
                5,5,6,6,7],

    'Age':[15,20,18,12,10,
           8,25,30,5,7,
           22,17,14,28,35,
           40,32,26,9,6],

    'Price':[3000000,3500000,4500000,5500000,6500000,
             7000000,8000000,8500000,9500000,10000000,
             3200000,4300000,4800000,5800000,6700000,
             7300000,8200000,9000000,9800000,10500000]
}

# Step 3: Convert into DataFrame
df = pd.DataFrame(data)

# Step 4: Display Dataset
print("Dataset:\n")
print(df)

# Step 5: Display First 5 Rows
print("\nFirst 5 Rows:\n")
print(df.head())

# Step 6: Check Shape
print("\nShape:")
print(df.shape)

# Step 7: Check Missing Values
print("\nMissing Values:\n")
print(df.isnull().sum())

# Step 8: Convert City into Numeric
encoder = LabelEncoder()

df['City'] = encoder.fit_transform(df['City'])

print("\nCity After Encoding:\n")
print(df[['City']].head())

# Step 9: Statistical Summary
print("\nStatistical Summary:\n")
print(df.describe())

# Step 10: Define Independent and Dependent Variables
X = df[['City','Bedrooms','Age']]

y = df['Price']

# Step 11: Normalize Features
scaler = MinMaxScaler()

X = scaler.fit_transform(X)

# Step 12: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Step 13: Create Model
model = LinearRegression()

# Step 14: Train Model
model.fit(X_train, y_train)

# Step 15: Prediction
predictions = model.predict(X_test)

# Step 16: Actual vs Predicted
result = pd.DataFrame({
    'Actual_Price': y_test,
    'Predicted_Price': predictions
})

print("\nActual vs Predicted:\n")
print(result)

# Step 17: Mean Squared Error
mse = mean_squared_error(y_test, predictions)

print("\nMean Squared Error:")
print(mse)

# Step 18: Root Mean Squared Error
rmse = np.sqrt(mse)

print("\nRoot Mean Squared Error:")
print(rmse)

# Step 19: R2 Score
r2 = r2_score(y_test, predictions)

print("\nR2 Score:")
print(r2)

# Step 20: Scatter Plot
plt.scatter(y_test, predictions)

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted House Prices")

plt.show()