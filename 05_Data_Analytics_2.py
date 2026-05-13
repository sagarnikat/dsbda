# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Step 2: Create Dataset
data = {

    'Age':[19,22,25,28,32,
           35,40,42,24,27,
           45,50,21,29,31,
           26,37,39,23,41],

    'Sex':['Male','Female','Male','Female','Male',
           'Female','Male','Female','Male','Female',
           'Male','Female','Male','Female','Male',
           'Female','Male','Female','Male','Female'],

    'Salary':[19000,25000,32000,45000,52000,
              58000,65000,70000,30000,35000,
              85000,90000,22000,48000,51000,
              33000,60000,62000,28000,72000],

    'Purchased':[0,0,0,1,1,
                 1,1,1,0,0,
                 1,1,0,1,1,
                 0,1,1,0,1]
}
# Convert dictionary to DataFrame
temp_df = pd.DataFrame(data)

# Save as CSV file
temp_df.to_csv("data.csv", index=False)

print("data.csv file created successfully!")

# STEP 3: LOAD DATA FROM CSV FILE


df = pd.read_csv("data.csv")

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

# Step 8: Statistical Summary
print("\nStatistical Summary:\n")
print(df.describe())

# Step 9: Convert Sex into Numeric
df['Sex'] = df['Sex'].map({'Male':1, 'Female':0})

print("\nDataset After Encoding:\n")
print(df.head())

# Step 9: Define X and Y
X = df[['Age','Sex','Salary']]

y = df['Purchased']

# Step 10: Normalize Features
scaler = MinMaxScaler()

X = scaler.fit_transform(X)

# Step 11: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Step 12: Create Logistic Regression Model
model = LogisticRegression()

# Step 13: Train Model
model.fit(X_train, y_train)

# Step 14: Prediction
predictions = model.predict(X_test)

# Step 15: Display Predictions
print("\nPredictions:\n")
print(predictions)

# Step 16: Confusion Matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:\n")
print(cm)

# Step 17: Heatmap of Confusion Matrix
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# Step 18: Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:")
print(accuracy)

# Step 19: Precision
precision = precision_score(y_test, predictions)

print("\nPrecision:")
print(precision)

# Step 20: Recall
recall = recall_score(y_test, predictions)

print("\nRecall:")
print(recall)

# Step 21: Calculate TP, TN, FP, FN
TN = cm[0][0]

FP = cm[0][1]

FN = cm[1][0]

TP = cm[1][1]

print("\nTrue Positive:", TP)

print("True Negative:", TN)

print("False Positive:", FP)

print("False Negative:", FN)

# Step 22: Error Rate
error_rate = 1 - accuracy

print("\nError Rate:")
print(error_rate)

# Step 23: Scatter Plot with Legend

colors = ['blue' if x == 0 else 'yellow'
          for x in df['Purchased']]

plt.scatter(df['Age'],
            df['Salary'],
            c=colors)

# Labels
plt.xlabel("Age")

plt.ylabel("Salary")

plt.title("Age vs Salary")

# Custom Legend
import matplotlib.patches as mpatches

blue_patch = mpatches.Patch(color='blue',
                            label='Not Purchased')

yellow_patch = mpatches.Patch(color='yellow',
                              label='Purchased')

plt.legend(handles=[blue_patch, yellow_patch])

plt.show()