# Step 1: Import Libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Step 2: Load Iris Dataset
data = {

    'sepal_length':[5.1,4.9,4.7,4.6,5.0,
                    5.4,4.6,5.0,5.2,5.5,
                    6.4,6.9,5.5,6.5,5.7,
                    6.3,4.9,6.6,5.2,5.0,
                    6.3,5.8,7.1,6.3,6.5],

    'sepal_width':[3.5,3.0,3.2,3.1,3.6,
                   3.9,3.4,3.4,3.5,4.2,
                   3.2,3.1,2.3,2.8,2.8,
                   3.3,2.4,2.9,2.7,2.0,
                   3.3,2.7,3.0,2.9,3.0],

    'petal_length':[1.4,1.4,1.3,1.5,1.4,
                    1.7,1.4,1.5,1.5,1.4,
                    4.5,4.9,4.0,4.6,4.5,
                    4.7,3.3,4.6,3.9,3.5,
                    6.0,5.1,5.9,5.6,5.8],

    'petal_width':[0.2,0.2,0.2,0.2,0.2,
                   0.4,0.3,0.2,0.2,0.2,
                   1.5,1.5,1.3,1.5,1.3,
                   1.6,1.0,1.3,1.4,1.0,
                   2.5,1.9,2.1,1.8,2.2],

    'species':['setosa','setosa','setosa','setosa','setosa',
               'setosa','setosa','setosa','setosa','setosa',
               'versicolor','versicolor','versicolor','versicolor','versicolor',
               'versicolor','versicolor','versicolor','versicolor','versicolor',
               'virginica','virginica','virginica','virginica','virginica']
}# Convert dictionary to DataFrame
temp_df = pd.DataFrame(data)

# Save as CSV file
temp_df.to_csv("data.csv", index=False)

print("data.csv file created successfully!")

# STEP 3: LOAD DATA FROM CSV FILE


iris = pd.read_csv("data.csv")
# Step 3: Display Dataset
print("Iris Dataset:\n")
print(iris)

# Step 4: Display First 5 Rows
print("\nFirst 5 Rows:\n")
print(iris.head())

# Step 5: Check Shape
print("\nShape:")
print(iris.shape)

# Step 6: Check Missing Values
print("\nMissing Values:\n")
print(iris.isnull().sum())

# Step 7: Statistical Summary
print("\nStatistical Summary:\n")
print(iris.describe())

# Step 8: Display Species Count
print("\nSpecies Count:\n")
print(iris['species'].value_counts())

# Step 9: Define X and Y
X = iris.drop('species', axis=1)

y = iris['species']

# Step 10: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Step 11: Create Naive Bayes Model
model = GaussianNB()

# Step 12: Train Model
model.fit(X_train, y_train)

# Step 13: Prediction
predictions = model.predict(X_test)

# Step 14: Display Predictions
print("\nPredictions:\n")
print(predictions)

# Step 15: Confusion Matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:\n")
print(cm)

# Step 16: Heatmap of Confusion Matrix
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# Step 17: Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:")
print(accuracy)

# Step 18: Precision
precision = precision_score(
    y_test,
    predictions,
    average='macro'
)

print("\nPrecision:")
print(precision)

# Step 19: Recall
recall = recall_score(
    y_test,
    predictions,
    average='macro'
)

print("\nRecall:")
print(recall)

# Step 20: Pairplot Visualization
sns.pairplot(iris,
             hue='species')

plt.show()

# Step 21: Actual vs Predicted
result = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': predictions
})

print("\nActual vs Predicted:\n")
print(result.head(10))

# Step 22: Error Rate
error_rate = 1 - accuracy

print("\nError Rate:")
print(error_rate)

# Step 23: Species Wise Mean
print("\nSpecies Wise Mean:\n")
print(iris.groupby('species').mean())

# Step 24: Final Observation
print("\nNaive Bayes successfully classified iris flowers.")