# Step 1: Import Libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Step 2: Create Iris Dataset
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
}
temp_df = pd.DataFrame(data)

# Save as CSV file
temp_df.to_csv("data.csv", index=False)

print("data.csv file created successfully!")

# STEP 3: LOAD DATA FROM CSV FILE


iris = pd.read_csv("data.csv")

# Step 4: Display Dataset
print("Iris Dataset:\n")
print(iris)

# Step 5: Display First 5 Rows
print("\nFirst 5 Rows:\n")
print(iris.head())

# Step 6: Check Shape
print("\nShape:")
print(iris.shape)

# Step 7: Check Missing Values
print("\nMissing Values:\n")
print(iris.isnull().sum())

# Step 8: Check Datatypes
print("\nDatatypes:\n")
print(iris.dtypes)

# Step 9: Display Feature Names
print("\nFeature Names:\n")
print(iris.columns)

# Step 10: Statistical Summary
print("\nStatistical Summary:\n")
print(iris.describe())

# Step 11: Histograms for All Features
iris.hist(figsize=(10,8))

plt.suptitle("Histogram of Iris Features")

plt.show()

# Step 12: Boxplot for All Features
plt.figure(figsize=(10,6))

sns.boxplot(data=iris.select_dtypes(include='number'))

plt.title("Boxplot of Iris Features")

plt.show()

# Step 13: Pairplot
sns.pairplot(
    iris,
    hue='species'
)

plt.show()

# Step 14: Sepal Length Histogram
plt.hist(
    iris['sepal_length'],
    bins=10,
    color='orange'
)

plt.title("Sepal Length Distribution")

plt.xlabel("Sepal Length")

plt.ylabel("Frequency")

plt.show()

# Step 15: Petal Length Boxplot
sns.boxplot(
    x='species',
    y='petal_length',
    data=iris
)

plt.title("Petal Length by Species")

plt.show()

# Step 16: Violin Plot
sns.violinplot(
    x='species',
    y='petal_width',
    data=iris
)

plt.title("Petal Width Distribution")

plt.show()

# Step 17: Correlation Matrix
corr = iris.corr(numeric_only=True)

print("\nCorrelation Matrix:\n")
print(corr)

# Step 18: Heatmap
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

# Step 19: Detect Outliers using Boxplot
sns.boxplot(
    y=iris['sepal_width']
)

plt.title("Outlier Detection - Sepal Width")

plt.show()

# Step 20: Final Inference
print("\nInference:")
print("1. Iris dataset contains numeric and categorical features.")
print("2. Petal length and petal width are highly correlated.")
print("3. Some outliers are present in sepal width.")
print("4. Setosa species is clearly separable from others.")
print("5. Versicolor and Virginica have overlapping distributions.")