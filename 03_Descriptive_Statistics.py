# Step 1: Import Libraries
import pandas as pd
import seaborn as sns
import numpy as np

# Step 2: Create Dataset
data = {

    'Name':['A','B','C','D','E',
            'F','G','H','I','J',
            'K','L','M','N','O',
            'P','Q','R','S','T'],

    'Age_Group':['Young','Young','Adult','Adult','Senior',
                 'Senior','Young','Adult','Senior','Young',
                 'Adult','Senior','Young','Adult','Senior',
                 'Young','Adult','Senior','Young','Adult'],

    'Income':[25000,27000,45000,47000,52000,
              54000,29000,49000,58000,30000,
              46000,56000,31000,48000,60000,
              32000,50000,62000,34000,51000]
}

# Step 3: Convert into DataFrame
df = pd.DataFrame(data)

# Step 4: Display Dataset
print("Dataset:\n")
print(df)

# Step 5: Check Shape
print("\nShape of Dataset:")
print(df.shape)

# Step 6: Display First 5 Rows
print("\nFirst 5 Rows:\n")
print(df.head())

# Step 7: Group Dataset by Age_Group
group = df.groupby('Age_Group')['Income']

# Step 8: Mean
print("\nMean:\n")
print(group.mean())

# Step 9: Median
print("\nMedian:\n")
print(group.median())

# Step 10: Minimum
print("\nMinimum:\n")
print(group.min())

# Step 11: Maximum
print("\nMaximum:\n")
print(group.max())

# Step 12: Standard Deviation
print("\nStandard Deviation:\n")
print(group.std())

# Step 13: Variance
print("\nVariance:\n")
print(group.var())

# Step 14: Complete Statistical Summary
print("\nComplete Summary:\n")
print(group.describe())

# Step 15: Create List for Each Category
young = df[df['Age_Group'] == 'Young']['Income'].tolist()

adult = df[df['Age_Group'] == 'Adult']['Income'].tolist()

senior = df[df['Age_Group'] == 'Senior']['Income'].tolist()

print("\nYoung Income List:")
print(young)

print("\nAdult Income List:")
print(adult)

print("\nSenior Income List:")
print(senior)

# Step 16: Load Iris Dataset
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
iris = pd.DataFrame(data)

# Step 17: Display Iris Dataset
print("\nIris Dataset:\n")
print(iris.head())

# Step 18: Iris Dataset Shape
print("\nIris Shape:")
print(iris.shape)

# Step 19: Statistical Details Species Wise
print("\nSpecies Wise Statistics:\n")
print(iris.groupby('species').describe())

# Step 20: Percentiles
print("\nPercentiles:\n")

print(iris.groupby('species').quantile([0.25,0.50,0.75]))