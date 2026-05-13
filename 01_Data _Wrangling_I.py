# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Step 2: Create Dataset
data = {
    'Emp_ID':[1,2,3,4,5,6,7,8,9,10,
              11,12,13,14,15,16,17,18,19,20],

    'Name':['Amit','Sneha','Raj','Pooja','Karan',
            'Neha','Ravi','Tina','Arjun','Meena',
            'Om','Riya','Mohit','Asha','Vikas',
            'Priya','Rohit','Kavya','Suraj','Anjali'],

    'Age':[23,25,28,24,30,29,31,26,27,32,
           24,23,33,28,29,27,35,26,30,31],

    'Gender':['Male','Female','Male','Female','Male',
              'Female','Male','Female','Male','Female',
              'Male','Female','Male','Female','Male',
              'Female','Male','Female','Male','Female'],

    'Department':['IT','HR','Sales','IT','Finance',
                  'HR','IT','Sales','Finance','IT',
                  'HR','Sales','Finance','IT','Sales',
                  'HR','Finance','IT','Sales','HR'],

    'Salary':[35000,42000,50000,39000,60000,
              52000,65000,47000,55000,70000,
              41000,36000,72000,58000,53000,
              49000,80000,46000,61000,64000],

    'Experience':[1,2,4,2,5,4,6,3,4,7,
                  2,1,8,5,4,3,10,3,5,6]
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
print("\nShape of Dataset:")
print(df.shape)

# Step 7: Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 8: Dataset Information
print("\nDataset Info:")
print(df.info())

# Step 9: Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Step 10: Datatypes
print("\nDatatypes:")
print(df.dtypes)

# Step 11: Label Encoding
df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})

# Step 12: One Hot Encoding
df = pd.get_dummies(df, columns=['Department'])

# Step 13: Normalization
scaler = MinMaxScaler()

df['Salary'] = scaler.fit_transform(df[['Salary']])

# Step 14: Final Dataset
print("\nFinal Processed Dataset:\n")
print(df.head())

# Step 15: Final Shape
print("\nFinal Shape:")
print(df.shape)