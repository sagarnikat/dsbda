# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Step 2: Create Dataset
data = {

    'Emp_ID':[1,2,3,4,5,6,7,8,9,10,
              11,12,13,14,15,16,17,18,19,20],

    'Name':['Yash','Komal','Aditya','Sonal','Nikhil',
            'Pallavi','Tejas','Mrunal','Harsh','Vaishnavi',
            'Sagar','Isha','Atharva','Ketaki','Manav',
            'Sakshi','Pratik','Tanvi','Ruturaj','Bhakti'],

    'Age':[22,24,29,np.nan,31,27,34,25,26,33,
           23,28,36,30,29,24,38,27,32,35],

    'Gender':['Male','Female','Male','Female','Male',
              'Female','Male','Female','Male','Female',
              'Male','Female','Male','Female','Male',
              'Female','Male','Female','Male','Female'],

    'Department':['Developer','HR','Marketing','Developer','Finance',
                  'HR','Developer',np.nan,'Finance','Developer',
                  'HR','Marketing','Finance','Developer','Marketing',
                  'HR','Finance','Developer','Marketing','HR'],

    'Salary':[38000,45000,52000,41000,67000,
              300000,72000,np.nan,59000,76000,
              43000,39000,81000,61000,56000,
              50000,95000,48000,69000,73000],

    'Experience':[1,2,5,2,6,
                  18,7,3,4,8,
                  2,1,10,5,4,
                  3,22,3,6,7]
}

# Convert dictionary to DataFrame
temp_df = pd.DataFrame(data)

# Add Duplicate Record
temp_df.loc[20] = temp_df.loc[4]

# Save as CSV file
temp_df.to_csv("employee_data.csv", index=False)

print("employee_data.csv file created successfully!")

# STEP 3: LOAD DATA FROM CSV FILE

df = pd.read_csv("employee_data.csv")

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

# ---------------------------------------------------
# HANDLING MISSING VALUES
# ---------------------------------------------------

df['Age'] = df['Age'].fillna(df['Age'].mean())

df['Salary'] = df['Salary'].fillna(df['Salary'].mean())

df['Department'] = df['Department'].fillna(df['Department'].mode()[0])

print("\nMissing Values After Handling:")
print(df.isnull().sum())

# ---------------------------------------------------
# CHECK DUPLICATE RECORDS
# ---------------------------------------------------

print("\nDuplicate Records:")
print(df.duplicated().sum())

# Remove duplicate records
df = df.drop_duplicates()

print("\nShape After Removing Duplicates:")
print(df.shape)

# ---------------------------------------------------
# HANDLING OUTLIERS USING IQR METHOD
# ---------------------------------------------------

Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

median_salary = df['Salary'].median()

# Replace outliers with median
df['Salary'] = np.where(df['Salary'] > upper_limit,
                        median_salary,
                        df['Salary'])

# ---------------------------------------------------
# Step 8: Dataset Information
# ---------------------------------------------------

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