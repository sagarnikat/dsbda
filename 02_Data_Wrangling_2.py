# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Create Dataset
data = {
    'Roll':[1,2,3,4,5,6,7,8,9,10,
            11,12,13,14,15,16,17,18,19,20],

    'Name':['Amit','Sneha','Raj','Pooja','Karan',
            'Neha','Ravi','Tina','Arjun','Meena',
            'Om','Riya','Mohit','Asha','Vikas',
            'Priya','Rohit','Kavya','Suraj','Anjali'],

    'Maths':[78,88,45,67,120,76,34,91,72,85,
             65,55,98,74,69,82,25,77,90,73],

    'Science':[80,92,50,70,98,85,40,95,68,89,
               60,48,96,78,72,84,30,79,92,75],

    'English':[75,85,40,65,90,np.nan,38,94,70,87,
               58,50,94,80,66,86,28,81,88,70],

    'Attendance':[90,95,60,80,96,88,55,97,np.nan,93,
                  72,68,99,85,79,91,45,87,95,82],

    'Result':['Pass','Pass','Fail','Pass','Pass',
              'Pass','Fail','Pass','Pass','Pass',
              'Pass','Fail','Pass','Pass','Pass',
              'Pass','Fail','Pass','Pass','Pass']
}

# Step 3: Convert into DataFrame
df = pd.DataFrame(data)

# Step 4: Display Dataset
print("Dataset:\n")
print(df)

# Step 5: Check Missing Values
print("\nMissing Values:\n")
print(df.isnull().sum())

# Step 6: Fill Missing Values using Mean
df['English'] = df['English'].fillna(df['English'].mean())

df['Attendance'] = df['Attendance'].fillna(df['Attendance'].mean())

# Step 7: Check Missing Values Again
print("\nMissing Values After Filling:\n")
print(df.isnull().sum())

# Step 8: Statistical Summary
print("\nStatistical Summary:\n")
print(df.describe())

# Step 9: Detect Outliers using Boxplot
plt.boxplot(df['Maths'])

plt.title("Boxplot of Maths Marks")

plt.ylabel("Marks")

plt.show()

# Step 10: Calculate Q1 and Q3
Q1 = df['Maths'].quantile(0.25)

Q3 = df['Maths'].quantile(0.75)

IQR = Q3 - Q1

print("\nQ1 =", Q1)
print("Q3 =", Q3)
print("IQR =", IQR)

# Step 11: Calculate Lower and Upper Limits
lower_limit = Q1 - 1.5 * IQR

upper_limit = Q3 + 1.5 * IQR

print("\nLower Limit =", lower_limit)
print("Upper Limit =", upper_limit)

# Step 12: Remove Outliers
df = df[(df['Maths'] >= lower_limit) &
        (df['Maths'] <= upper_limit)]

# Step 13: Dataset After Removing Outliers
print("\nDataset After Removing Outliers:\n")
print(df)

# Step 14: Apply Log Transformation
df['Attendance_Log'] = np.log(df['Attendance'])

# Step 15: Display Transformed Dataset
print("\nDataset After Log Transformation:\n")
print(df.head())

# Step 16: Check Skewness
print("\nSkewness:\n")
print(df.skew(numeric_only=True))

# Step 17: Histogram of Attendance
plt.hist(df['Attendance'], bins=10)

plt.title("Attendance Distribution")

plt.xlabel("Attendance")

plt.ylabel("Frequency")

plt.show()

# Step 18: Histogram of Log Transformed Attendance
plt.hist(df['Attendance_Log'], bins=10)

plt.title("Log Transformed Attendance")

plt.xlabel("Attendance Log")

plt.ylabel("Frequency")

plt.show()

# Step 19: Final Shape
print("\nFinal Shape:")
print(df.shape)

# Step 20: Final Dataset
print("\nFinal Dataset:\n")
print(df.head())