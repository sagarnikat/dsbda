# Step 1: Import Libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Step 2: Create Same Titanic Dataset
data = {

    'PassengerId':[1,2,3,4,5,
                   6,7,8,9,10,
                   11,12,13,14,15,
                   16,17,18,19,20,
                   21,22,23,24,25],

    'Survived':[0,1,1,1,0,
                0,0,1,1,1,
                1,1,0,0,1,
                0,1,0,1,0,
                1,0,1,0,1],

    'Pclass':[3,1,3,1,3,
              3,1,3,3,2,
              3,1,3,2,3,
              2,1,3,2,3,
              1,3,2,3,1],

    'Sex':['male','female','female','female','male',
           'male','male','male','female','female',
           'female','female','male','male','female',
           'male','female','male','female','male',
           'male','male','female','male','female'],

    'Age':[22,38,26,35,35,
           30,54,2,27,14,
           4,58,20,39,14,
           31,25,20,29,16,
           34,18,28,8,45],

    'Fare':[7.25,71.28,7.92,53.10,8.05,
            8.45,51.86,21.07,11.13,30.07,
            16.70,26.55,8.05,31.27,7.85,
            10.50,13.00,7.22,26.00,46.90,
            13.00,8.05,35.50,21.07,90.00],
}
temp_df = pd.DataFrame(data)

# Save as CSV file
temp_df.to_csv("data.csv", index=False)

print("data.csv file created successfully!")

# STEP 3: LOAD DATA FROM CSV FILE


df = pd.read_csv("data.csv")

# Step 4: Display Dataset
print("Titanic Dataset:\n")
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

# Step 9: Boxplot of Age with Gender and Survival
sns.boxplot(
    x='Sex',
    y='Age',
    hue='Survived',
    data=df
)

# Step 10: Graph Labels
plt.title("Age Distribution by Gender and Survival")

plt.xlabel("Gender")

plt.ylabel("Age")

plt.show()

# Step 11: Violin Plot
sns.violinplot(
    x='Sex',
    y='Age',
    hue='Survived',
    data=df,
    split=True
)

plt.title("Violin Plot of Age Distribution")

plt.show()

# Step 12: Strip Plot
sns.stripplot(
    x='Sex',
    y='Age',
    hue='Survived',
    data=df,
    dodge=True
)

plt.title("Strip Plot of Age Distribution")

plt.show()

# Step 13: Count Plot
sns.countplot(
    x='Sex',
    hue='Survived',
    data=df
)

plt.title("Gender Wise Survival Count")

plt.show()

# Step 14: Average Age by Gender
print("\nAverage Age by Gender:\n")

print(df.groupby('Sex')['Age'].mean())

# Step 15: Average Age by Survival
print("\nAverage Age by Survival:\n")

print(df.groupby('Survived')['Age'].mean())

# Step 16: Final Observations
print("\nObservations:")
print("1. Females survived more than males.")
print("2. Children had higher survival chances.")
print("3. Male passengers had wider age distribution.")
print("4. Many elderly males did not survive.")
print("5. Female passengers had better survival ratio.")