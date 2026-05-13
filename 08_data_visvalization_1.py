# Step 1: Import Libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Step 2: Create Dataset
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

# Step 3: Load Dataset into DataFrame
df = pd.DataFrame(data)

# Step 4: Display Dataset
print("Titanic Dataset:\n")
print(df)

# Step 5: First 5 Rows
print("\nFirst 5 Rows:\n")
print(df.head())

# Step 6: Shape
print("\nShape:")
print(df.shape)

# Step 7: Missing Values
print("\nMissing Values:\n")
print(df.isnull().sum())

# Step 8: Statistical Summary
print("\nStatistical Summary:\n")
print(df.describe())

# Step 9: Survival Count
sns.countplot(x='Survived', data=df)

plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Count")

plt.show()

# Step 10: Gender Wise Survival
sns.countplot(x='Sex',
              hue='Survived',
              data=df)

plt.title("Gender Wise Survival")
plt.xlabel("Gender")
plt.ylabel("Count")

plt.show()

# Step 11: Passenger Class Survival
sns.countplot(x='Pclass',
              hue='Survived',
              data=df)

plt.title("Passenger Class Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Count")

plt.show()

# Step 12: Histogram of Fare
plt.hist(df['Fare'].dropna(),
         bins=20,
         color='orange')

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")

plt.show()

# Step 13: KDE Plot
sns.kdeplot(df['Fare'].dropna(),
            fill=True)

plt.title("Fare Density Plot")
plt.xlabel("Fare")

plt.show()

# Step 14: Boxplot of Fare
sns.boxplot(x=df['Fare'])

plt.title("Fare Boxplot")

plt.show()

# Step 15: Correlation Matrix
corr = df.corr(numeric_only=True)

print("\nCorrelation Matrix:\n")
print(corr)

# Step 16: Heatmap
sns.heatmap(corr,
            annot=True,
            cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.show()

# Step 17: Final Observations
print("\nObservations:")
print("1. Females survived more than males.")
print("2. First class passengers had higher survival rate.")
print("3. Fare distribution is right skewed.")
print("4. Some outliers are present in Fare column.")