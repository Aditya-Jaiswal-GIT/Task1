
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Titanic-Dataset.csv')
print("Dataset loaded successfully!\n")


print("Basic Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nSummary Statistics:")
print(df.describe())


df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin due to many nulls
df.drop(columns='Cabin', inplace=True)

# Drop Ticket and Name as they're not useful for ML
df.drop(columns=['Ticket', 'Name'], inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# Convert 'Sex' to 0/1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert 'Embarked' to numerical values
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})


df['Age'] = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())
df['Fare'] = (df['Fare'] - df['Fare'].min()) / (df['Fare'].max() - df['Fare'].min())


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.boxplot(df['Age'])
plt.title("Boxplot of Age")

plt.subplot(1, 2, 2)
plt.boxplot(df['Fare'])
plt.title("Boxplot of Fare")

plt.tight_layout()
plt.show()

# Remove Outliers using IQR (on Fare)
Q2 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q2

before = len(df)
df = df[~((df['Fare'] < (Q2 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR)))]
after = len(df)

print(f"\nOutliers removed: {before - after}")

#  Final Preview
print("\nCleaned Data Preview:")
print(df.head())

print(f"\nFinal Shape: {df.shape}")

#  Save Cleaned Data
df.to_csv('titanic_cleaned.csv', index=False)
print("\nCleaned dataset saved as 'titanic_cleaned.csv'")