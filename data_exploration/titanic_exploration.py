import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
df = pd.read_csv('titanic.csv')  # Fixed filename

# 1️⃣ Inspect the dataset
print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

# 2️⃣ Visualization: Survival count by gender
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title("Survival Count by Gender")
plt.show()

# 3️⃣ Visualization: Age distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'].dropna(), bins=20)
plt.title("Age Distribution")
plt.show()