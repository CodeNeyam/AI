import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Step 1: Create your dataset
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7],
    'Sleep_Hours': [6, 6.5, 7, 5, 8, 7.5, 6],
    'Passed_Exam': [0, 0, 0, 0, 1, 1, 1]  # 0 = No, 1 = Yes
}

df = pd.DataFrame(data)

# Step 2: View the dataset
print("Full dataset:")
print(df)

# Step 3: Separate features (X) and label (y)
X = df[['Hours_Studied', 'Sleep_Hours']]
y = df['Passed_Exam']

print("\nFeatures (X):")
print(X)

print("\nLabel (y):")
print(y)

# Step 4: Train a simple model
model = DecisionTreeClassifier()
model.fit(X, y)

# Step 5: Make a prediction
new_data = pd.DataFrame({
    'Hours_Studied': [5, 4],
    'Sleep_Hours': [8, 6]
})

predictions = model.predict(new_data)

print("\nPrediction on new data:")
print(predictions)  # Expected: [0 1]