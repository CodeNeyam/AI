import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create your dataset
data = { 
    'Years_of_Experience': [1, 2, 3, 4, 5, 6, 7],
    'Salary': [30000, 35000, 40000, 45000, 50000, 55000, 60000],
}

df = pd.DataFrame(data)

# Split the dataset into features (X) and label (y)
X = df[['Years_of_Experience']]
y = df['Salary']

# Train a simple model
model = LinearRegression()
model.fit(X, y)

# Predict salary for 10 years of experience
new_experience = [[10]]
prediction = model.predict(new_experience)
print("Predicted Salary for 10 Years of Experience:")
print(prediction[0])

# Visualize the data and regression line
plt.scatter(X, y, color='blue', label='Data Points')  # Plot data points
plt.plot(X, model.predict(X), color='red', label='Regression Line')  # Plot regression line
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression: Years of Experience vs Salary')
plt.legend()
plt.grid(True)
plt.show()