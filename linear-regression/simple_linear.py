import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Simple Dataset: Hours studied vs Marks
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([40, 50, 65, 75, 90])

# Train the model
model = LinearRegression()
model.fit(X, y)

# Plot the dataset and regression line
plt.scatter(X, y, label="Actual Data")
plt.plot(X, model.predict(X), label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Study Hours vs Marks Prediction")
plt.legend()
plt.show()

# Predict for user input (same as your original logic)
hours = float(input("Enter how many hours you studied: "))
predicted_marks = model.predict([[hours]])

print(f"Based on your Hours {hours} you may score around {predicted_marks[0]:.2f}")
