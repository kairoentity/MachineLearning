import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression

data = pd.read_csv('test_scores.csv')
x = data['math'].values
y = data['cs'].values

m,b = 0.0,0.0
learning_rate =  0.0001
iterations = 1000000
n = float(len(x))

def cost_function(x,y,m,b):
    return np.sum((y - (m * x + b))**2) / n

previous_cost = float('inf')

for i in range(iterations):
    y_pred = m * x + b
    md = -(2/n) * np.sum(x * (y-y_pred))
    bd = -(2/n) * np.sum(y-y_pred)
    m = m - learning_rate * md
    b = b - learning_rate * bd

    current_cost = cost_function(x,y,m,b)
    print(f"Iteration {i + 1}:  m = {m}, b = {b},Cost = {current_cost}")

    if math.isclose(current_cost,previous_cost, abs_tol=1e-9):
        print(f"Converged after {i} iterations")
        break

    previous_cost = current_cost
if i == iterations -1:
    print(f"Did not converge.Completed {i+1} iterations")

print(f"Gradient Descent results: m = {m}, b = {b} , cost = {current_cost} ,iterations = {i}")

x = x.reshape(-1, 1)
model = LinearRegression()
model.fit(x, y)

m_sklearn = model.coef_[0]
b_sklearn = model.intercept_

print(f"sklearn results: m = {m_sklearn}, b = {b_sklearn}")