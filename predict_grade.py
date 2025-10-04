import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import norm

data = pd.DataFrame({
    'GPA': [15.2, 14.8, 16.5, 13.4, 17.2],
    'Prereq_Avg': [14, 13.5, 17, 12, 18],
    'Attendance': [12, 10, 15, 8, 16],
    'Quiz1': [18, 15, 19, 12, 20],
    'FinalGrade': [17, 15, 18.5, 13, 19]
})

features = ['GPA', 'Prereq_Avg', 'Attendance', 'Quiz1']
X = data[features]
y = data['FinalGrade']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
std_error = np.sqrt(mse)

new_student = pd.DataFrame({
    'GPA': [15.5],
    'Prereq_Avg': [14.2],
    'Attendance': [13],
    'Quiz1': [17]
})

y_new_pred = model.predict(new_student)[0]
z = norm.ppf(0.975)
lower = y_new_pred - z * std_error
upper = y_new_pred + z * std_error
d = 1.0
confidence_pct = 2 * norm.cdf(d / std_error) - 1

print({
    "Predicted Final Grade": round(y_new_pred, 2),
    "Confidence Interval (95%)": (round(lower, 2), round(upper, 2)),
    "Std. Error": round(std_error, 2),
    "Confidence ±1 Margin (%)": round(confidence_pct * 100, 2)
})
