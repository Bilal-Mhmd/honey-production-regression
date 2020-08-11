import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv(
    "https://s3.amazonaws.com/codecademy-content/programs/data-science-path/linear_regression/honeyproduction.csv")

# print(df.head())
prod_per_year = df.groupby('year').totalprod.mean().reset_index()
# print(prod_per_year)
# print(prod_per_year["year"])


# Create a variable called X that is the column of years in this prod_per_year DataFrame.
X = prod_per_year["year"]
X = X.values.reshape(-1, 1)  # reshape X to get it into the right format
# print(X)

y = prod_per_year["totalprod"]

plt.scatter(X, y)
# plt.show()

# Create and Fit a Linear Regression Model
regr = linear_model.LinearRegression()  # Create a linear regression model
regr = regr.fit(X, y)  # Fit the model to the data

print("the slope of the model:", regr.coef_)
print("the intercept of the model:", regr.intercept_)

# the predictions that the model regr would make on X data
y_predict = regr.predict(X)

plt.plot(X, y_predict)
#plt.show()


# Predict the Honey DECLINE!

# NumPy array called X_future that is the range from 2013 to 2050:
X_future = np.array(range(2013, 2050))

#reshape
X_future = X_future.reshape(-1, 1)

# new list of y_values to the new data
future_predict = regr.predict(X_future)

plt.plot(X_future, future_predict)


plt.title("Honey Production Decline")
plt.xlabel("year")
plt.ylabel("production per year")
plt.show()
