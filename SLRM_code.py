import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.metrics import mean_squared_error
import pickle

dataset = pd.read_csv(r"C:\Users\hii\python\simple linear eq\Salary_Data.csv")

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20, train_size=0.80, random_state=0)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_test, y_test, color='red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title('Salary vs Exp (Test set)')
plt.xlabel('Year of Exp')
plt.ylabel('Salary')
plt.show()

m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

y_20 = m_slope*20 + c_intercept
print(y_20)


bias = regressor.score(x_train, y_train)
print(bias)
variance = regressor.score(x_test, y_test)
print(variance)
train_mse = mean_squared_error(y_train, regressor.predict(x_train))
test_mse = mean_squared_error(y_test, y_pred)


print(f"Training Score (R^2):{bias:.2f}")
print(f"Testing Score (R^2): {variance: .2f}")
print(f"Training MSE: {train_mse:.2f} ")
print(f"Test MSE: {test_mse: .2f}")

#ANOVA
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

y=y[0:6]
SSE = np.sum((y- y_pred)**2)
print(SSE)

mean_total = np.mean(dataset.values)
#here df.to_numpy() will convert pandas df to numpy
SST = np.sum((dataset.values- mean_total)**2)
print(SST)

r_square = 1 - (SSR / SST)
r_square

print(r_square)
print(bias)
print(variance)


import pickle

filename = 'linear_regression_model.pkl'
with open (filename,'wb') as file:
    pickle.dump(regressor, file)
    
print("Model has been pickled and saves as linear_regression_model.pkl")


import os 
os.getcwd()

