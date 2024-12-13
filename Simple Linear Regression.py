import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv(r"D:\FSDS Material\Dataset\Salary_Data.csv")
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.20, random_state=0)

x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

# Visualizing the training set results
plt.scatter(x_train, y_train, color = 'red') # Real salary data testing
plt.plot(x_train, regressor.predict(x_train), color = 'blue') # Regression line 
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the training set results
plt.scatter(x_test, y_test, color = 'red') # Real salary data testing
plt.plot(x_train, regressor.predict(x_train), color = 'blue') # Regression line 
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

print(regressor)

m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

y_12 = m_slope * 12 + c_intercept
print(y_12)

# Statistic Concept 
# Mean

dataset.mean()

dataset['Salary'].mean()

# Median

dataset.median()

dataset['Salary'].median()

# Mode 

dataset['Salary'].mode()

# Variance

dataset.var()

dataset['Salary'].var()

# Standard deviation 

dataset.std()

dataset['Salary'].std()

# Coefficient of variation

from scipy.stats import variation
variation(dataset.values)

variation(dataset['Salary']) 

# Correlation

dataset.corr()

dataset['Salary'].corr(dataset['YearsExperience']) 

# Skewness

dataset.skew()

dataset['Salary'].skew()

#  Standard Error

dataset.sem()

dataset['Salary'].sem() 

# inferentional Statistics
# Z-score

import scipy.stats as stats
dataset.apply(stats.zscore)

stats.zscore(dataset['Salary']) 

#  Degree of Freedom

a = dataset.shape[0]
b = dataset.shape[1]

degree_of_freedom = a-b
print(degree_of_freedom) 

y_mean = np.mean(y)
# Sum of Squares Regression (SSR)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

# SSE
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

# SST 
mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

# R2 Square

r_square = 1 - (SSR/SST)
r_square

bias = regressor.score(x_train,y_train)
print(bias)

variance = regressor.score(x_test,y_test)
print(variance)
