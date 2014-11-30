from numpy import column_stack
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Organize data
data = read_csv('train.csv')
hour = [int(item.split(' ')[1].split(':')[0]) for item in data['datetime']]
date = [item.split(' ')[0].split('-') for item in data['datetime']]
year = [int(item[0]) for item in date]
month = [int(item[1]) for item in date]
day = [int(item[2]) for item in date]

data['year'] = year
data['month'] = month
data['day'] = day
data['hour'] = hour

# Do a least squares fit in the x-axis
X = data['hour'].as_matrix()
X = X.reshape(-1, 1)
print(X.shape)
y = data['count'].as_matrix()
y = y.reshape(-1, 1)
model = LinearRegression(fit_intercept=False)
model.fit(X, y)
lin_out = model.predict(X)

hourly_data = data.groupby('hour').mean()
plt.figure('Average Bike Use By Hour')
plt.title('Average Bike Use By Hour')
plt.plot(hourly_data.index, hourly_data['count'], 'b', label='Actual Use')
plt.plot(X, lin_out, 'r', label='Linear Best Fit')
plt.xlim((0, 23))
plt.grid()
plt.legend(loc=0)
plt.xlabel('Time of Day (hour)')
plt.ylabel('Average Number of Bikes Rented')
plt.show()