from numpy import column_stack, arange, mean, var
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from predict_bike_use import evaluate_predictor, support_vector_regression, write_data

# Organize data
data = []
for set_name in ['train.csv', 'test.csv']:

    data_set = read_csv(set_name)
    hour = [int(item.split(' ')[1].split(':')[0]) for item in data_set['datetime']]
    date = [item.split(' ')[0].split('-') for item in data_set['datetime']]
    year = [int(item[0]) for item in date]
    month = [int(item[1]) for item in date]
    day = [int(item[2]) for item in date]

    data_set['year'] = year
    data_set['month'] = month
    data_set['day'] = day
    data_set['hour'] = hour

    data.append(data_set)

training_data = data[0]
test_data = data[1]

# Get data of interest
X = training_data[['year', 'month', 'hour', 'workingday', 'weather', 'windspeed']].as_matrix()
X_p = test_data[['year', 'month', 'hour', 'workingday', 'weather', 'windspeed']].as_matrix()
X_w = test_data[['year', 'month', 'day', 'hour']].as_matrix()
#X = X.reshape(-1, 1)
y = training_data['count'].as_matrix()
#y = y.reshape(-1, 1)
#print(X_p)

# Evaluate SVM with a one input variable
out = evaluate_predictor(X, y, support_vector_regression, n_trials=3, C=1, epsilon=1)
print('Estimated Error:\n\tMean: {0:.3e}\n\tVar: {1:.3e}'.format(mean(out), var(out)))

# Linear Regression
#model = LinearRegression(fit_intercept=False)
#model.fit(X, y)
#lin_out = model.predict(X)

# SVR
#model = SVR(C=10)
#model.fit(X, y)
#lin_out = model.predict(X_p)
#write_data(X_w, lin_out, 'out_svr2')

'''
# Plot data
hourly_data = data.groupby('hour').mean()
plt.figure('Average Bike Use By Hour')
plt.title('Average Bike Use By Hour')
plt.plot(hourly_data.index, hourly_data['count'], 'b', label='Actual Use')
#plt.plot(X_new, lin_out, 'r', label='Linear Best Fit')
plt.plot(X_new, lin_out, 'r', label='SVM Gaussian Fit')
plt.xlim((0, 23))
plt.grid()
plt.legend(loc=0)
plt.xlabel('Time of Day (hour)')
plt.ylabel('Average Number of Bikes Rented')
plt.show()
'''