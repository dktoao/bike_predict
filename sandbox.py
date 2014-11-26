import predict_bike_use as pbu
from numpy import mean, var, array, diag
from scipy.linalg import svd
from sklearn import linear_model
import matplotlib.pyplot as plt

# Load data from csv files
a, y = pbu.import_data('train.csv')  #, col_skip=[4, 9])
b = pbu.import_data('test.csv', False)  #, col_skip=[4, 9])

# Simple linear regression
#out = pbu.evaluate_predictor(a, y, pbu.simple_linear_regression, n_trials=5)
#print('Mean: {0:.3e}, Var: {1:.3e}'.format(mean(out), var(out)))
