import predict_bike_use as pbu
from numpy import mean, var
from numpy import dot, zeros
from numpy import array
from numpy import abs
from numpy.linalg import pinv
from sklearn import svm

a, y = pbu.import_data('train.csv')
b = pbu.import_data('test.csv', False)

# Support Vector Regression
#clf = svm.SVR(kernel='poly', C=1e2, degree=2)
#clf.fit(A, b)
#result = zeros((B.shape[0], 1))
#for n, val in enumerate(B[:]):
#    result[n] = clf.predict(val)

# Simple linear regression
out = pbu.evaluate_predictor(a, y, pbu.simple_linear_regression)
print(out)
print(mean(out))
print(var(out))

