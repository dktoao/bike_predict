from numpy import linspace, pi, sin
from numpy.random import randn
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Make up fake data
x_axis = linspace(-pi, pi, 100)
truth_data = sin(x_axis)
noisy_data = truth_data + randn(100) * 0.5

# Plot the data
plt.figure()

arr_eps = [0.5, 0.9]
num_epsilon = len(arr_eps)
arr_C = [1]
num_C = len(arr_C)

for m, eps in enumerate(arr_eps):
    for n, C in enumerate(arr_C):

        model = SVR(C=C, epsilon=eps)
        x_data = x_axis.reshape(-1, 1)
        model.fit(x_data, noisy_data)
        guess = model.predict(x_data)

        n_rows = max(num_epsilon, num_C)
        n_cols = min(num_epsilon, num_C)

        ax = plt.subplot(n_rows, n_cols, (m*n_cols) + n + 1)
        plt.plot(x_axis, guess)
        plt.plot(x_axis, guess + eps, 'r-')
        plt.plot(x_axis, guess - eps, 'r-')
        plt.scatter(x_axis, noisy_data, marker='+', color='g')
        plt.title('C = {0:.2f}, eps = {1:.2f}'.format(C, eps))
        plt.xlim([-pi, pi])
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()