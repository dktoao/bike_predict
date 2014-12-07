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

for m, epsilon in enumerate([0.1, 1.0, 2.0]):
    for n, C in enumerate([1, 10, 100]):

        model = SVR(C=C, epsilon=epsilon)
        x_data = x_axis.reshape(-1, 1)
        model.fit(x_data, noisy_data)
        guess = model.predict(x_data)

        ax = plt.subplot(3, 3, (m*3) + n + 1)
        plt.plot(x_axis, guess)
        plt.plot(x_axis, guess + epsilon, 'r-')
        plt.plot(x_axis, guess - epsilon, 'r-')
        plt.scatter(x_axis, noisy_data, marker='+', color='g')
        plt.title('C = {0:.2f}, eps = {1:.2f}'.format(C, epsilon))
        plt.xlim([-pi, pi])
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()