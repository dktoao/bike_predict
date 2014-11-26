"""
Python script with various function to predict the usage of ride share bikes
"""

# Imports
from random import sample

from numpy import array, zeros, dot, sqrt, log, sum
from numpy.linalg import pinv


def import_data(file_name, output=True, col_skip=[]):
    """
    Imports the CSV file with the

    :param file_name: name of the csv file with training data
    :param output: if the file has output data
    :return: numpy arrays (A, b) with the data matrix and solution
    """

    raw_data = open(file_name)
    k = 0
    data_matrix = []
    training_vector = []

    for line in raw_data:
        if k != 0:
            raw_data_point = line.strip().split(',')

            # Get Date and time data
            date, time = raw_data_point[0].split(' ')
            year, month, day = date.split('-')
            hour = time.split(':')[0]

            # Concatenate with the rest of the data
            pts = [x for x in range(0, 12) if x not in col_skip]
            clean_data = [year, month, day, hour]
            clean_data.extend(raw_data_point[1:9])

            # Convert to float and append
            clean_data = [float(x) for x in clean_data]
            data_matrix.append(clean_data)

            if output:
                # Get the value for training
                training_vector.append(float(raw_data_point[11]))

        k += 1

    if output:
        return array(data_matrix)[:, pts], array(training_vector)
    else:
        return array(data_matrix)[:, pts]


def write_data(data, result, file_name):
    """
    Writes data to format specified by kaggle, will zero out any negative value.

    :param result: Prediction result
    :param file_name: name of file to write to ( will append .csv if not specified)
    """

    # Put correct extension on filename
    if '.csv' not in file_name:
        file_name += '.csv'

    # Zero out negative values in the result
    result = result.clip(0)

    out_file = open(file_name, 'w')

    print('datetime,count', file=out_file)
    for n, val in enumerate(result):
        print('{0}-{1:02d}-{2} {3:02d}:00:00,{4}'.format(
            int(data[n, 0]),
            int(data[n, 1]),
            int(data[n, 2]),
            int(data[n, 3]),
            int(val)),
            file=out_file
        )


def evaluate_predictor(training_data, training_outcomes, predictor, train_fraction=0.9, n_trials=5):
    """
    Evaluates the performance of the function "predictor" without having to enter
    results into kaggle. Divides the known data into random sets and evaluates the
    results against the rest.

    :param training_data: Data set with known result
    :param training_outcomes: know result of A
    :param predictor: function that takes arguments func(b, a, y) and returns a prediction
    :param train_fraction: percent of data to use for training
    :param n_trials: number of times to evaluate the function
    :return: Float value representing the predicted RMSLE error
    """

    outcomes = zeros(n_trials)
    data_length = training_data.shape[0]
    num_samples = int(data_length * train_fraction)
    num_tests = data_length - num_samples
    all_indices = range(data_length)

    for trial in range(n_trials):
        # Divide data array into training set and
        training_indices = sample(all_indices, num_samples)
        training_indices.sort()
        test_indices = [x for x in all_indices if x not in training_indices]
        test_indices.sort()

        # Create the training dataset
        sampled_training_data = training_data[training_indices]
        sampled_training_outcomes = training_outcomes[training_indices]
        sampled_test_data = training_data[test_indices]
        sampled_test_outcomes = training_outcomes[test_indices]

        # Predict outcomes and save values
        predicted_outcome = predictor(sampled_test_data, sampled_training_data, sampled_training_outcomes)
        rmsle = sqrt((1/num_tests)*sum((log(predicted_outcome + 1) - log(sampled_test_outcomes + 1))**2))
        outcomes[trial] = rmsle

    return outcomes


def simple_linear_regression(test_data, training_data, training_outcomes):
    """
    Predicts result of data set B from similar data set A with known result y

    :param test_data: Data set with no known result
    :param training_data: Data set with known result
    :param training_outcomes: known result of A
    :return: Predicted result from data set B
    """

    # Create the matrix for training matrix
    n_rows, n_cols = training_data.shape
    train_matrix = zeros((n_rows, n_cols + 1))
    for row in range(n_rows):
        for col in range(n_cols):
            train_matrix[row, col] = training_data[row, col]
        train_matrix[row, n_cols] = 1

    # Create matrix for the unknown linear equation
    n_rows, n_cols = test_data.shape
    predict_matrix = zeros((n_rows, n_cols + 1))
    for row in range(n_rows):
        for col in range(n_cols):
            predict_matrix[row, col] = test_data[row, col]
        predict_matrix[row, n_cols] = 1

    # Find new function weights
    w = dot(pinv(train_matrix), training_outcomes)
    result = dot(predict_matrix, w)
    result = result.clip(0)
    # Return the predicted values
    return result