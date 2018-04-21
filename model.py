import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def sigmoid(x, theta):
    z = np.dot(
        np.transpose(theta),
        x
    )
    return 1/(1 + np.exp(-z))

def costFunction(x, y, m, theta):
    loss = 0
    for i in range(m):
        loss += y[i] * np.log(sigmoid(x, theta)) + (1 - y[i]) * np.log(1 - sigmoid(x, theta))
    return -(1/m)

def gradientDescent(x, y, m, theta, alpha, iterations=1500):
    for i in range(iterations):
        gradient = 0
        for j in range(m):
            gradient += (sigmoid(x[j], theta) - y[j]) * x[j]
        theta = theta - ((alpha/m) * gradient)
        print('Current Error is:', costFunction(x, y, m, theta))
    return theta

if __name__ == '__main__':

    training_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    training_data = training_data.dropna() # Removes any rows containing Null values
    print(len(training_data['Age']) == len(training_data['Survived']))

    theta = np.random.uniform(size=len(training_data['Age']))
    features = np.asarray(training_data['Age'])
    actual_values = np.asarray(training_data['Survived'])
    print('Final theta\'s \n', gradientDescent(features, actual_values, len(features), theta, 0.0000001))
    #print(training_data.head())
    # print(training_data.corr())

