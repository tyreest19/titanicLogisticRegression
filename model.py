import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def costFunction(x, y, m, theta):
    #return np.transpose(y) * np.log(sigmoid(x * theta)) + np.transpose(1 - y) * np.log(1 - sigmoid(x * theta))
    loss = 0
    for i in range(m):
        z = np.dot(
            np.transpose(theta),
            x[i]
        )
        loss += y[i] * np.log(sigmoid(z)) + (1 - y[i]) * np.log(1 - sigmoid(z))
    return -(1/m) * loss

def gradientDescent(x, y, m, theta, alpha, iterations=1500):
    #return theta - (alpha/m)* np.transpose(x) * (sigmoid(x * theta) - y)
    for iteration in range(iterations):
        for j in range(len(theta)):
            gradient = 0
            for i in range(m):
                z = np.dot(
                    np.transpose(theta),
                    x[i]
                )
                gradient += (sigmoid(z) - y[i]) * x[i]
            theta[j] = theta[j] - ((alpha/m) * gradient)
        print('Current Error is:', costFunction(x, y, m, theta))
    return theta

def test(x, y, m, theta):
    correct = 0
    for i in range(m):
        pass

if __name__ == '__main__':

    training_data = pd.read_csv('data/train.csv')
    training_data = training_data.dropna() # Removes any rows containing Null values

    features = np.asarray([[age] for age in training_data['Age']])
    actual_values = np.asarray(training_data['Survived'])
    theta = np.random.uniform(size=len(features[0]))
    #print(theta)
    #print(len(features))
    #print(costFunction(features, actual_values, len(features), theta))
    #print(np.transpose(theta))
    print('Final theta\'s \n', gradientDescent(features, actual_values, len(features[0]), theta, 0.001))
