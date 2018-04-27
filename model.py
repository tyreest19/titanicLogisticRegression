import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
                gradient += (sigmoid(z) - y[i]) * x[i][j]
            theta[j] = theta[j] - ((alpha/m) * gradient)
        print('Current Error is:', costFunction(x, y, m, theta))
    return theta

def test(x, y, m, theta):
    correct = 0
    for i in range(m):
        z = np.dot(
                np.transpose(theta),
                x[i]
            )
        predicted_value = sigmoid(z)
        if predicted_value >= 0.5 and y[i] == 1:
            correct += 1
        elif predicted_value < 0.5 and y[i] == 0:
            correct += 1
    return correct/m, (1 - (correct/m))

def submit(theta, dataset):
    data = {'PassengerId': [], 'Survived': []}
    submission = pd.DataFrame(data=data, dtype=int)
    count = 0
    total = 0
    for row in dataset.iterrows():
        z = np.dot(np.transpose(theta), [row[1]['Age']/100, row[1]['Fare']/100])
        prediction = sigmoid(z)
        if not math.isnan(prediction):
            submission.loc[count] = [int(row[1]['PassengerId']), int(prediction + 0.5)]
            count += 1
        total +=1
    submission = submission.set_index('PassengerId')
    print(count, total)
    submission.to_csv('data/submission.csv')

def replaceNans(dataframe, columnName):
    for index, row in dataframe.iterrows():
        if math.isnan(row[columnName]):
            dataframe.loc[index, columnName] = dataframe[columnName].mean()

if __name__ == '__main__':

    training_data = pd.read_csv('data/train.csv')
    testing_data = pd.read_csv('data/test.csv')
    #print(training_data.head())
    #print(training_data.corr())
    print(training_data['Fare'].isnull().values.any()) # No null values for fare
    print(training_data['Age'].isnull().values.any())
    replaceNans(training_data, 'Age') # Replace Nan's with average age
    print(training_data['Age'].isnull().values.any())

    X = np.asarray([[age/100, fare/100] for age, fare  in zip(training_data['Age'], training_data['Fare'])])
    y = np.asarray(training_data['Survived'])
    training_features, testing_features, training_output, testing_output = train_test_split(X, y, test_size=0.3,
                                                                                            train_size=0.7,
                                                                                            random_state=42)
    theta = np.random.uniform(0, 0.1, size=len(training_features[0]))
    print('Final theta\'s \n', gradientDescent(training_features, training_output, len(training_features[0]), theta,
                                               0.01))
    print(testing_data['Fare'].isnull().values.any()) # Prints True
    print(testing_data['Age'].isnull().values.any()) # Prints True
    replaceNans(testing_data, 'Fare')
    replaceNans(testing_data, 'Age')
    print(testing_data['Fare'].isnull().values.any())
    print(testing_data['Age'].isnull().values.any())

    submit(theta, testing_data)
    accuracy_rate, error_rate = test(testing_features, testing_output, len(testing_output), theta)
    print('Accuracy: {accuracy} \nError: {error}'.format(accuracy=accuracy_rate, error=error_rate))
