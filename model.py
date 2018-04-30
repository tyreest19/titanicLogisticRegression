import os
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import pandas as pd
import pylab
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

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

def gradientDescent(x, y, m, theta, alpha, iterations=1500, lossList=None):
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
        currentLoss = costFunction(x, y, m, theta)
        if type(lossList) is list:
            lossList.append(currentLoss)
        print('Current Error is:', currentLoss)
    return theta

def test(x, y, m, theta):
    correct = 0
    predictedValues = []
    for i in range(m):
        z = np.dot(
                np.transpose(theta),
                x[i]
            )
        predicted_value = sigmoid(z)
        predictedValues.append(predicted_value)
        if predicted_value >= 0.5 and y[i] == 1:
            correct += 1
        elif predicted_value < 0.5 and y[i] == 0:
            correct += 1
    return correct/m, (1 - (correct/m)), predictedValues

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
    os.system('kaggle competitions submit -c titanic -f data/submission.csv -m \"new Message\"')

def replaceNans(dataframe, columnName):
    for index, row in dataframe.iterrows():
        if math.isnan(row[columnName]):
            dataframe.loc[index, columnName] = dataframe[columnName].mean()

if __name__ == '__main__':
    lossList = []
    training_data = pd.read_csv('data/train.csv')
    testing_data = pd.read_csv('data/test.csv')
    #print(training_data.head())
    print(training_data.corr())
    print(training_data['Fare'].isnull().values.any()) # No null values for fare
    print(training_data['Age'].isnull().values.any())
    replaceNans(training_data, 'Age') # Replace Nan's with average age
    print(training_data['Age'].isnull().values.any())
    X = np.asarray([[age/100, fare/100] for age, fare  in zip(training_data['Age'], training_data['Fare'])])
    y = np.asarray(training_data['Survived'])
    training_features, testing_features, training_output, testing_output = train_test_split(X, y, test_size=0.3,
                                                                                            train_size=0.7,
                                                                                            random_state=42)
    theta = np.random.uniform(-1/(math.sqrt(2)), 1/(math.sqrt(2)), size=len(training_features[0]))
    print('Final theta\'s \n', gradientDescent(training_features, training_output, len(training_features[0]), theta,
                                               0.01, iterations=600000, lossList=lossList))
    print(testing_data['Fare'].isnull().values.any()) # Prints True
    print(testing_data['Age'].isnull().values.any()) # Prints True
    replaceNans(testing_data, 'Fare')
    replaceNans(testing_data, 'Age')
    print(testing_data['Fare'].isnull().values.any())
    print(testing_data['Age'].isnull().values.any())
    submit(theta, testing_data)
    accuracy_rate, error_rate, predictions = test(testing_features, testing_output, len(testing_output), theta)
    print('Accuracy: {accuracy} \nError: {error}'.format(accuracy=accuracy_rate, error=error_rate))
    plt.plot([i for i in range(len(lossList))], lossList)
    plt.title('Loss Graphed')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig('LossGraph')
    preds = predictions
    fpr, tpr, threshold = roc_curve(testing_output, preds)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

