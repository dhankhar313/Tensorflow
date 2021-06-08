import pandas as pd
import numpy as np
import sklearn
import tensorflow
import pickle
from sklearn import linear_model
from sklearn import model_selection
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

data = pd.read_csv('student-mat.csv', sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

'''
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy > best:
        best = accuracy
        with open('trained_model.pickle', 'wb') as file:
            pickle.dump(linear, file)'''

with open('trained_model.pickle', 'rb') as file:
    linear = pickle.load(file)

print("Coefficient or m: \n", linear.coef_)
print("Intercept or c: \n", linear.intercept_)

predictions = linear.predict(x_test)

print('Predicted Value------Actual Value:')
for i in range(len(predictions)):
    print(int(predictions[i]), y_test[i])

name = ["G1", "G2", "studytime", "failures", "absences"]


for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.scatter(data[name[i]], data["G3"], c="blue", s=5)
    plt.xlabel(name[i], fontsize=10, labelpad=-2)
    plt.ylabel("Final Marks")

plt.show()