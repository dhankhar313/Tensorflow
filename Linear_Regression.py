import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn import model_selection
from sklearn.utils import shuffle

data = pd.read_csv('student-mat.csv', sep=";")
# print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=25)

# print(x_train)
# print('-------------------------------------')
# print(x_test)
# print('-------------------------------------')
# print(y_train)
# print('-------------------------------------')
# print(y_test)
# print('-------------------------------------')

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)

score = linear.score(x_test, y_test)

print(score)

print("Coefficient or m: \n", linear.coef_)
print("Intercept or c: \n", linear.intercept_)

predictions = linear.predict(x_test)

print('Predicted Value------Actual Value:')
for i in range(len(predictions)):
    print(int(predictions[i]), y_test[i])
