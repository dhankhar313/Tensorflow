import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn import model_selection
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

dataset = datasets.load_breast_cancer()
# print(list(dataset.feature_names))
# print(list(dataset.target_names))

X = dataset.data
y = dataset.target

# classes = ['malignant', 'benign']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

classifier1 = svm.SVC(kernel='linear', C=2)
classifier1.fit(X_train, y_train)

classifier2 = KNeighborsClassifier(n_neighbors=15)
classifier2.fit(X_train, y_train)

predictions1 = classifier1.predict(X_test)
accuracy1 = metrics.accuracy_score(predictions1, y_test)

predictions2 = classifier2.predict(X_test)
accuracy2 = metrics.accuracy_score(predictions2, y_test)

print(accuracy1, accuracy2)
