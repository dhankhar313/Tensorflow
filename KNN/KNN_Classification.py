import sklearn
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

data = pd.read_csv('car.csv')
# print(data.head())

encode = preprocessing.LabelEncoder()
buying = encode.fit_transform(list(data['buying']))
maint = encode.fit_transform(list(data['maint']))
door = encode.fit_transform(list(data['door']))
persons = encode.fit_transform(list(data['persons']))
lug_boot = encode.fit_transform(list(data['lug_boot']))
safety = encode.fit_transform(list(data['safety']))
cls = encode.fit_transform(list(data['cls']))

X = list(zip(buying, maint, door, persons, lug_boot, safety))
# print(X)
y = list(cls)
# print(y)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
model = KNeighborsClassifier(n_neighbors=9)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(accuracy)

predictions = model.predict(X_test)

print('Predicted Value------Actual Value:')
for i in range(len(predictions)):
    print(int(predictions[i]), y_test[i])
