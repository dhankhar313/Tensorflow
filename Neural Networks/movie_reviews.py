import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=100000)
# print(train_data[0], "\n", train_labels[0], "\n", test_data[0], "\n", test_labels[0])

word_index = data.get_word_index()
word_index = {key: (value + 3) for (key, value) in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

word_index_reversed = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=256)


def decode_review(text):
    return " ".join([word_index_reversed.get(i, "?") for i in text])


'''
model = keras.Sequential()
model.add(keras.layers.Embedding(100000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, batch_size=512, epochs=40, validation_data=(x_val, y_val), verbose=1)

accuracy_score = model.evaluate(test_data, test_labels)

print("Model Accuracy: ", accuracy_score[-1])

model.save("reviews.h5")
'''


def encode_review(review):
    encoded_review = [1]
    for word in review:
        if word.lower() in word_index:
            encoded_review.append(word_index[word.lower()])
        else:
            encoded_review.append(2)
    return encoded_review


model = keras.models.load_model("reviews.h5")

with open("test.txt") as file:
    for line in file.readlines():
        word_list = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace("\"", "").replace(
            ":", "").strip().split(" ")
        encoded_list = encode_review(word_list)
        encoded_list = keras.preprocessing.sequence.pad_sequences([encoded_list], value=word_index["<PAD>"],
                                                                  padding="post", maxlen=256)
        prediction = model.predict(encoded_list)
        print("Actual Line: ", line)
        print("Encoded Line: ", encoded_list)
        print("Prediction: ", prediction[0])
