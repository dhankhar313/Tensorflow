import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tflearn
import json
import pickle
import time
import random

with open("intents.json") as file:
    data = json.load(file)

try:
    with open('data.pickle', 'rb') as file:
        words, tags, que_encoding, tag_encoding = pickle.load(file)

except:
    tags = []
    words = []
    questions = []
    q_category = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            temp = nltk.word_tokenize(pattern)
            words.extend(temp)
            questions.append(pattern)
            q_category.append(intent["tag"])

        if intent["tag"] not in tags:
            tags.append(intent["tag"])

    words = [stemmer.stem(i.lower()) for i in words]
    words = sorted(list(set(words)))

    tags = sorted(tags)

    que_encoding = []
    tag_encoding = []

    output_empty = [0 for i in range(len(tags))]

    for idx, question in enumerate(questions):
        bag = []

        temp = [stemmer.stem(i.lower()) for i in nltk.word_tokenize(question)]

        for word in words:
            if word in temp:
                bag.append(1)
            else:
                bag.append(0)

        q_tag = output_empty.copy()
        q_tag[tags.index(q_category[idx])] = 1

        que_encoding.append(bag)
        tag_encoding.append(q_tag)

        # print("Question: ", question)
        # print("Stemmed Question: ", temp)
        # print(len(words), len(bag))
        # print("Bag: ", bag)
        # print("Output Row: ", q_tag)

    que_encoding = np.array(que_encoding)
    tag_encoding = np.array(tag_encoding)

    with open('data.pickle', 'wb') as file:
        pickle.dump((words, tags, que_encoding, tag_encoding), file)

# print(tags)
# print(questions)
# print(q_category)
# print(words)
# print(training[0])
# print(output[0])

neural_net = tflearn.input_data(shape=[None, len(que_encoding[0])])
neural_net = tflearn.fully_connected(neural_net, 9)
neural_net = tflearn.fully_connected(neural_net, 9)
neural_net = tflearn.fully_connected(neural_net, len(tag_encoding[0]), activation="softmax")
neural_net = tflearn.regression(neural_net)
model = tflearn.DNN(neural_net)

try:
    model.load("./Model/bot.tflearn")
except:
    model.fit(que_encoding, tag_encoding, n_epoch=2000, batch_size=9, show_metric=True)
    model.save("./Model/bot.tflearn")


# noinspection PyShadowingNames
def user_input_prediction(user_string, words):
    bag = [0 for i in range(len(words))]
    user_words = [stemmer.stem(i.lower()) for i in nltk.word_tokenize(user_string)]

    for user_word in user_words:
        for idx, word in enumerate(words):
            if word == user_word:
                bag[idx] = 1

    return np.array(bag)


def chat():
    while True:
        que = input("User: ")
        if que.lower() == 'quit':
            break

        prediction = model.predict([user_input_prediction(que, words)])
        user_tag = tags[np.argmax(prediction)]

        # if prediction[user_tag] > 0.7:
        for tag in data['intents']:
            if tag['tag'] == user_tag:
                responses = tag['responses']

        print(random.choice(responses))
    else:
        print("I didn't quite understand!! Please ask another question!!")


if __name__ == '__main__':
    print(''' 
 ▄         ▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄       ▄▄  ▄▄▄▄▄▄▄▄▄▄▄ 
▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░▌     ▐░░▌▐░░░░░░░░░░░▌
▐░▌       ▐░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░▌          ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░▌░▌   ▐░▐░▌▐░█▀▀▀▀▀▀▀▀▀ 
▐░▌       ▐░▌▐░▌          ▐░▌          ▐░▌          ▐░▌       ▐░▌▐░▌▐░▌ ▐░▌▐░▌▐░▌          
▐░▌   ▄   ▐░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░▌          ▐░▌          ▐░▌       ▐░▌▐░▌ ▐░▐░▌ ▐░▌▐░█▄▄▄▄▄▄▄▄▄ 
▐░▌  ▐░▌  ▐░▌▐░░░░░░░░░░░▌▐░▌          ▐░▌          ▐░▌       ▐░▌▐░▌  ▐░▌  ▐░▌▐░░░░░░░░░░░▌
▐░▌ ▐░▌░▌ ▐░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░▌          ▐░▌          ▐░▌       ▐░▌▐░▌   ▀   ▐░▌▐░█▀▀▀▀▀▀▀▀▀ 
▐░▌▐░▌ ▐░▌▐░▌▐░▌          ▐░▌          ▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          
▐░▌░▌   ▐░▐░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄▄▄ 
▐░░▌     ▐░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌
 ▀▀       ▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀  ▀▀▀▀▀▀▀▀▀▀▀ 
                                                                                                                                                                                                                                                   
    ''')
    chat()
