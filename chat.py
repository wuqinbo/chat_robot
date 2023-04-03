import random
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

with open('intents.json') as file:
    data = json.load(file)

words = []
classes = []
documents = []
ignore_words = ['?', '!']

for intent in data['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = word_tokenize(pattern)
        # add word to the words list
        words.extend(w)
        # add word/sentence to the documents list
        documents.append((w, intent['tag']))
        # add tag to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize and lowercase each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# sort classes list
classes = sorted(list(set(classes)))

# create training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# shuffle the training data and convert to numpy array
random.shuffle(training)
training = np.array(training)

# split training data into input and output
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# define model architecture
model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(train_y[0]), activation='softmax')
])

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5)

# save the model
model.save('chatbot_model.h5')

# load the model
model = keras.models.load_model('chatbot_model.h5')


# define a function to preprocess the input text
def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)

