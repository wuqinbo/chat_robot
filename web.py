import random
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import json



model = keras.models.load_model('chatbot_model.h5')

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    # get the user input
    input_text = request.form['input_text']
    # get the prediction
    predicted_class, confidence = get_prediction(input_text)
    # create a response
    response = {'class': predicted_class, 'confidence': float(confidence)}
    return jsonify(response)


def get_prediction(input_text):
    lemmatizer = WordNetLemmatizer()
    words = ['hello', 'how', 'are', 'you', 'today', 'weather', 'in', 'what', 'time', 'is', 'it', 'thanks']

    classes = ['greeting', 'goodbye', 'thanks']

    documents = []
    ignore_words = ['?', '!']
    bag = []
    # preprocess the input text
    input_text_words = word_tokenize(input_text)
    input_text_words = [lemmatizer.lemmatize(word.lower()) for word in input_text_words]

    # create a bag of words
    for word in words:
        if word in input_text_words:
            bag.append(1)
        else:
            bag.append(0)

    # make a prediction
    results = model.predict(np.array([bag]))[0]
    # get the predicted class
    predicted_class = classes[np.argmax(results)]
    # get the confidence score for the predicted class
    confidence = results[np.argmax(results)]
    return predicted_class, confidence


if __name__ == '__main__':
    app.run(debug=True)
