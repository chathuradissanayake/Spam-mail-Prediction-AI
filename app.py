from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

app = Flask(__name__)

# Load the trained model
model = load_model('spam_classifier.h5')

import pickle

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_sequence_length = 100  
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['text']
    tokens = nltk.word_tokenize(data)
    sequences = tokenizer.texts_to_sequences([tokens])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)


    # Make a prediction
    prediction = model.predict(padded_sequences)
    print(f"Prediction raw output: {prediction}")  # Debugging line
    prediction_label = 'spam' if prediction[0][0] > 0.5 else 'ham'
    print(f"Predicted label: {prediction_label}")  # Debugging line
    
    return jsonify({'prediction': prediction_label})

if __name__ == "__main__":
    app.run(debug=True)
