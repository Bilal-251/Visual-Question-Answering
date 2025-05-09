from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and tokenizer
model = load_model('vqa_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load DenseNet feature extractor
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model

def load_fe_model():
    base_model = DenseNet201(include_top=True, weights='imagenet')
    return Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

fe_model = load_fe_model()

# Prediction function
def predict_answer(image_path, question, options, tokenizer, model, fe_model, max_len=30, img_size=224):
    # Image feature extraction
    img = load_img(image_path, target_size=(img_size, img_size))
    img = img_to_array(img) / 255.
    img = np.expand_dims(img, axis=0)
    img_feat = fe_model.predict(img, verbose=0).reshape(1, -1)

    # Text processing
    qopts = [question + " " + opt for opt in options]
    tokens = tokenizer.texts_to_sequences(qopts)
    padded = pad_sequences(tokens, maxlen=max_len, padding='post')
    text_feat = np.array(padded).reshape(1, 4, max_len)

    # Predict
    pred = model.predict([text_feat, img_feat])
    return options[np.argmax(pred)]

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        image = request.files['image']
        question = request.form['question']
        options = [request.form[f'option{i}'] for i in range(1, 5)]

        path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(path)

        result = predict_answer(path, question, options, tokenizer, model, fe_model)

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
