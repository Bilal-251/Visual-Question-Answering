# Visual Question Answering (VQA) Web App

This project is a Flask-based web application for Visual Question Answering (VQA), designed to answer multiple-choice questions based on educational images (mainly scientific diagrams). It combines **deep learning models** using **CNN + LSTM**, along with **textual embeddings (Keras Tokenizer and BERT)**, to understand both visual and textual input.



## 🔍 Project Overview

The system allows users to:

- Upload an image (e.g., scientific diagram).
- Enter a question and four answer options.
- Get the model’s predicted answer as output.

It is powered by a hybrid deep learning model trained on a **combined dataset of AI2D and TQA**.



## 🚀 Features

- Flask-based interactive frontend.
- Image and text preprocessing.
- Multi-modal neural network: CNN (for image) + LSTM (for text).
- Optionally supports BERT-based embeddings.
- Automatically predicts the most likely correct answer out of four options.



## 🧠 Tech Stack

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS (with animations)
- **Deep Learning:** TensorFlow / Keras, HuggingFace Transformers
- **Data:** Combined AI2D + TQA Dataset

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Bilal-251/Visual-Question-Answering.git
cd Visual-Question-Answering

### 2. Create a Virtual Environment (Recommended)

python -m venv venv
 venv\Scripts\activate

### 3. Install Dependencies

pip install -r requirements.txt


### 4. Run the App

python app.py

