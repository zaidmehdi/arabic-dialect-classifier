import os
import pickle

import gradio as gr
import numpy as np
from transformers import AutoModel, AutoTokenizer

from .utils import extract_hidden_state


# Load model
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
model_file = os.path.join(models_dir, 'logistic_regression.pkl')

if os.path.exists(model_file):
    with open(model_file, "rb") as f:
        model = pickle.load(f)
else:
    print(f"Error: {model_file} not found.")

# Load html
html_dir = os.path.join(os.path.dirname(__file__), "templates")
index_html_path = os.path.join(html_dir, "index.html")

if os.path.exists(index_html_path):
    with open(index_html_path, "r") as html_file:
        index_html = html_file.read()
else:
    print(f"Error: {index_html_path} not found.")

# Load pre-trained model
model_name = "moussaKam/AraBART"
tokenizer = AutoTokenizer.from_pretrained(model_name)
language_model = AutoModel.from_pretrained(model_name)


def classify_arabic_dialect(text):
    text_embeddings = extract_hidden_state(text, tokenizer, language_model)
    probabilities = model.predict_proba(text_embeddings)[0]
    labels = model.classes_
    predictions = {labels[i]: probabilities[i] for i in range(len(probabilities))}

    return predictions


with gr.Blocks() as demo:
    gr.HTML(index_html)
    input_text = gr.Textbox(label="Your Arabic Text")
    submit_btn = gr.Button("Submit")
    predictions = gr.Label(num_top_classes=3)
    submit_btn.click(
        fn=classify_arabic_dialect, 
        inputs=input_text, 
        outputs=predictions)
    gr.HTML("""
            <p style="text-align: center;font-size: large;">
            Checkout the <a href="https://github.com/zaidmehdi/arabic-dialect-classifier">Github Repo</a>
            </p>
            """)


if __name__ == "__main__":
    demo.launch()