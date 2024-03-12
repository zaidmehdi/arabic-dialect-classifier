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
    top_three_indices = np.argsort(-probabilities)[:3]

    top_three_labels = model.classes_[top_three_indices]
    top_three_probabilities = probabilities[top_three_indices]

    return top_three_labels, top_three_probabilities


with gr.Blocks() as demo:
    gr.HTML(index_html)
    input_text = gr.Textbox(label="Your Arabic Text")
    submit_btn = gr.Button("Submit")
    with gr.Row():
        first_country = gr.Textbox()
        second_country = gr.Textbox()
        third_country = gr.Textbox()
    submit_btn.click(
        fn=classify_arabic_dialect, 
        inputs=input_text, 
        outputs=[first_country, second_country, third_country])
    gr.HTML("<p>Checkout the Github Repo:</p>")


if __name__ == "__main__":
    demo.launch()

