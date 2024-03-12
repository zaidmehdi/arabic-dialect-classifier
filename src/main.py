import os
import pickle

import gradio as gr
from transformers import AutoModel, AutoTokenizer

from .utils import extract_hidden_state


models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
model_file = os.path.join(models_dir, 'logistic_regression.pkl')

if os.path.exists(model_file):
    with open(model_file, "rb") as f:
        model = pickle.load(f)
else:
    print(f"Error: {model_file} not found.")

model_name = "moussaKam/AraBART"
tokenizer = AutoTokenizer.from_pretrained(model_name)
language_model = AutoModel.from_pretrained(model_name)

def classify_arabic_dialect(text):
    text_embeddings = extract_hidden_state(text, tokenizer, language_model)
    predicted_class = model.predict(text_embeddings)[0]
    
    return predicted_class

demo = gr.Interface(
    fn=classify_arabic_dialect,
    inputs=["text"],
    outputs=["text"],
)


if __name__ == "__main__":
    demo.launch()

