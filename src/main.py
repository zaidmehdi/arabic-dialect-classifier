import os

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .utils import load_data


# Load model
model_name = "moussaKam/AraBART"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=21)

models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
model_file = os.path.join(models_dir, 'best_model_checkpoint.pth')
if os.path.exists(model_file):
    with open(model_file, "rb") as f:
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint)
else:
    print(f"Error: {model_file} not found.")

# Load label encoder
encoder_file = os.path.join(models_dir, 'label_encoder.pkl')
label_encoder = load_data(encoder_file)

# Load html
html_dir = os.path.join(os.path.dirname(__file__), "templates")
index_html_path = os.path.join(html_dir, "index.html")

if os.path.exists(index_html_path):
    with open(index_html_path, "r") as html_file:
        index_html = html_file.read()
else:
    print(f"Error: {index_html_path} not found.")


def classify_arabic_dialect(text):
    tokenized_text = tokenizer(text, return_tensors="pt")
    output = model(**tokenized_text)
    probabilities = F.softmax(output.logits, dim=1)[0]
    labels = label_encoder.inverse_transform(range(len(probabilities)))
    predictions = {labels[i]: probabilities[i] for i in range(len(probabilities))}

    return predictions


def main():
    with gr.Blocks() as demo:
        gr.HTML(index_html)

        input_text = gr.Textbox(label="Your Arabic Text")
        submit_btn = gr.Button("Submit")
        predictions = gr.Label(num_top_classes=3)
        submit_btn.click(
            fn=classify_arabic_dialect, 
            inputs=input_text, 
            outputs=predictions)
        
        gr.Markdown("## Text Examples")
        examples = gr.Examples(
            examples=[
                "واش نتا خدام ولا لا",
                "بصح راك فاهم لازم الزيت",
                "حضرتك بروح زي كدا؟ على طول النهار ده",
            ],
            inputs=input_text,
        )
        gr.HTML("""
                <p style="text-align: center;font-size: large;">
                Checkout the <a href="https://github.com/zaidmehdi/arabic-dialect-classifier">Github Repo</a>
                </p>
                """)
    
    demo.launch(server_name="0.0.0.0", server_port=8080)


if __name__ == "__main__":
    main()