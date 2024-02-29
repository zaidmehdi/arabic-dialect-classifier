import os
import pickle

from flask import Flask, request, jsonify
from transformers import AutoModel, AutoTokenizer

from src.utils import extract_hidden_state

app = Flask(__name__)

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


@app.route("/classify", methods=["POST"])
def classify_arabic_dialect():
    try:
        data = request.json
        text = data.get("text")
        if not text:
            return jsonify({"error": "No text has been received"}), 400
        
        text_embeddings = extract_hidden_state(text, tokenizer, language_model)
        predicted_class = model.predict(text_embeddings)[0]
        
        return jsonify({"class": predicted_class}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    app.run(debug=True)


if __name__ == "__main__":
    main()