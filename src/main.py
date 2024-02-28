import pickle

from transformers import AutoTokenizer


def classify_arabic_dialect(text:str, model, tokenizer) -> str:
    text_embeddings = tokenizer(text, padding=True)
    predicted_class = model.predict(text_embeddings)

    return predicted_class


def main():
    with open("../models/logistic_regression.pkl", "rb") as f:
        model = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained("moussaKam/AraBART")
    return


if __name__ == "__main__":
    main()