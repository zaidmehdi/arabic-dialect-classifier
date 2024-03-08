import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch


def extract_hidden_state(input_text, tokenizer, language_model):
    tokens = tokenizer(input_text, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = language_model(**tokens)

    return outputs.last_hidden_state[:,0].numpy()


def serialize_data(data, output_path:str):
    with open(output_path, "wb") as f:
        pickle.dump(data, f) 


def load_data(input_path:str):
    with open(input_path, "rb") as f:
        return pickle.load(f)


def plot_confusion_matrix(y_true, y_preds):
    labels = sorted(set(y_true.tolist() + y_preds.tolist()))
    cm = confusion_matrix(y_true, y_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()