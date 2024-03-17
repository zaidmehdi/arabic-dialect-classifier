import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import DatasetDict, Dataset
from sklearn.metrics import confusion_matrix


def get_datasetdict_object(df_train, df_val, df_test):
    mapper = {"#2_tweet": "tweet", "#3_country_label": "label"}
    columns_to_keep = ["tweet", "label"]

    df_train = df_train.rename(columns=mapper)[columns_to_keep]
    df_val = df_val.rename(columns=mapper)[columns_to_keep]
    df_test = df_test.rename(columns=mapper)[columns_to_keep]

    train_dataset = Dataset.from_pandas(df_train)
    val_dataset = Dataset.from_pandas(df_val)
    test_dataset = Dataset.from_pandas(df_test)
    
    return DatasetDict({'train': train_dataset, 'val': val_dataset,
                        'test': test_dataset})


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