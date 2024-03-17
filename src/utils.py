import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import DatasetDict, Dataset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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


def tokenize(batch, tokenizer):
    return tokenizer(batch["tweet"], padding='max_length', max_length=256)


def get_dataset(train_path:str, test_path:str, tokenizer):
    df_train = pd.read_csv(train_path, sep="\t")
    df_train, df_val = train_test_split(df_train, test_size=0.23805, random_state=42, 
                                        stratify=df_train["#3_country_label"])
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = pd.read_csv(test_path, sep="\t")

    encoder = LabelEncoder()
    df_train["#3_country_label"] = encoder.fit_transform(df_train["#3_country_label"])
    df_val["#3_country_label"] = encoder.transform(df_val["#3_country_label"])
    df_test["#3_country_label"] = encoder.transform(df_test["#3_country_label"])

    dataset = get_datasetdict_object(df_train, df_val, df_test)
    dataset = dataset.map(lambda x: tokenize(x, tokenizer), batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    return dataset

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