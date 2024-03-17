import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

from utils import get_datasetdict_object


model_name = "moussaKam/AraBART"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_dataset(train_path:str, test_path:str):
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

    return get_datasetdict_object(df_train, df_val, df_test)


def tokenize(batch):
    return tokenizer(batch["tweet"], padding=True)


def main():
    dataset = get_dataset("data/DA_train_labeled.tsv", "data/DA_dev_labeled.tsv")
    dataset = dataset.map(tokenize, batched=True)
    
    print(set(dataset["train"]["label"]))


if __name__ == "__main__":
    main()