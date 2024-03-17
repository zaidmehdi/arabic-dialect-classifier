import pandas as pd
from sklearn.model_selection import train_test_split

from utils import get_datasetdict_object


def get_dataset(train_path:str, test_path:str):
    df_train = pd.read_csv(train_path, sep="\t")
    df_train, df_val = train_test_split(df_train, test_size=0.23805, random_state=42, 
                                        stratify=df_train["#3_country_label"])
    df_test = pd.read_csv(test_path, sep="\t")

    return get_datasetdict_object(df_train, df_val, df_test)


def main():
    dataset = get_dataset("data/DA_train_labeled.tsv", "data/DA_dev_labeled.tsv")
    print(dataset)


if __name__ == "__main__":
    main()