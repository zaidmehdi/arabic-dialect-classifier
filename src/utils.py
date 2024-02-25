from datasets import DatasetDict, Dataset
import pandas as pd


def convert_df_to_dataset_dict(df_train:pd.DataFrame, df_test:pd.DataFrame)-> DatasetDict:
    mapper = {"#2_tweet": "tweet", "#3_country_label": "label"}
    df_train = df_train.rename(columns=mapper)
    df_test = df_test.rename(columns=mapper)

    columns_to_keep = ["tweet", "label"]
    train_dataset = Dataset.from_pandas(df_train[columns_to_keep])
    test_dataset = Dataset.from_pandas(df_test[columns_to_keep])
    
    return DatasetDict({'train': train_dataset, 'test': test_dataset})