import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import DatasetDict, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer

from .utils import serialize_data, load_data


class PreProcessor:
    def __init__(self, model_name, train_path:str, test_path:str, output_path:str):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.df_train = pd.read_csv(train_path, sep="\t")
        self.df_test = pd.read_csv(test_path, sep="\t")
        self.output_path = output_path

    def _get_datasetdict_object(self):
        mapper = {"#2_tweet": "tweet", "#3_country_label": "label"}
        columns_to_keep = ["tweet", "label"]

        df_train = self.df_train.rename(columns=mapper)[columns_to_keep]
        df_test = self.df_test.rename(columns=mapper)[columns_to_keep]

        train_dataset = Dataset.from_pandas(df_train)
        test_dataset = Dataset.from_pandas(df_test)
        data = DatasetDict({'train': train_dataset, 'test': test_dataset})

        return data
    
    def _tokenize(self, batch):
        return self.tokenizer(batch["tweet"], padding=True)

    def _encode_data(self, data):
        data_encoded = data.map(self._tokenize, batched=True, batch_size=None)
        return data_encoded

    def _extract_hidden_states(self, batch):
        inputs = {k:v.to(self.device) for k,v in batch.items()
                    if k in self.tokenizer.model_input_names}
        with torch.no_grad():
            last_hidden_state = self.model(**inputs).last_hidden_state

        return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}
    
    def _get_features(self, data_encoded):
        data_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        data_hidden = data_encoded.map(self._extract_hidden_states, batched=True, batch_size=50)
        return data_hidden  
        
    def preprocess_data(self):
        data = self._get_datasetdict_object()
        data_encoded = self._encode_data(data)
        data_hidden = self._get_features(data_encoded)
        serialize_data(data_hidden, output_path=self.output_path)


class ClassificationHead(nn.Module):
    def __init__(self, ) -> None:
        super(ClassificationHead, self).__init__()


class Model():
    def __init__(self, data_input_path:str, model_name:str):
        self.model_name = model_name
        self.model = None
        self.data = load_data(input_path=data_input_path)
        self.X_train = np.array(self.data["train"]["hidden_state"])
        self.X_test = np.array(self.data["test"]["hidden_state"])
        self.y_train = np.array(self.data["train"]["label"])
        self.y_test = np.array(self.data["test"]["label"])

    def _train_logistic_regression(X_train, y_train):
        lr_model = LogisticRegression(multi_class='multinomial', 
                                    class_weight="balanced", 
                                    max_iter=1000, 
                                    random_state=2024)
        lr_model.fit(X_train, y_train)
        return lr_model
    
    def _train_classification_head(X_train, y_train, base, train_base=False):
        
        return

    def train_model(self, output_path):
        if self.model_name == "lr":
            self.model = self._train_logistic_regression(self.X_train, self.y_train)
        elif self.model_name == "classification_head":
            self.model = self._train_classification_head(self.X_train, self.y_train)
        else:
            raise ValueError(f"Model name {self.model_name} does not exist. Please try 'lr'!")

        serialize_data(self.model, output_path)
    
    def _get_metrics(self, y_true, y_preds):
        accuracy = accuracy_score(y_true, y_preds)
        f1_macro = f1_score(y_true, y_preds, average="macro")
        f1_weighted = f1_score(y_true, y_preds, average="weighted")
        print(f"Accuracy: {accuracy}")
        print(f"F1 macro average: {f1_macro}")
        print(f"F1 weighted average: {f1_weighted}")

    def evaluate_predictions(self):
        train_preds = self.model.predict(self.X_train)
        test_preds = self.model.predict(self.X_test)

        print(self.model_name)
        print("\nTrain set:")
        self._get_metrics(self.y_train, train_preds)
        print("-"*50)
        print("Test set:")
        self._get_metrics(self.y_test, test_preds)


def main():
    file_path = "../data/data_hidden.pkl"
    preprocessor = PreProcessor(model_name="moussaKam/AraBART",
                                train_path="../data/DA_train_labeled.tsv",
                                test_path="../data/DA_dev_labeled.tsv",
                                output_path=file_path)
    preprocessor.preprocess_data()
    model = Model(data_input_path=file_path, model_name="lr")
    model.train_model("../models/logistic_regression.pkl")
    model.evaluate_predictions()

if __name__ == "__main__":
    main()

