import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig

from utils import get_dataset


class Model(nn.Module):
    def __init__(self, model_name, config, num_labels):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, config=config)
        self.classification_head = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classification_head(pooled_output)
        probabilities = F.softmax(logits, dim=-1)
        
        return probabilities
    

def main():
    model_name = "moussaKam/AraBART"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name) 

    dataset = get_dataset("data/DA_train_labeled.tsv", "data/DA_dev_labeled.tsv", tokenizer)
    num_labels = len(set(dataset["train"]["label"]))
    model = Model(model_name, config, num_labels)
    
    print(dataset["train"])


if __name__ == "__main__":
    main()