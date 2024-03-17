from transformers import AutoTokenizer, AutoModel

from utils import get_dataset


class Model():
    def __init__(self) -> None:
        pass


def main():
    model_name = "moussaKam/AraBART"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    dataset = get_dataset("data/DA_train_labeled.tsv", "data/DA_dev_labeled.tsv", tokenizer)
    
    print(dataset["train"])


if __name__ == "__main__":
    main()