import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import get_dataset, serialize_data, plot_training_history, get_model_accuracy


def train_model(model, optimizer, train_loader, val_loader, num_epochs=100, patience=10):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    num_training_steps = num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))

    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    best_valid_loss = float("inf")
    epochs_no_improve = 0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
            train_loss += loss.item()
            
            _, predicted_train = torch.max(outputs.logits, 1)
            labels_train = batch["labels"]
            correct_train += (predicted_train == labels_train).sum().item()
            total_train += labels_train.size(0)
        
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)
        
        model.eval()
        valid_loss = 0.0
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                valid_loss += loss.item()
                
                _, predicted_valid = torch.max(outputs.logits, 1)
                labels_valid = batch["labels"]
                correct_valid += (predicted_valid == labels_valid).sum().item()
                total_valid += labels_valid.size(0)
        
        valid_loss /= len(val_loader)
        valid_losses.append(valid_loss)
        
        valid_accuracy = correct_valid / total_valid
        valid_accuracies.append(valid_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.4f}')    

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_no_improve = 0
            best_model = model.state_dict()
            torch.save(best_model, "../models/best_model_checkpoint.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping after {epoch+1} epochs with no improvement.')
                break
                
    model.load_state_dict(best_model)
    history = {"train_loss": train_losses,
               "valid_loss": valid_losses,
               "train_accuracies": train_accuracies,
               "valid_accuracies": valid_accuracies}

    return model, history

    
def main():
    model_name = "moussaKam/AraBART"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset, label_encoder = get_dataset("data/DA_train_labeled.tsv", "data/DA_dev_labeled.tsv", tokenizer)
    serialize_data(label_encoder, "../models/label_encoder.pkl")

    for data in dataset:
        dataset[data] = dataset[data].remove_columns(["tweet"])
        dataset[data] = dataset[data].rename_column("label", "labels")
        dataset[data].set_format("torch")

    train_loader = DataLoader(dataset["train"], batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset["val"], batch_size=8)
    test_loader = DataLoader(dataset["test"], batch_size=8)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=21)
    for param in model.parameters():
        param.requires_grad = False # We don't retrain the pretrained model due to lack of GPU
    for param in model.classification_head.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 100

    model, history = train_model(model, optimizer, train_loader, val_loader, num_epochs=num_epochs)
    plot_training_history(history)

    test_accuracy = get_model_accuracy(model, test_loader)
    print("The accuracy of the model on the test set is:", test_accuracy)

if __name__ == "__main__":
    main()