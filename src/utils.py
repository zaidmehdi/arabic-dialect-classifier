import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix


def get_metrics(y_true, y_preds):
    accuracy = accuracy_score(y_true, y_preds)
    f1_macro = f1_score(y_true, y_preds, average="macro")
    f1_weighted = f1_score(y_true, y_preds, average="weighted")
    print(f"Accuracy: {accuracy}")
    print(f"F1 macro average: {f1_macro}")
    print(f"F1 weighted average: {f1_weighted}")


def evaluate_predictions(model:str, train_preds, y_train, test_preds, y_test):
    print(model)
    print("\nTrain set:")
    get_metrics(y_train, train_preds)
    print("-"*50)
    print("Test set:")
    get_metrics(y_test, test_preds)


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