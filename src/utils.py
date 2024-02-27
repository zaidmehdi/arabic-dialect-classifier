from sklearn.metrics import accuracy_score, f1_score


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