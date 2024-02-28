import pickle


def main():
    with open("../models/logistic_regression.pkl", "rb") as f:
        model = pickle.load(f)
        
    return


if __name__ == "__main__":
    main()