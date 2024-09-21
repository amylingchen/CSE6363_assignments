
import numpy as np
from sklearn.metrics import accuracy_score

from models import GaussianNBModel
from utils import load_and_prepare_data, print_result



def rgb_benchmark():
    print("Loading data...", end="")
    # Load and prepare the data
    train_data, train_labels, test_data, test_labels = load_and_prepare_data()
    print("done.")

    # Reshape the data
    train_data = train_data.reshape(len(train_data), -1)
    test_data = test_data.reshape(len(test_data), -1)


    # Create and fit the LDA model
    gnb = GaussianNBModel()
    print("Fitting GaussianNB model...", end="")
    gnb.fit(train_data, train_labels)
    print("done.")

    # Predict the test set labels
    print("Predicting test set labels...", end="")
    test_preds = gnb.predict(test_data)
    print("done.")

    # print_result(test_labels, test_preds)

    # train_pred = gnb.predict(train_data)
    # train_acc = accuracy_score(train_labels, train_pred)
    # print(f"Train accuracy: {train_acc}")
    # Calculate test set accuracies
    test_acc = accuracy_score(test_labels, test_preds)
    print(f"Test accuracy: {test_acc}")

def grayscale_benchmark():
    print("Loading data...", end="")
    # Load and prepare the data
    train_data, train_labels, test_data, test_labels = load_and_prepare_data(True)
    print("done.")

    # Reshape the data
    train_data = train_data.reshape(len(train_data), -1)
    test_data = test_data.reshape(len(test_data), -1)

    # Create and fit the LDA model
    gnb = GaussianNBModel()
    print("Fitting GaussianNB model...", end="")
    gnb.fit(train_data, train_labels)
    print("done.")

    # Predict the test set labels
    print("Predicting test set labels...", end="")
    test_preds = gnb.predict(test_data)
    print("done.")

    # print_result(test_labels, test_preds)

    # train_pred = gnb.predict(train_data)
    # train_acc = accuracy_score(train_labels, train_pred)
    # print(f"Train accuracy: {train_acc}")

    # Calculate test set accuracies
    test_acc = accuracy_score(test_labels, test_preds)
    print(f"Test accuracy: {test_acc}")


def main():
    print("****************************************")
    print("*           GNB RGB Solution           *")
    print("****************************************")
    rgb_benchmark()

    print("\n****************************************")
    print("*           GNB Grayscale Solution       *")
    print("******************************************")
    grayscale_benchmark()


if __name__ == "__main__":
    main()