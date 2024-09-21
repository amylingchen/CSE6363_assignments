import logging

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from models import QDAModel
from utils import load_and_prepare_data, print_result
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rgb_benchmark():
    print("Loading data...", end="")
    # Load and prepare the data
    train_data, train_labels, test_data, test_labels = load_and_prepare_data()
    print("done.")

    # Reshape the data
    train_data = train_data.reshape(len(train_data), -1)
    test_data = test_data.reshape(len(test_data), -1)



    # # principal component analysis data
    # print("Dimensionality reduction data...", end="")
    # pca = PCA()
    # pca.fit(train_data)
    # cumulative_variances = np.cumsum(pca.explained_variance_ratio_)
    # threshold = 0.90
    # n_components = np.argmax(cumulative_variances >= threshold) + 1
    # pca = PCA(n_components=n_components)
    # train_data = pca.fit_transform(train_data)
    # test_data = pca.transform(test_data)
    # print("done.")
    # Create and fit the LDA model
    qda = QDAModel()
    print("Fitting QDA model...", end="")
    qda.fit(train_data, train_labels)
    print("done.")

    # Predict the test set labels
    test_preds = qda.predict(test_data)
    #
    # print_result(test_labels, test_preds)

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
    qda1 = QDAModel()
    print("Fitting QDA model...", end="")
    qda1.fit(train_data, train_labels)
    print("done.")

    # Predict the test set labels
    test_preds = qda1.predict(test_data)

    # print_result(test_labels, test_preds)
    test_acc = accuracy_score(test_labels, test_preds)
    print(f"Test accuracy: {test_acc}")


from sklearn import datasets


def digit_benchmark():
    print("Loading data...", end="")
    # Load and prepare the data

    data = datasets.load_digits()
    X = data.images.reshape((len(data.images), -1))
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("done.")

    # Create and fit the LDA model
    qda1 = QDAModel()
    print("Fitting QDA model...", end="")
    qda1.fit(X_train, y_train)
    print("done.")

    # Predict the test set labels
    print("Predicting test set labels...", end="")
    y_preds = qda1.predict(X_test)
    print("done.")

    # Calculate test set accuracies
    test_acc = accuracy_score(y_test, y_preds)
    print(f"Test accuracy: {test_acc}")


def main():
    print("****************************************")
    print("*           QDA RGB Solution           *")
    print("****************************************")
    rgb_benchmark()

    print("\n****************************************")
    print("*           QDA Grayscale Solution       *")
    print("******************************************")
    grayscale_benchmark()


if __name__ == "__main__":
    main()
