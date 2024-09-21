
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from models import LDAModel
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
    lda = LDAModel()
    print("Fitting LDA model...", end="")
    lda.fit(train_data, train_labels)
    print("done.")

    # Predict the test set labels
    print("predict LDA model...", end="")
    test_preds = lda.predict(test_data)
    print("done.")

    # print_result(test_labels, test_preds)

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
    lda = LDAModel()
    print("Fitting LDA model...", end="")
    lda.fit(train_data, train_labels)
    print("done.")

    # Predict the test set labels
    print("predict LDA model...", end="")
    test_preds = lda.predict(test_data)
    print("done.")

    # print_result(test_labels, test_preds)

    # Calculate test set accuracies
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
    lda1 = LDAModel()
    print("Fitting QDA model...", end="")
    lda1.fit(X_train, y_train)
    print("done.")

    # Predict the test set labels
    print("Predicting test set labels...", end="")
    y_preds = lda1.predict(X_test)
    print("done.")

    # Calculate test set accuracies
    test_acc = accuracy_score(y_test, y_preds)
    print(f"Test accuracy: {test_acc}")
def main():
    print("****************************************")
    print("*           LDA RGB Solution           *")
    print("****************************************")
    rgb_benchmark()

    print("\n****************************************")
    print("*           LDA Grayscale Solution       *")
    print("******************************************")
    grayscale_benchmark()

if __name__ == "__main__":
    main()