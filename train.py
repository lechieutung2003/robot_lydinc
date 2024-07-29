import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import base
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import joblib
from sklearn.svm import SVC

df_train = pd.read_csv(".\\csv\\train.csv")
df_testWM = pd.read_csv(".\\csv\\testWM.csv")
df_testMM = pd.read_csv(".\\csv\\testMM.csv")
df_testHM = pd.read_csv(".\\csv\\testHM.csv")
df_submisWM = pd.read_csv(".\\csv\\submissionWM.csv")
df_submisMM = pd.read_csv(".\\csv\\submissionMM.csv")
df_submisHM = pd.read_csv(".\\csv\\submissionHM.csv")

# Split images & labels as X & y
X_train = df_train.drop("label", axis=1)
y_train = df_train["label"]

y_testWM = df_submisWM["label"]
X_testWM = df_testWM

y_testMM = df_submisMM["label"]
X_testMM = df_testMM

y_testHM = df_submisHM["label"]
X_testHM = df_testHM

# Convert from Pandas DataFrame to Numpy Array to be able to perform reshape operations in the next step
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

X_testWM = X_testWM.to_numpy()
y_testWM = y_testWM.to_numpy()
X_testMM = X_testMM.to_numpy()
y_testMM = y_testMM.to_numpy()
X_testHM = X_testHM.to_numpy()
y_testHM = y_testHM.to_numpy()

# Reshape the labels to be a single column
y_train = y_train.reshape((-1, 1))
y_testWM = y_testWM.reshape((-1, 1))
y_testMM = y_testMM.reshape((-1, 1))
y_testHM = y_testHM.reshape((-1, 1))

def modelMPL(solver, activation, number_of_layer, hidden_layer_unit, early_stopping=True, verbose=False, max_iter=200):

    hidden_layer_sizes = ()
    for i in range(number_of_layer):
        hidden_layer_sizes += (hidden_layer_unit,)

    model = MLPClassifier(
        solver=solver,
        activation=activation,
        hidden_layer_sizes=hidden_layer_sizes,
        early_stopping=early_stopping,
        verbose=verbose,
        max_iter=max_iter,
    )
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(
        model,
        f".\\model\\modelMPL_{solver}_{activation}_{number_of_layer}_{hidden_layer_unit}.joblib",
    )

    accuracy_model = model.score(X_train, y_train)

    with open(".\\evaluate\\modelMPL.txt", "a") as f:
        f.write("====================================")
        f.write("\n")
        f.write(
            f"Solver: {solver} - Activation: {activation} - hidden layer sizes {hidden_layer_sizes}"
        )
        f.write("\n")
        f.write(f"Accuracy Model: {accuracy_model}")
        f.write("\n")

    # Evaluate the model with test data
    evaluation_model(model, X_train, y_train, "Testing in train data")
    evaluation_model(model, X_testWM, y_testWM, "Testing in WM data")
    evaluation_model(model, X_testMM, y_testMM, "Testing in MM data")
    evaluation_model(model, X_testHM, y_testHM, "Testing in HM data")

def modelSVC(C=1.0, kernel='linear', probability=True, decision_function_shape="ovr"):

    model = SVC(
        kernel=kernel,
        probability=probability,
        decision_function_shape=decision_function_shape,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, f".\\model\\modelSVM.joblib")

    accuracy_model = model.score(X_train, y_train)

    with open(".\\evaluate\\modelSVM.txt", "a") as f:
        f.write("====================================")
        f.write("\n")
        f.write(f"Accuracy Model: {accuracy_model}")
        f.write("\n")

    # Evaluate the model with test data
    evaluation_model(model, X_train, y_train, "Testing in train data")
    evaluation_model(model, X_testWM, y_testWM, "Testing in WM data")
    evaluation_model(model, X_testMM, y_testMM, "Testing in MM data")
    evaluation_model(model, X_testHM, y_testHM, "Testing in HM data")


def evaluation_model(model, X_test, y_test, message):
    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape((-1, 1))
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    evaluation_report = classification_report(y_test, y_pred)

    with open(".\\evaluate\\model.txt", "a") as f:
        f.write(f"============= {message} =============")
        f.write("\n")
        f.write(f"Testing the trained model:")
        f.write("\n")

        # Calculate confusion matrix
        f.write(f"Confusion Matrix:")
        f.write("\n")
        f.write(np.array2string(cm))
        f.write("\n")
        # Calculate accuracy
        f.write(f"Accuracy: {accuracy}")
        f.write("\n")

        # Calculate and print evaluation report
        f.write(f"Evaluation Report:")
        f.write(evaluation_report)
        f.write("\n")

# modelMPL()
# modelSVC()