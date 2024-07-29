import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import base
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import joblib

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


# Train model with different hyperparameters
def modelMPL():
    SOLVER = ["sgd", "lbfgs", "adam"]
    ACTIVATION = ["identity", "tanh", "logistic", "relu"]
    NUMBER_OF_LAYERS = [1,2,3]
    HIDDEN_LAYER_UNITS = [32, 64, 128, 256, 512, 1024]

    for solver in SOLVER:
        for activation in ACTIVATION:
            for layer in NUMBER_OF_LAYERS:
                for layer_inits in HIDDEN_LAYER_UNITS:

                    hidden_layer_sizes = ()
                    for i in range(layer):
                        hidden_layer_sizes += (layer_inits,)

                    model = MLPClassifier(
                        solver=solver,
                        activation=activation,
                        hidden_layer_sizes=hidden_layer_sizes,
                        early_stopping=True,
                        verbose=False,
                        max_iter=200,
                    )
                    
                    model.fit(X_train, y_train)
                                
                    accuracy_model = model.score(X_train, y_train)
                    
                    with open(".\\evaluate\\model.txt", "a") as f:
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

def modelSVC():
    C = [1.0]
    KERNEL = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
    PROBABILITIES = [True, False]
    DECISION_FUNCTION_SHAPES = ["ovo", "ovr"]  # one vs one and one vs rest

    for c in C:
        for kernel in KERNEL:
            for probability in PROBABILITIES:
                for decision_function_shape in DECISION_FUNCTION_SHAPES:
                    model = SVC(
                        C = c,
                        kernel=kernel,
                        probability=probability,
                        decision_function_shape=decision_function_shape,
                    )

                    model.fit(X_train, y_train)

                    accuracy_model = model.score(X_train, y_train)

                    with open(".\\evaluate\\modelSVM.txt", "a") as f:
                        f.write("====================================")
                        f.write("\n")
                        f.write(
                            f"C: {c} - kernel: {kernel} - probability: {probability} - decision_function_shape: {decision_function_shape}"
                        )
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

modelMPL()
