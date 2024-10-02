import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import joblib


def evaluation_model(model, X_test, y_test, message):
    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape((-1, 1))
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    evaluation_report = classification_report(y_test, y_pred)

    with open(".\\evaluate\\modelSVM.txt", "a") as f:
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


df_train = pd.read_csv(".\\csv\\databaseV4\\train.csv")
df_testWM = pd.read_csv(".\\csv\\testWM.csv")
df_testMM = pd.read_csv(".\\csv\\testMM.csv")
df_testHM = pd.read_csv(".\\csv\\testHM.csv")
df_submisWM = pd.read_csv(".\\csv\\submissionWM.csv")
df_submisMM = pd.read_csv(".\\csv\\submissionMM.csv")
df_submisHM = pd.read_csv(".\\csv\\submissionHM.csv")


X_train = df_train.drop("label", axis=1)
y_train = df_train["label"]

y_testWM = df_submisWM["label"]
X_testWM = df_testWM

y_testMM = df_submisMM["label"]
X_testMM = df_testMM

y_testHM = df_submisHM["label"]
X_testHM = df_testHM

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

X_testWM = X_testWM.to_numpy()
y_testWM = y_testWM.to_numpy()
X_testMM = X_testMM.to_numpy()
y_testMM = y_testMM.to_numpy()
X_testHM = X_testHM.to_numpy()
y_testHM = y_testHM.to_numpy()

y_train = y_train.reshape((-1, 1))
y_testWM = y_testWM.reshape((-1, 1))
y_testMM = y_testMM.reshape((-1, 1))
y_testHM = y_testHM.reshape((-1, 1))

C_regularization = 0.0
KERNEL = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
probability = True
decision_function_shape = ["ovo", "ovr"]  # one vs one and one vs rest

model = SVC(kernel="linear", probability=True, decision_function_shape="ovr")
model.fit(X_train, y_train)
joblib.dump(model, f".\\model\\databaseV4\\modelSVM.joblib")

accuracy_model = model.score(X_train, y_train)

with open(".\\evaluate\\modelSVM.txt", "a") as f:
    f.write("====================================")
    f.write("\n")
    f.write(f"Accuracy Model: {accuracy_model}")
    f.write("\n")

# Evaluate the model with test data
evaluation_model(model, X_train, y_train, "Model")
evaluation_model(model, X_testWM, y_testWM, "WM")
evaluation_model(model, X_testMM, y_testMM, "MM")
evaluation_model(model, X_testHM, y_testHM, "HM")
