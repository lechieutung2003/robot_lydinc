import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import joblib
import os

models = ['ArcFace', 'Facenet', 'Facenet512', 'GhostFaceNet', 'SFace']
# models = ['Facenet512']
for mod in models:
    csv_dir = f"D:\\Artificial Intelligence\\git clone\\robot_lydinc\\csv\\identifyV2\\{mod}"
    model_dir = "D:\\Artificial Intelligence\\git clone\\robot_lydinc\\model"
    evaluate_dir = "D:\\Artificial Intelligence\\git clone\\robot_lydinc\\evaluate\\SVM"

    # Construct the full file paths
    train_file = os.path.join(csv_dir, "train.csv")
    testWM_file = os.path.join(csv_dir, "testWM.csv")
    testMM_file = os.path.join(csv_dir, "testMM.csv")
    testHM_file = os.path.join(csv_dir, "testHM.csv")
    submisWM_file = os.path.join(csv_dir, "submissionWM.csv")
    submisMM_file = os.path.join(csv_dir, "submissionMM.csv")
    submisHM_file = os.path.join(csv_dir, "submissionHM.csv")

    df_train = pd.read_csv(train_file)
    df_testWM = pd.read_csv(testWM_file)
    df_testMM = pd.read_csv(testMM_file)
    df_testHM = pd.read_csv(testHM_file)
    df_submisWM = pd.read_csv(submisWM_file)
    df_submisMM = pd.read_csv(submisMM_file)
    df_submisHM = pd.read_csv(submisHM_file)


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

    C_regularization = 1.0
    KERNEL = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
    probability = True
    decision_function_shape = ["ovo", "ovr"]  # one vs one and one vs rest

    model = SVC(kernel="linear", probability=True, decision_function_shape="ovr")
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(model_dir, f"modelSVM_{mod}.joblib"))

    accuracy_model = model.score(X_train, y_train)

    with open(os.path.join(evaluate_dir, f"modelSVM_{mod}.txt"), "a") as f:
        f.write("====================================")
        f.write("\n")
        f.write(f"Accuracy Model: {accuracy_model}")
        f.write("\n")


    def evaluation_model(model, X_test, y_test, message):
        y_pred = model.predict(X_test)
        y_pred = y_pred.reshape((-1, 1))
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        evaluation_report = classification_report(y_test, y_pred)

        with open(os.path.join(evaluate_dir, f"modelSVM_{mod}.txt"), "a") as f:
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

    # Evaluate the model with test data
    evaluation_model(model, X_train, y_train, "Model")
    evaluation_model(model, X_testWM, y_testWM, "WM")
    evaluation_model(model, X_testMM, y_testMM, "MM")
    evaluation_model(model, X_testHM, y_testHM, "HM")