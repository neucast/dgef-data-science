import os
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import \
    MLPClassifier  # Source https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
from sklearn.tree import \
    DecisionTreeClassifier, \
    DecisionTreeRegressor  # Source: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC  # Source: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

# Configure.
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# inputPath to the CSV file.
inputPath = os.path.join(os.path.expanduser("~"), "development", "dgef-data-science",
                         "exercises", "module3_part_1",
                         "data", "evaluacion-modulo-3-clasificacion.csv")

# Prints the absolute inputPath to the CSV file.
print("The input CSV file is: ", inputPath)

# Reads the CSV data file.
sourceMatrix = pd.read_csv(inputPath, dtype='str', encoding="ISO-8859-1")
sourceMatrix[["Y", "X1", "X2"]] = sourceMatrix[["Y", "X1", "X2"]].astype(float)
print(sourceMatrix.head())


# Mean square error function.
def meanSquareErrorFunction(v1, v2):
    accumulator, columns = 0, len(v1)

    for i in range(columns):
        accumulator = accumulator + (v1[1] - v2[1]) ** 2
    return (accumulator ** 0.5) / columns


# Data sample.
test_sample_size = 0.2
X_split_training_set, X_split_testing_set, Y_split_training_set, Y_split_testing_set = train_test_split(
    sourceMatrix[["X1", "X2"]],
    sourceMatrix[["Y"]],
    test_size=test_sample_size,
    random_state=1)

# Transforming data as a Numpy array.
X_training_set = np.asarray(X_split_training_set[["X1", "X2"]].copy(deep=True).reset_index(drop=True))
X_testing_set = np.asarray(X_split_testing_set[["X1", "X2"]].copy(deep=True).reset_index(drop=True))
Y_training_set = np.asarray(Y_split_training_set[["Y"]].copy(deep=True).reset_index(drop=True))
Y_testing_set = np.asarray(Y_split_testing_set[["Y"]].copy(deep=True).reset_index(drop=True))

# Calculation using python scikit-learn defined functions.
# scikit-learn method comparison.

# ---------------------------------- Logistic Regression ----------------------------------
print("----------------------------- Logistic Regression -----------------------------")
# Logistic Regression.
logisticRegression = LogisticRegression()
logisticRegression.fit(X_training_set, Y_training_set)
Y_LogisticRegression_predicted_set = logisticRegression.predict(X_testing_set)
logisticRegressionMeanSquareError = meanSquareErrorFunction(Y_LogisticRegression_predicted_set, Y_testing_set)
print("Logistic Regression mean square error = ", logisticRegressionMeanSquareError)

# Logistic Regression Accuracy score.
logisticRegressionAccuracyScore = accuracy_score(pd.DataFrame(Y_LogisticRegression_predicted_set), Y_testing_set)
print("Logistic Regression Accuracy score:")
print(logisticRegressionAccuracyScore)

# Logistic Regression confusion matrix.
print("Logistic Regression confusion matrix:")
print(confusion_matrix(Y_testing_set, Y_LogisticRegression_predicted_set))

# Logistic Regression ROC curve compute.
Y_LogisticRegression_predicted_set_prob = logisticRegression.predict_proba(X_testing_set)[:, 1]
X_LogisticRegression_rate, Y_LogisticRegression_rate, _ = roc_curve(Y_testing_set,
                                                                    Y_LogisticRegression_predicted_set_prob)
fig = plt.figure(figsize=(5., 5.))
ax = fig.add_subplot(1, 1, 1)
plt.plot(X_LogisticRegression_rate, Y_LogisticRegression_rate, color="blue", linestyle="-",
         label="Logistic Regression ROC curve")
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='best', ncol=2)
plt.show()

# Logistic Regression area under the ROC curve.
print("Logistic Regression area under the ROC curve is",
      roc_auc_score(Y_testing_set, Y_LogisticRegression_predicted_set_prob))
# ------------------------------------------------------------------------------------------------------

# ---------------------------------- Support Vector Machines ----------------------------------
print("----------------------------- Support Vector Machines -----------------------------")
# Support Vector Machines.
# Linear SVC
print("------- Linear SVC -------")
linearSVC = LinearSVC(C=0.1, max_iter=500, fit_intercept=True)
linearSVC.fit(X_training_set, Y_training_set)
Y_LinearSVC_predicted_set = linearSVC.predict(X_testing_set)
linearSVCMeanSquareError = meanSquareErrorFunction(Y_LinearSVC_predicted_set, Y_testing_set)
print("Linear SVC mean square error = ", linearSVCMeanSquareError)

# Linear SVC Accuracy score.
linearSVCAccuracyScore = accuracy_score(pd.DataFrame(Y_LinearSVC_predicted_set), Y_testing_set)
print("Linear SVC Accuracy score:")
print(linearSVCAccuracyScore)

# Linear SVC confusion matrix.
print("Linear SVC confusion matrix:")
print(confusion_matrix(Y_testing_set, Y_LinearSVC_predicted_set))

# Linear SVC ROC curve compute.
Y_LinearSVC_predicted_set_prob = linearSVC._predict_proba_lr(X_testing_set)[:, 1]
X_LinearSVC_rate, Y_LinearSVC_rate, _ = roc_curve(Y_testing_set,
                                                  Y_LinearSVC_predicted_set_prob)
fig = plt.figure(figsize=(5., 5.))
ax = fig.add_subplot(1, 1, 1)
plt.plot(X_LinearSVC_rate, Y_LinearSVC_rate, color="blue", linestyle="-",
         label="Linear SVC ROC curve")
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='best', ncol=2)
plt.show()

# Linear SVC area under the ROC curve.
print("Linear SVC area under the ROC curve is",
      roc_auc_score(Y_testing_set, Y_LinearSVC_predicted_set_prob))

# SVC
print("------- SVC Linear -------")
svcLinear = SVC(kernel='linear', C=0.1, max_iter=500, probability=True)
svcLinear.fit(X_training_set, Y_training_set)
Y_SVCLinear_predicted_set = svcLinear.predict(X_testing_set)
svcLinearMeanSquareError = meanSquareErrorFunction(Y_SVCLinear_predicted_set, Y_testing_set)
print("SVC linear mean square error = ", svcLinearMeanSquareError)

# SVC linear Accuracy score.
svcLinearAccuracyScore = accuracy_score(pd.DataFrame(Y_SVCLinear_predicted_set), Y_testing_set)
print("SVC linear Accuracy score:")
print(svcLinearAccuracyScore)

# SVC linear confusion matrix.
print("SVC linear confusion matrix:")
print(confusion_matrix(Y_testing_set, Y_SVCLinear_predicted_set))

# SVC linear ROC curve compute.
Y_SVCLinear_predicted_set_prob = svcLinear.predict_proba(X_testing_set)[:, 1]
X_SVCLinear_rate, Y_SVCLinear_rate, _ = roc_curve(Y_testing_set,
                                                  Y_SVCLinear_predicted_set_prob)
fig = plt.figure(figsize=(5., 5.))
ax = fig.add_subplot(1, 1, 1)
plt.plot(X_SVCLinear_rate, Y_SVCLinear_rate, color="blue", linestyle="-",
         label="SVC linear ROC curve")
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='best', ncol=2)
plt.show()

# SVC linear area under the ROC curve.
print("SVC linear area under the ROC curve is",
      roc_auc_score(Y_testing_set, Y_SVCLinear_predicted_set_prob))

print("------- SVC polynomic -------")
svcPolynomic = SVC(kernel='poly', C=0.1, max_iter=500, probability=True)
svcPolynomic.fit(X_training_set, Y_training_set)
Y_SVCPolynomic_predicted_set = svcPolynomic.predict(X_testing_set)
svcPolynomicMeanSquareError = meanSquareErrorFunction(Y_SVCPolynomic_predicted_set, Y_testing_set)
print("SVC polynomic mean square error = ", svcPolynomicMeanSquareError)

# SVC polynomic Accuracy score.
svcPolynomicAccuracyScore = accuracy_score(pd.DataFrame(Y_SVCPolynomic_predicted_set), Y_testing_set)
print("SVC polynomic Accuracy score:")
print(svcPolynomicAccuracyScore)

# SVC polynomic confusion matrix.
print("SVC polynomic confusion matrix:")
print(confusion_matrix(Y_testing_set, Y_SVCPolynomic_predicted_set))

# SVC polynomic ROC curve compute.
Y_SVCPolinomic_predicted_set_prob = svcPolynomic.predict_proba(X_testing_set)[:, 1]
X_SVCPolynomic_rate, Y_SVCPolynomic_rate, _ = roc_curve(Y_testing_set,
                                                        Y_SVCPolinomic_predicted_set_prob)
fig = plt.figure(figsize=(5., 5.))
ax = fig.add_subplot(1, 1, 1)
plt.plot(X_SVCPolynomic_rate, Y_SVCPolynomic_rate, color="blue", linestyle="-",
         label="SVC polynomic ROC curve")
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='best', ncol=2)
plt.show()

# SVC polynomic area under the ROC curve.
print("SVC polynomic area under the ROC curve is",
      roc_auc_score(Y_testing_set, Y_SVCPolinomic_predicted_set_prob))
# ------------------------------------------------------------------------------------------------------

# ---------------------------------- Decision Tree Classification ----------------------------------
print("----------------------------- Decision Tree Classification -----------------------------")
# Decision tree classification procedure using the entropy function.
decisionTreeClassifier = DecisionTreeClassifier(criterion='entropy')
decisionTreeClassifier.fit(X_training_set, Y_training_set)
Y_Decision_Tree_Classifier_predicted_set = decisionTreeClassifier.predict(X_testing_set)
DecisionTreeClassifierMeanSquareError = meanSquareErrorFunction(Y_Decision_Tree_Classifier_predicted_set, Y_testing_set)
print("Decision Tree Classifier (DTC) with entropy criterion mean square error = ",
      DecisionTreeClassifierMeanSquareError)

# Decision tree classification Accuracy score.
decisionTreeClassifierAccuracyScore = accuracy_score(pd.DataFrame(Y_Decision_Tree_Classifier_predicted_set),
                                                     Y_testing_set)
print("DTC Binary tree Accuracy score:")
print(decisionTreeClassifierAccuracyScore)

# Decision tree classifier confusion matrix.
print("DTC Binary tree confusion matrix: ")
print(confusion_matrix(Y_testing_set, Y_Decision_Tree_Classifier_predicted_set))

# Decision tree classifier ROC curve compute.
Y_DecisionTreeClassifier_predicted_set_prob = decisionTreeClassifier.predict_proba(X_testing_set)[:, 1]
X_DecisionTreeClassifier_rate, Y_DecisionTreeClassifier_rate, _ = roc_curve(Y_testing_set,
                                                                            Y_DecisionTreeClassifier_predicted_set_prob)
fig = plt.figure(figsize=(5., 5.))
ax = fig.add_subplot(1, 1, 1)
plt.plot(X_DecisionTreeClassifier_rate, Y_DecisionTreeClassifier_rate, color="blue", linestyle="-",
         label="DTC Binary tree ROC curve")
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='best', ncol=2)
plt.show()

# Decision tree classifier area under the ROC curve.
print("DTC Binary tree area under the ROC curve is",
      roc_auc_score(Y_testing_set, Y_DecisionTreeClassifier_predicted_set_prob))
# ------------------------------------------------------------------------------------------------------

# ---------------------------------- Binary trees with simplified bagging (BTSB) ----------------------------------
print("----------------------------- Binary trees with simplified bagging (BTSB) -----------------------------")
# Number of points.
points_number = 200

# Bagging definition.
n_estimators = 10
bag = np.empty((n_estimators), dtype=object)  # Empty array.

# Bootstraping (note that sampling is made on the training data)
for i in range(n_estimators):
    randomIndex = np.random.choice(range(0, len(X_training_set)), size=int(len(X_training_set)),
                                   replace=True)  # Array with random values got from X_training_set array with range(0,len(X_training_set)).
    np.unique(randomIndex)  # Function unique returns non repeated values ordered.
    X_bootstrap_training_set = X_training_set[
        np.unique(randomIndex)]  # Rows (vectors) are selected from X_training_set given by randomIndex.
    Y_bootstrap_training_set = Y_training_set[
        np.unique(
            randomIndex)]  # Rows (vectors) are selected from Y_training_set given by randomIndex.
    bag[i] = DecisionTreeRegressor(random_state=1)
    bag[i].fit(X_bootstrap_training_set, Y_bootstrap_training_set)

# Bagging prediction (given each sample adjustment the forecast is made on the data testing set).
Y_BTSB_predicted_set = np.zeros(len(Y_testing_set))
# Y_BTSB_predicted_set_prob = np.zeros(len(Y_testing_set))
for i in range(n_estimators):
    Y_BTSB_predicted_set = Y_BTSB_predicted_set + bag[i].predict(X_testing_set)
    # Y_BTSB_predicted_set_prob = Y_BTSB_predicted_set_prob + bag[i]._._predict_proba(X_testing_set)[:, 1]
Y_BTSB_predicted_set = Y_BTSB_predicted_set / n_estimators
BTSBMeanSquareError = meanSquareErrorFunction(Y_BTSB_predicted_set, Y_testing_set)
print("Binary trees with simplified bagging (BTSB) mean square error = ", BTSBMeanSquareError)

# Binary trees with simplified bagging (BTSB) Accuracy score.
# BTSBAccuracyScore = accuracy_score(pd.DataFrame(Y_BTSB_predicted_set),
#                                    Y_testing_set)
# print("Binary trees with simplified bagging (BTSB) Accuracy score:")
# print(BTSBAccuracyScore)

# Binary trees with simplified bagging (BTSB) confusion matrix.
# print("Binary trees with simplified bagging (BTSB) confusion matrix: ")
# print(confusion_matrix(Y_testing_set, Y_BTSB_predicted_set))

# Binary trees with simplified bagging (BTSB) ROC curve compute.
# X_BTSB_rate, Y_BTSB_rate, _ = roc_curve(Y_testing_set, Y_BTSB_predicted_set_prob)
# fig = plt.figure(figsize=(5., 5.))
# ax = fig.add_subplot(1, 1, 1)
# plt.plot(X_BTSB_rate, Y_BTSB_rate, color="blue", linestyle="-",
#          label="Binary trees with simplified bagging (BTSB) ROC curve")
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.legend(loc='best', ncol=2)
# plt.show()

# Decision tree classifier area under the ROC curve.
# print("Binary trees with simplified bagging (BTSB) area under the ROC curve is",
#       roc_auc_score(Y_testing_set, Y_BTSB_predicted_set_prob))


# yhatAB_aux = np.copy(Y_training_set).tolist()
# y_aux = np.copy(Y_training_set).tolist()
# yhatAB_aux.extend(Y_BTSB_predicted_set)
# y_aux.extend(Y_testing_set)
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(1, 1, 1)
# plt.plot(range(1, len(yhatAB_aux) + 1), yhatAB_aux, color="black", linewidth=1.5, label="Árbol binario")
# plt.plot(range(1, len(y_aux) + 1), y_aux, color="red", ls="--", linewidth=1.0, label="Serie verdadera")
# plt.axvline(x=points_number - test_sample_size * points_number, color="green", linewidth=3)
# plt.legend(loc="best")
# plt.show()

# Plot.
# yhatAB_aux = np.copy(y_train).tolist()
# y_aux = np.copy(y_train).tolist()
# yhatAB_aux.extend(yhatbag)
# y_aux.extend(y_test)
# fig = plt.figure(figsize=(10,5))
# ax = fig.add_subplot(1,1,1)
# plt.plot(range(1,len(yhatAB_aux)+1),yhatAB_aux,color="black",linewidth=1.5,label="Árbol binario")
# plt.plot(range(1,len(y_aux)+1),y_aux,color="red",ls="--",linewidth=1.0,label="Serie verdadera")
# plt.axvline(x=n_points-tam_test*n_points,color="green",linewidth=3)
# plt.legend(loc="best")
# plt.show()
# ------------------------------------------------------------------------------------------------------

# ---------------------------------- Random Forest ----------------------------------
print("----------------------------- Random Forest -----------------------------")
# Random Forest.
randomForestRegressor = RandomForestRegressor(n_estimators=10, max_features=1, random_state=1)
randomForestRegressor.fit(X_training_set, Y_training_set)
Y_Random_Forest_predicted_set = randomForestRegressor.predict(X_testing_set)
randomForestMeanSquareError = meanSquareErrorFunction(Y_Random_Forest_predicted_set, Y_testing_set)
print("Random forest mean square error = ",
      randomForestMeanSquareError)

# Random forest Accuracy score.
# randomForestAccuracyScore = accuracy_score(pd.DataFrame(Y_Random_Forest_predicted_set),
#                                            Y_testing_set)
# print("Random forest Accuracy score:")
# print(randomForestAccuracyScore)

# Random forest confusion matrix.
# print("Random forest confusion matrix: ")
# print(confusion_matrix(Y_testing_set, Y_Random_Forest_predicted_set))

# Random forest ROC curve compute.
# Y_RandomForest_predicted_set_prob = randomForestRegressor.predict_proba(X_testing_set)[:, 1]
# X_RandomForest_rate, Y_RandomForest_rate, _ = roc_curve(Y_testing_set,
#                                                         Y_RandomForest_predicted_set_prob)
# fig = plt.figure(figsize=(5., 5.))
# ax = fig.add_subplot(1, 1, 1)
# plt.plot(X_RandomForest_rate, Y_RandomForest_rate, color="blue", linestyle="-",
#          label="Random forest ROC curve")
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.legend(loc='best', ncol=2)
# plt.show()

# Random forest area under the ROC curve.
# print("Random forest area under the ROC curve is",
#       roc_auc_score(Y_testing_set, Y_RandomForest_predicted_set_prob))

# Plot.
# yhatBA_aux = np.copy(y_train).tolist()
# y_aux = np.copy(y_train).tolist()
# yhatBA_aux.extend(yhatBA)
# y_aux.extend(y_test)
# fig = plt.figure(figsize=(10,5))
# ax = fig.add_subplot(1,1,1)
# plt.plot(range(1,len(yhatBA_aux)+1),yhatBA_aux,color="black",linewidth=1.5,label="Bosque aleatorio")
# plt.plot(range(1,len(y_aux)+1),y_aux,color="red",ls="--",linewidth=1.0,label="Serie verdadera")
# plt.axvline(x=n_points-tam_test*n_points,color="green",linewidth=3)
# plt.legend(loc="best")
# plt.show()
# ------------------------------------------------------------------------------------------------------

# ---------------------------------- Neural network ----------------------------------
print("----------------------------- Neural network -----------------------------")
# Neural network with "tanh" activation function.
neuralNetworkModelTanh = MLPClassifier(activation='tanh', alpha=0.001, solver='lbfgs',
                                       hidden_layer_sizes=(4, 1), max_iter=10000)
neuralNetworkModelTanh.fit(X_training_set, Y_training_set)
Y_Neural_Network_predicted_set = neuralNetworkModelTanh.predict(X_testing_set)
neuralNetworkMeanSquareError = meanSquareErrorFunction(Y_Neural_Network_predicted_set, Y_testing_set)
print("Neural Network with tanh activation function mean square error = ",
      neuralNetworkMeanSquareError)

# Neural network Accuracy score.
neuralNetworkAccuracyScore = accuracy_score(pd.DataFrame(Y_Neural_Network_predicted_set),
                                            Y_testing_set)
print("Neural network Accuracy score:")
print(neuralNetworkAccuracyScore)

# Neural network confusion matrix.
print("Neural network confusion matrix: ")
print(confusion_matrix(Y_testing_set, Y_Neural_Network_predicted_set))

# Neural network ROC curve compute.
Y_NeuralNetwork_predicted_set_prob = neuralNetworkModelTanh.predict_proba(X_testing_set)[:, 1]
X_NeuralNetwork_rate, Y_NeuralNetwork_rate, _ = roc_curve(Y_testing_set,
                                                          Y_NeuralNetwork_predicted_set_prob)
fig = plt.figure(figsize=(5., 5.))
ax = fig.add_subplot(1, 1, 1)
plt.plot(X_NeuralNetwork_rate, Y_NeuralNetwork_rate, color="blue", linestyle="-",
         label="Neural network ROC curve")
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='best', ncol=2)
plt.show()

# Neural network area under the ROC curve.
print("Neural network area under the ROC curve is",
      roc_auc_score(Y_testing_set, Y_NeuralNetwork_predicted_set_prob))
# ------------------------------------------------------------------------------------------------------
