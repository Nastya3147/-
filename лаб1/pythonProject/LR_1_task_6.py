import numpy as np
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utilities import visualize_classifier
input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
classifierNB = GaussianNB()
classifier  =SVC(kernel='linear', C=0.4)
classifier.fit(X, y)
classifierNB.fit(X, y)
y_pred = classifier.predict(X)
y_predNB = classifierNB.predict(X)
accuracy = 100.0 * (y == y_predNB).sum() / X.shape[0]
print("Accuracy of  classifier =", round(accuracy_score(y, y_pred)*100,2), "%")
print("Accuracy of  classifierNB =", round(accuracy,
2), "%")
print("*******************************")
visualize_classifier(classifier, X, y)
visualize_classifier(classifierNB, X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
classifier_new = GaussianNB()
svclassifier = SVC(kernel='linear', C=0.4)
classifier_new.fit(X_train, y_train)
y_test_predNB = classifier_new.predict(X_test)
svclassifier.fit(X_train, y_train)
y_test_pred = svclassifier.predict(X_test)
print("Accuracy of the new classifier =", round(accuracy_score(y_test, y_test_pred)*100, 2),"%")
accuracy_values = cross_val_score(svclassifier,X, y, scoring='accuracy', cv=3)
print("Accuracy: " + str(round(100 * accuracy_values.mean(), 2)) + "%")
precision_values = cross_val_score(svclassifier, X, y, scoring='precision_weighted', cv=3)
print("Precision: " + str(round(100 * precision_values.mean(), 2)) + "%")
recall_values = cross_val_score(svclassifier, X, y, scoring='recall_weighted', cv=3)
print("Recall: " + str(round(100 * recall_values.mean(), 2)) + "%")
f1_values = cross_val_score(svclassifier,X, y, scoring='f1_weighted', cv=3)
print("F1: " + str(round(100 * f1_values.mean(), 2)) + "%")
print("*******************************")
num_folds = 3
accuracy = 100.0 * (y_test == y_test_predNB).sum() / X_test.shape[0]
print("Accuracy of the new classifierNB =", round(accuracy, 2),"%")
accuracy_values = cross_val_score(classifierNB,X, y, scoring='accuracy', cv=num_folds)
print("AccuracyNB: " + str(round(100 * accuracy_values.mean(), 2)) + "%")
precision_values = cross_val_score(classifierNB, X, y, scoring='precision_weighted', cv=num_folds)
print("PrecisionNB: " + str(round(100 * precision_values.mean(), 2)) + "%")
recall_values = cross_val_score(classifierNB, X, y, scoring='recall_weighted', cv=num_folds)
print("RecallNB: " + str(round(100 * recall_values.mean(), 2)) + "%")
f1_values = cross_val_score(classifierNB,X, y, scoring='f1_weighted', cv=num_folds)
print("F1NB: " + str(round(100 * f1_values.mean(), 2)) + "%")
# Візуалізація роботи класифікатора
visualize_classifier(svclassifier, X_test, y_test)
visualize_classifier(classifier_new, X_test, y_test)
