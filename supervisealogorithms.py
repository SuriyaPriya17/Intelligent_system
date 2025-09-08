from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
X, y = load_breast_cancer(return_X_y=True)
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

clf = LogisticRegression(max_iter=10000, random_state=0)
clf.fit(X_train, y_train)

acc = accuracy_score(y_test, clf.predict(X_test)) * 100
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)
print(f"Logistic Regression model accuracy: {acc:.2f}%")

from sklearn.svm import SVC
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

svm_clf = SVC(max_iter=10000,C=1.0, random_state=0)
svm_clf.fit(X_train, y_train)

y_pred = svm_clf.predict(X_test)

acc = accuracy_score(y_test, y_pred) * 100

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

print(f"SVM model accuracy: {acc:.2f}%")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)


acc = accuracy_score(y_test, y_pred) * 100

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)
print(f"KNN model accuracy: {acc:.2f}%")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

dt_clf = DecisionTreeClassifier(random_state=42,Information_gain=True)

dt_clf.fit(X_train, y_train)

y_pred = dt_clf.predict(X_test)

acc = accuracy_score(y_test, y_pred) * 100

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

print(f"Decision Tree model accuracy: {acc:.2f}%")
