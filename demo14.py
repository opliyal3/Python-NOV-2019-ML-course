from sklearn import datasets, svm, model_selection

iris = datasets.load_iris()

svc = svm.SVC()
scores = model_selection.cross_val_score(svc, iris.data, iris.target, cv=5)
print(scores)
print("Accuracy: ", scores.mean())