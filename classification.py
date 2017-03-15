import numpy as np
import os
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import *
from sklearn.metrics import *
import itertools

DATA_DIR = "./student"
N = 10

dataset = np.loadtxt(os.path.join(DATA_DIR, "merged.csv"), 
                     delimiter=";")
X = dataset[:, 0:-1]
y = dataset[:, -1]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, random_state=0)

#clf = SVC(C=100., gamma = 0.001)
clf = ExtraTreesClassifier()
clf.fit(Xtrain, ytrain)

y_pred = clf.predict(Xtest)

'''
A search for 10 most important features
'''

forest = clf
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(N):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

'''
Create new classifier using 10 most important features
'''

X_new = np.zeros((X.shape[0], N), np.float32)
for i,f in enumerate(indices[:N]):
	X_new[:, i] = X[:,f]

Xtrain_new, Xtest_new, ytrain_new, ytest_new = train_test_split(X_new, y, test_size=0.4, random_state=0)

clf2 = ExtraTreesClassifier()
clf2.fit(Xtrain_new, ytrain_new)

y_p_new = clf2.predict(Xtest_new)

for y_p in [y_pred, y_p_new]:

	setname = "whole" if y_p is y_pred else str(N)
	print("\nResult for {} set\n".format(setname))

	print("Accuracy: {:.3f}".format(accuracy_score(ytest, y_p)))
	print()
	print("Confusion Matrix")
	print(confusion_matrix(ytest, y_p))
	print()
	print("Classification Report")
	print(classification_report(ytest, y_p))
