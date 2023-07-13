import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree


iris = load_iris()
test_idx = [0, 50,100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))


# visualize code
# IT not working
# from sklearn.externals.six import String10
# import pydot
# dot_data = String10()
# tree.export_graphviz(clf, 
#                     out_file=dot_data, 
#                     feature_names=iris.feature_names, 
#                     class_names=iris.target_names, 
#                     filled=True, rounded=True, 
#                     impurity=False)

# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("iris.pdf")

# Visualizing a Decision Tree using a Classifier (discrete variables, labels, etc.)
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree

# Prepare the data data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Fit the classifier with default hyper-parameters
clf = DecisionTreeClassifier()
model = clf.fit(X, y)


# 1
text_representation = tree.export_text(clf)
print(text_representation)

# if you want to save the tree...
with open("decistion_tree.log", "w") as fout:
    fout.write(text_representation)


print(test_data[0], test_target[0])
print(iris.feature_names, iris.target_names)