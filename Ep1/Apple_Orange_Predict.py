from sklearn import tree
features = [[140,1],[130,1],[150,0],[170,0]] # [0] denote weight, [1] denote texture --> 1 for Bumpy, 0 for Smooth
lables = [0,0,1,1]  # 0 for Orange, 1 for Apple
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, lables)
print(clf.predict([[150, 0]]))
