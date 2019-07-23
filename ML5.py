#Support Vector Machine from SKLearn Demo

import numpy as np
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle


df = pd.read_csv('breast-cancer-wisconsin(1).data')
df.replace('?',-99999, inplace=True)
df.drop(['id'],1, inplace=True)

x = np.array(df.drop(['class'],1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = .2)

clf = svm.SVC()
clf.fit(x_train,y_train)
# with open('kneighbors.pickle','wb') as f:
#     pickle.dump(clf,f)
#
# pickle_in = open('kneighbors.pickle','rb')
# clf = pickle_in

accuracy = clf.score(x_test,y_test)
print(accuracy)

test_points = np.array([[6,2,1,1,1,1,3,4,5],[6,2,1,1,1,1,4,7,7]])
test_points = test_points.reshape(len(test_points),-1)
prediction = clf.predict(test_points)
print(prediction)