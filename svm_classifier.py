#!/usr/local/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

#submission_test = pd.read_csv("test.csv") # 28,000 examples with no labels

'''
parameters: train_file (String): input file which contains training examples and labels
return: (X_train, X_test, y_train, y_test)
'''
def get_training_set(train_file):
	train = pd.read_csv("train.csv") # 42,000 examples with 1st columns the labels
	X = train.iloc[:5000,1:]
	y = train.iloc[:5000,:1]
	del(train)
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

	assert X_train.shape == (4000,784) and X_test.shape == (1000,784)
	assert y_train.shape == (4000,1) and y_train.keys().values.shape == (1,) and y_train.keys().values[0] == 'label' 
	assert y_test.shape == (1000,1) and y_test.keys().values.shape == (1,) and y_test.keys().values[0] == 'label'
	
	X_train = X_train/255.0
	X_test = X_test/255.0
	return (X_train, X_test, y_train, y_test)


(X_train, X_test, y_train, y_test) = get_training_set("train.csv")
#import pdb;pdb.set_trace()
clf = svm.SVC()
clf.fit(X_train, y_train.values.ravel())# since X, y should be array-like, so pandas dataframe or numpy array both can be passed as parameters here
# y_predict = clf.predict(X_test)
# result = 1-(np.count_nonzero(y_predict - y_test.values.ravel())/1000.0)
clf_score = clf.score(X_test,y_test.values.ravel())
print(clf_score)
