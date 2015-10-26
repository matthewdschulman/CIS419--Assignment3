'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''

import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3, numOfClasses=9):
        '''
        Constructor
        '''
    	self.numBoostingIters = numBoostingIters
	self.maxTreeDepth = maxTreeDepth
	self.clf = None
	self.numOfClasses = numOfClasses

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
	self.clf = DecisionTreeClassifier(max_depth = self.maxTreeDepth)
	n,d = X.shape
	instance_weights = np.zeros(n)
	#initialize each value of initial_weights to 1/n
	for weightIndex in range(0, n):
	    instance_weights[weightIndex] = 1.0/n

	print instance_weights
	for iterNum in range(0, self.numBoostingIters):
	    #train model h_t on X,y with instance weights w_t
	    self.clf.fit(X, y, sample_weight=instance_weights)

	    #compute the weighted training error rate of h_t
	    error = 0
	    cur_pred = self.clf.predict(X)
	    for i in range(0,n):
 		if y[i] != cur_pred[i]:
		     error += instance_weights[i]

	    #choose beta based on AdaBoost-SAMME equation
	    cur_beta = 0.5*(np.log((1-error)/error) + np.log(self.numOfClasses - 1))

	    #update all instance weights

	    #normalize w_t+1 to be a distribution

	# set H(x) to the hypothesis

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
	return self.clf.predict(X)
