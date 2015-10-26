'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''

import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        Constructor
        '''
    	self.numBoostingIters = numBoostingIters
	self.maxTreeDepth = maxTreeDepth
	self.clf = [None] * numBoostingIters
	self.numOfClasses = -1
	self.betas = None
	self.classes = None

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
	# update the num of classes and what those classes are
	self.numOfClasses = np.unique(y).size 
	self.classes = np.unique(y)
	
	n_size,d_size = X.shape

	self.betas = np.zeros(self.numBoostingIters)

	instance_weights = np.zeros(n_size)
	#initialize each value of initial_weights to 1/n
	for weightIndex in range(0, n_size):
	    instance_weights[weightIndex] = 1.0/n_size

	for iterNum in range(0, self.numBoostingIters):
	    #train model h_t on X,y with instance weights w_t
	    curDecisionTreeClassifier = DecisionTreeClassifier(max_depth = self.maxTreeDepth)
	    self.clf[iterNum] = curDecisionTreeClassifier.fit(X, y, sample_weight=instance_weights)

	    #compute the weighted training error rate of h_t
	    error = 0
	    cur_pred = self.clf[iterNum].predict(X)
	    for i in range(0,n_size):
 		if y[i] != cur_pred[i]:
		     error += instance_weights[i]

	    #choose beta based on AdaBoost-SAMME equation
	    cur_beta = 0.5*(np.log((1-error)/error) + np.log(self.numOfClasses - 1))
	    self.betas[iterNum] = cur_beta

	    #update all instance weights
	    for i in range(0,n_size):
	        instance_weights[i] = instance_weights[i] * np.exp(-1*cur_beta*y[i]*cur_pred[i])
	
	    #normalize w_t+1 to be a distribution
	    normal_sum = np.sum(instance_weights)
	    for i in range(0,n_size):
		instance_weights[i] = instance_weights[i] / normal_sum

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
	n,d = X.shape

        beta_matrix = np.zeros((n, self.numOfClasses))
	    
        for i in range(0, self.numBoostingIters):
            y_pred = self.clf[i].predict(X)
            for k in range(0, self.numOfClasses):
	        beta_matrix[:, k] = beta_matrix[:, k] + (y_pred == self.classes[k]) * self.betas[i]
	      
        return self.classes[np.argmax(beta_matrix, axis = 1)]
