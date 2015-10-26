'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''

import numpy as np
from sklearn import tree

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        Constructor
        '''
        self.numBoostingIters = numBoostingIters
        self.maxTreeDepth = maxTreeDepth
        self.clf = [None] * numBoostingIters
        self.betas = None
        self.numOfClasses = None
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
		
	n,d = X.shape
	self.betas = np.zeros(self.numBoostingIters)
	instance_weights = np.ones(n) / float(n)
        
        # For T iterations
        for iterNum in range(0, self.numBoostingIters):
            #train model h_t on X,y with instance weights w_t
            self.clf[iterNum] = tree.DecisionTreeClassifier(max_depth = self.maxTreeDepth)
            self.clf[iterNum] = self.clf[iterNum].fit(X, y, sample_weight = instance_weights)
          
            #compute the weighted training error rate of h_t
	    error = 0
	    cur_pred = self.clf[iterNum].predict(X)
	    for i in range(0,n):
 		if y[i] != cur_pred[i]:
	  	    error += instance_weights[i]

	    #choose beta based on AdaBoost-SAMME equation
	    cur_beta = 0.5*(np.log((1-error)/error) + np.log(self.numOfClasses - 1))
	    self.betas[iterNum] = cur_beta
          
            # Update all instance Weights
	    for i in range(0,n):
		isCorrect = 0
		if cur_pred[i] == y[i]:
		    isCorrect = 1
		if isCorrect == 1:
		    instance_weights[i] = instance_weights[i] * np.exp(-1*cur_beta)

	    # Normalize the distribution
            normal_sum = np.sum(instance_weights)
	    for i in range(0,n):
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
            cur_y_pred = self.clf[i].predict(X)
            for j in range(0, self.numOfClasses):
                beta_matrix[:, j] = beta_matrix[:, j] + (cur_y_pred == self.classes[j]) * self.betas[i]
          
        return self.classes[np.argmax(beta_matrix, axis = 1)]
