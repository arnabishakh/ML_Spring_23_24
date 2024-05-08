## Simple Regression Exercise

import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as la


# Compute the sample mean and standard deviations for each feature (column)
# across the training examples (rows) from the data matrix X.
def mean_std(X):
  mean = np.zeros(X.shape[1])
  std = np.ones(X.shape[1])

  ## Your code here. Hint: You can use numpy to compute mean and std.
  #mean=
  #std=
  return mean, std


# Standardize the features of the examples in X by subtracting their mean and 
# dividing by their standard deviation, as provided in the parameters.
def standardize(X, mean, std):
  S = np.zeros(X.shape)

  ## Your code here.
  #S=
  return S

# Read data matrix X and labels t from text file.
def read_data(file_name):
  data=np.loadtxt(file_name)
  #  Your code here. Load data features in X and labels in t.
  #X =
  #t =
  return X, t


# Implement gradient descent algorithm to compute w = [w0, w1].
def train(X, t, eta, epochs):
  
  costs=[]
  ep=[]
  w = np.zeros(X.shape[1])
  #  YOUR CODE here. Implement gradient descent to compute w for given epochs.
  #  Use 'compute_gradient' function below to find gradient of cost function and update w each epoch.
  #  Compute and append cost and epoch number to variables costs and ep every 10 epochs.
  
  return w,ep,costs

# Compute RMSE on dataset (X, t).
def compute_rmse(X, t, w):
# YOUR CODE here:
# rmse = 
  return rmse


# Compute objective function (cost) on dataset (X, t).
def compute_cost(X, t, w):
# YOUR CODE here:
# cost = 
  return cost


# Compute gradient of the objective function (cost) on dataset (X, t).
def compute_gradient(X, t, w):
  grad = np.zeros(w.shape)
  # YOUR CODE here:

  return grad


# BONUS: Implement stochastic gradient descent algorithm to compute w = [w0, w1].
def train_SGD(X, t, eta, epochs):
#  YOUR CODE here:
  costs=[]
  ep=[]
  w = np.zeros(X.shape[1])
  #  YOUR CODE here. Implement stochastic gradient descent to compute w for given epochs. 
  #  Compute and append cost and epoch number to variables costs and ep every 10 epochs.

  return w,ep,costs


##======================= Main program =======================##
parser = argparse.ArgumentParser('Simple Regression Exercise.')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='../data/simple',
                    help='Directory for the simple houses dataset.')
FLAGS, unparsed = parser.parse_known_args()

# Read the training and test data.
Xtrain, ttrain = read_data(FLAGS.input_data_dir + "/train.txt")
Xtest, ttest = read_data(FLAGS.input_data_dir + "/test.txt")

#  YOUR CODE here: 
#  Standardize the training and test features using the mean and std computed over *training*.
#  Make sure you add the bias feature to each training and test example.
#  The bias features should be a column of ones added as the first columns of training and test examples 


# Computing parameters for each training method for eta=0.1 and 200 epochs
eta=0.1
epochs=200

w,eph,costs=train(Xtrain,ttrain,eta,epochs)
#wsgd,ephsgd,costssgd=train_SGD(Xtrain,ttrain,eta,epochs)


# Print model parameters.
print('Params GD: ', w)
#print('Params SGD: ', wsgd)

# Print cost and RMSE on training data.
print('Training RMSE: %0.2f.' % compute_rmse(Xtrain, ttrain, w))
print('Training cost: %0.2f.' % compute_cost(Xtrain, ttrain, w))

# Print cost and RMSE on test data.
print('Test RMSE: %0.2f.' % compute_rmse(Xtest, ttest, w))
print('Test cost: %0.2f.' % compute_cost(Xtest, ttest, w))

# Plotting epochs vs. cost for gradient descent methods
plt.xlabel(' epochs')
plt.ylabel('cost')
plt.yscale('log')
plt.plot( eph,costs , 'bo-', label= 'train_jw_gd')
#plt.plot( ephsgd,costssgd , 'ro-', label= 'train_j_w_sgd')
plt.legend()
plt.savefig('gd_cost_simple.png')
plt.close()

# Plotting linear approximation for each training method
plt.xlabel('Floor sizes')
plt.ylabel('House prices')
plt.plot( X,ttrain , 'bo', label= 'Training data')
plt.plot( Xt,ttest , 'g^', label= 'Test data')
plt.plot( X,w[0]+w[1]*X , 'b', label= 'GD')
#plt.plot( X,wsgd[0]+wsgd[1]*X , 'g', label= 'SGD')
plt.legend()
plt.savefig('train-test-line.png')
plt.close()
