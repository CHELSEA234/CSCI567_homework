"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, regularized_linear_regression,
tune_lambda, and test_error.
"""

import numpy as np
import pandas as pd

###### Q4.1 ######
def linear_regression_noreg(X, y):
  Xtrain = X
  ytrain = y
  Xtrain_transpose = Xtrain.transpose()
  inverse_matrix = np.linalg.inv(np.matmul(Xtrain_transpose, Xtrain))
  w = np.matmul(np.matmul(inverse_matrix, Xtrain_transpose), ytrain)

  return w

###### Q4.2 ######
def regularized_linear_regression(X, y, lambd):
  Xtrain = X 
  ytrain = y 

  Xtrain_transpose = Xtrain.transpose()
  indentity_matrix = np.identity(np.matmul(Xtrain_transpose, Xtrain).shape[0])
  inverse_matrix = np.linalg.inv((np.matmul(Xtrain_transpose, Xtrain)+lambd * np.array(indentity_matrix)))
  w = np.matmul(np.matmul(inverse_matrix, Xtrain_transpose), ytrain)

  return w

###### Q4.3 ######
def tune_lambda(Xtrain, ytrain, Xval, yval, lambds):
	max = 1000000000
	num = 0
	error_temp = max
	for i in range(len(lambds)):
		w = regularized_linear_regression(Xtrain, ytrain, lambds[i])
		y_pred = np.matmul(Xval, w)
		error = np.mean(np.square(yval-y_pred))
		# print (error,"comes from", lambds[i])
		if error <= error_temp:
			error_temp = error
			num = i
	bestlambda = lambds[num]

	return bestlambda

###### Q4.4 ######
def test_error(w, X, y):

	y_pred = np.matmul(X, w)
	err = np.mean(np.square(y_pred - y))

	return err


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

def data_processing():
  white = pd.read_csv('winequality-white.csv', low_memory=False, sep=';').values

  [N, d] = white.shape			# GX: the shape is 4898x12

  np.random.seed(3)
  # prepare data
  ridx = np.random.permutation(N)			# GX: randomize the order
  ntr = int(np.round(N * 0.8))
  nval = int(np.round(N * 0.1))
  ntest = N - ntr - nval

  # spliting training, validation, and test
  Xtrain = np.hstack([np.ones([ntr, 1]), white[ridx[0:ntr], 0:-1]])		# GX: notice the last line is standard/ result
  ytrain = white[ridx[0:ntr], -1]

  Xval = np.hstack([np.ones([nval, 1]), white[ridx[ntr:ntr + nval], 0:-1]])
  yval = white[ridx[ntr:ntr + nval], -1]

  Xtest = np.hstack([np.ones([ntest, 1]), white[ridx[ntr + nval:], 0:-1]])
  ytest = white[ridx[ntr + nval:], -1]
  return Xtrain, ytrain, Xval, yval, Xtest, ytest


def main():

	np.set_printoptions(precision=3)
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing()		# data's shapes are 3918, 490, 490

	# =========================Q3.1 linear_regression=================================
	w = linear_regression_noreg(Xtrain, ytrain)
	print("======== Question 3.1 Linear Regression ========")
	print("dimensionality of the model parameter is ", len(w), ".", sep="")
	print("model parameter is ", np.array_str(w))

	# =========================Q3.2 regularized linear_regression=====================
	lambd = 5.0
	wl = regularized_linear_regression(Xtrain, ytrain, lambd)
	print("\n")
	print("======== Question 3.2 Regularized Linear Regression ========")
	print("dimensionality of the model parameter is ", len(wl), sep="")
	print("lambda = ", lambd, ", model parameter is ", np.array_str(wl), sep="")

	# =========================Q3.3 tuning lambda======================
	lambds = [0, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10, 10 ** 2]
	bestlambd = tune_lambda(Xtrain, ytrain, Xval, yval, lambds)
	print("\n")
	print("======== Question 3.3 tuning lambdas ========")
	print("tuning lambda, the best lambda =  ", bestlambd, sep="")

	# =========================Q3.4 report mse on test ======================
	wbest = regularized_linear_regression(Xtrain, ytrain, bestlambd)
	mse = test_error(wbest, Xtest, ytest)
	print("\n")
	print("======== Question 3.4 report MSE ========")
	print("MSE on test is %.3f" % mse)
  
if __name__ == "__main__":
    main()
    