"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is compute_distances, predict_labels, compute_accuracy
and find_best_k.
"""

import numpy as np
import json

###### Q5.1 ######
def compute_distances(Xtrain, X):
	dists = np.zeros((X.shape[0], Xtrain.shape[0]))
	for i in range(X.shape[0]):
		for j in range(Xtrain.shape[0]):
			dist = np.sqrt(np.sum(np.square(X[i,:] - Xtrain[j,:])))
			dists[i, j] = dist

	return dists

###### Q5.2 ######
def predict_labels(k, ytrain, dists):
	ypred = np.zeros((dists.shape[0],))
	for i in range(dists.shape[0]):			# how many test samples
		test_dis_list = dists[i]
		sorted_list = sorted(test_dis_list)
		result_list = []
		for j in range(k):					# how many training samples you will consider
			value = np.where(test_dis_list == sorted_list[j])
			result_list.append(ytrain[value[0][0]])
		ypred[i] = max(set(result_list), key = result_list.count)	

	return ypred

###### Q5.3 ######
def compute_accuracy(y, ypred):
	pred = (y==ypred)
	acc = (np.sum(pred)/ypred.shape[0])*100
	return acc

###### Q5.4 ######
def find_best_k(K, ytrain, dists, yval):

	num = 0
	acc_temp = 0
	acc_list = []
	for i in range (len(K)):
		ypred = predict_labels(K[i], ytrain, dists)
		acc = compute_accuracy(yval, ypred)
		acc_list.append(acc)
		if acc>acc_temp:
			acc_temp = acc
			num = i
	best_k = K[num]
	validation_accuracy = acc_list
	return best_k, validation_accuracy


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

def data_processing(data):
	train_set, valid_set, test_set = data['train'], data['valid'], data['test']
	Xtrain = train_set[0]
	ytrain = train_set[1]
	Xval = valid_set[0]
	yval = valid_set[1]
	Xtest = test_set[0]
	ytest = test_set[1]
	
	Xtrain = np.array(Xtrain)
	Xval = np.array(Xval)
	Xtest = np.array(Xtest)
	
	ytrain = np.array(ytrain)
	yval = np.array(yval)
	ytest = np.array(ytest)
	
	return Xtrain, ytrain, Xval, yval, Xtest, ytest
	
def main():
	input_file = 'mnist_subset.json'
	output_file = 'knn_output.txt'

	with open(input_file) as json_data:
		data = json.load(json_data)
	
	#==================Compute distance matrix=======================
	K=[1, 3, 5, 7, 9]	
	
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)
	
	dists = compute_distances(Xtrain, Xval)
	
	#===============Compute validation accuracy when k=5=============
	k = 5
	ypred = predict_labels(k, ytrain, dists)
	acc = compute_accuracy(yval, ypred)
	print("The validation accuracy is", acc, "when k =", k)
	
	#==========select the best k by using validation set==============
	best_k,validation_accuracy = find_best_k(K, ytrain, dists, yval)

	
	#===============test the performance with your best k=============
	dists = compute_distances(Xtrain, Xtest)
	ypred = predict_labels(best_k, ytrain, dists)
	test_accuracy = compute_accuracy(ytest, ypred)
	
	#====================write your results to file===================
	f=open(output_file, 'w')
	for i in range(len(K)):
		f.write('%d %.3f' % (K[i], validation_accuracy[i])+'\n')
	f.write('%s %.3f' % ('test', test_accuracy))
	f.close()
	
if __name__ == "__main__":
	main()
