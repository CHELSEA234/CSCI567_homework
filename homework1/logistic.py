"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is logistic_train, logistic_test, and feature_square.
"""

import json
import numpy
import sys

###### Q6.1 ######
def logistic_train(Xtrain, ytrain, w, b, step_size, max_iterations):
    """
    Inputs:
    - Xtrain: A list of num_train elements, where each element is a list of D-dimensional features. 
    - ytrain: A list of num_train labels
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector of logistic regression and initialized to be all 0s
    - b: a scalar corresponding to the bias of logistic regression model
    - step_size: step size (or learning rate) used in gradient descent
    - max_iterations: the maximum number of iterations to update parameters

    Returns:
    - learnt w and b.
    """
    # you need to fill in your solution here

    identity_list = numpy.ones(len(ytrain),)
    Xtrain_transpose = list(map(list, zip(*Xtrain)))    # (2, 200), this is for transpose
    sample_num = len(Xtrain)
    for i in range(max_iterations):
        h_x = sig(numpy.matmul(Xtrain, w) + b )                  # why is there different with Matlab in coursra
        obj = -1 * numpy.mean(ytrain*numpy.log(h_x)+(identity_list-ytrain)*numpy.log(1-h_x))
        grad_w = (numpy.matmul(Xtrain_transpose, (h_x - ytrain)))/sample_num   
        grad_b = numpy.mean(h_x - ytrain)                               # grad should have same dimension as theta
        w = w - step_size*grad_w
        b = b - step_size*grad_b



    return w, b

###### Q6.2 ######
def logistic_test(Xtest, ytest, w, b, t = 0.5):

    y_pred = sig(numpy.matmul(Xtest, w) + b)
    pred = (y_pred>0.5)
    result = (ytest == pred)
    test_acc = (numpy.sum(result)/y_pred.shape[0])*100

    return test_acc

###### Q6.3 ######
def feature_square(Xtrain, Xtest):
    """
    - Xtrain: training features, consists of num_train data points, each of which contains a D-dimensional feature
    - Xtest: testing features, consists of num_test data points, each of which contains a D-dimensional feature

    Returns:
    - element-wise squared Xtrain and Xtest.
    """
    Xtrain_s = numpy.square(Xtrain)
    Xtest_s = numpy.square(Xtest)
    return Xtrain_s, Xtest_s


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

def data_loader_toydata(dataset):

    with open(dataset, 'r') as f:
            data_set = json.load(f)
    train_set, test_set = data_set['train'], data_set['test']

    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    return Xtrain, ytrain, Xtest, ytest

def data_loader_mnist(dataset):

    with open(dataset, 'r') as f:
            data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xvalid = valid_set[0]
    yvalid = valid_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    for i, v in enumerate(ytrain):
        if v < 5:
            ytrain[i] = 0
        else:
            ytrain[i] = 1
    for i, v in enumerate(ytest):
        if v < 5:
            ytest[i] = 0
        else:
            ytest[i] = 1

    return Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest

def inti_parameter(Xtrain):
    m, n = numpy.array(Xtrain).shape
    w = numpy.array([0.0] * n)
    b = 0
    step_size = 0.1
    max_iterations = 500
    return w, b, step_size, max_iterations

def logistic_toydata1():

    Xtrain, ytrain, Xtest, ytest = data_loader_toydata(dataset = 'toydata1.json')
    w, b, step_size, max_iterations = inti_parameter(Xtrain)

    w_l, b_l = logistic_train(Xtrain, ytrain, w, b, step_size, max_iterations)
    test_acc = logistic_test(Xtest, ytest, w_l, b_l)

    return test_acc

def logistic_toydata2():

    Xtrain, ytrain, Xtest, ytest = data_loader_toydata(dataset = 'toydata2.json')
    w, b, step_size, max_iterations = inti_parameter(Xtrain)

    w_l, b_l = logistic_train(Xtrain, ytrain, w, b, step_size, max_iterations)
    test_acc = logistic_test(Xtest, ytest, w_l, b_l)

    return test_acc

def logistic_toydata2s():

    Xtrain, ytrain, Xtest, ytest = data_loader_toydata(dataset = 'toydata2.json') # squared data

    Xtrain_s, Xtest_s = feature_square(Xtrain, Xtest)
    w, b, step_size, max_iterations = inti_parameter(Xtrain_s)


    w_l, b_l = logistic_train(Xtrain_s, ytrain, w, b, step_size, max_iterations)
    test_acc = logistic_test(Xtest_s, ytest, w_l, b_l)

    return test_acc

def logistic_mnist():

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = data_loader_mnist(dataset = 'mnist_subset.json')
    w, b, step_size, max_iterations = inti_parameter(Xtrain)
    w_l, b_l = logistic_train(Xtrain, ytrain, w, b, step_size, max_iterations)
    test_acc = logistic_test(Xtest, ytest, w_l, b_l)

    return test_acc

def sig(x):
    answer = 1/(1+numpy.exp(-x))
    return answer


def main():

    test_acc = dict()

    # Xtrain, ytrain, Xtest, ytest = data_loader_toydata(dataset = 'toydata1.json')       # 200, 40, ytrain and ytest are labels
    # w, b, step_size, max_iterations = inti_parameter(Xtrain)        # learning rate is 0.1, iteration is 500

    # Xtrain = numpy.asarray(Xtrain)
    # Xtest = numpy.asarray(Xtest)
    # print (step_size)
    # print (max_iterations)
    # print (Xtrain.shape)
    # print (w.shape)
    # result = numpy.matmul(Xtrain, w)
    # print (result.shape)

    # print (numpy.asarray(b).shape)
    # result = numpy.matmul(Xtrain, w)+b
    # print (type(result))
    # print (result.shape)

    # y_pred = sig(numpy.matmul(Xtest, w) + b)
    # pred = (y_pred>0.5)

    # result = (ytest == pred)
    # print (result)

    # acc = logistic_test(Xtest, ytest, w, b)
    # print (acc)
    # w, b = logistic_train(Xtrain, ytrain, w, b, step_size, max_iterations)
    # acc = logistic_test(Xtest, ytest, w, b)
    # print (acc)

    # #=========================toydata1===========================
    
    test_acc['toydata1'] = logistic_toydata1() # results on toydata1
    print('toydata1, test acc = %.4f \n' % (test_acc['toydata1']))

    # #=========================toydata2===========================
    
    test_acc['toydata2'] = logistic_toydata2() # results on toydata2
    
    print('toydata2, test acc = %.4f \n' % (test_acc['toydata2']))
    test_acc['toydata2s'] = logistic_toydata2s() # results on toydata2 but with squared feature 
    
    print('toydata2 w/ squared feature, test acc = %.4f \n' % (test_acc['toydata2s']))
    
    # #=========================mnist_subset=======================

    test_acc['mnist'] = logistic_mnist()    # results on mnist
    print('mnist test acc = %.4f \n' % (test_acc['mnist']))

    
    with open('logistic.json', 'w') as f_json:
        json.dump([test_acc], f_json)

if __name__ == "__main__":
    main()
