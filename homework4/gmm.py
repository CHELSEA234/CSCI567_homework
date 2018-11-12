import json
import random
import numpy as np

# input arguments are ndarray, cur_cov is size of (2, 2)
def guassian_calculator(cur_point, cur_mu, cur_cov):
    inverse_cov = np.linalg.inv(cur_cov)
    diff = cur_point - cur_mu
    exp_value = -1 / 2 * (np.matmul(np.matmul(diff, inverse_cov), np.transpose(diff)))
    
    deter = np.linalg.det(cur_cov)
    coeff = 1 / (2 * np.pi* np.sqrt(deter))
    result = coeff * np.exp(exp_value)

    return result

def resp_generator(pro, pi):
    resp = []
    resp_sum = 0
    
    for i in range(len(pi)):
        cur_resp = (pi[i] * pro[i])
        resp_sum = resp_sum + cur_resp
        resp.append(cur_resp)

    resp = resp / resp_sum
    resp = resp.tolist()

    return resp

def gmm_clustering(X, K):
    """
    Train GMM with EM for clustering.

    Inputs:
    - X: A list of data points in 2d space, each elements is a list of 2
    - K: A int, the number of total cluster centers

    Returns:
    - mu: A list of all K means in GMM, each elements is a list of 2
    - cov: A list of all K covariance in GMM, each elements is a list of 4
            (note that covariance matrix is symmetric)
    """

    # Initialization:
    pi = []
    mu = []
    cov = []
    for k in range(K):
        pi.append(1.0 / K)
        mu.append(list(np.random.normal(0, 0.5, 2)))
        temp_cov = np.random.normal(0, 0.5, (2, 2))
        temp_cov = np.matmul(temp_cov, np.transpose(temp_cov))
        cov.append(list(temp_cov.reshape(4)))

    ### you need to fill in your solution starting here ###
    # Run 100 iterations of EM updates
    m = len(X)


    for j in range(100):
        data_result = []
        for index in range(m):
            cur_point = np.array(X[index])
            pro = []
            resp = []
            for i in range(len(mu)):
                cur_mu = np.array(mu[i])
                cur_cov = np.array(cov[i]).reshape(2, 2)
                result = guassian_calculator(cur_point, cur_mu, cur_cov)
                pro.append(result)

            resp = resp_generator(pro, pi)
            data_result.append(resp)

        # print (data_result)
        # print (np.repeat(np.array(pi).reshape(1, 3), repeats = m, axis = 0 ))
        part_1 = np.repeat(np.array(pi).reshape(1, 3), repeats = m, axis = 0 )
        likelihood_value = part_1 * data_result
        # print (likelihood_value)
        # print (np.sum(likelihood_value))

        sum_c = np.sum(data_result, axis = 0)

        # update pi/ size:
        pi = np.mean(np.array(data_result), axis = 0)


        # update mean:
        r_i_c = np.array(data_result)
        x_i = np.array(X)

        for k in range(K):
            temp = r_i_c[:,k].reshape(600, 1)
            mu[k] = np.sum( ( x_i * np.repeat(temp, repeats = 2, axis = 1) ), axis = 0) / sum_c[k]
            mu[k] = mu[k].tolist()


        # update variance:
        for k in range(K):
            cur_mu = np.array(mu[k])
            cur_r_i_c = r_i_c[:,k]
            sum_cov = 0
            for index in range(m):
                cur_point = X[index]
                diff = cur_point - cur_mu
                diff = diff.reshape(1, 2)
                temp_cov = np.matmul(np.transpose(diff), diff)
                temp_cov = cur_r_i_c[index] * temp_cov
                sum_cov = sum_cov + temp_cov
            sum_cov = sum_cov / sum_c[k]
            cov[k] = list(sum_cov.reshape(4))

    return mu, cov

def main():
    # load data
    with open('hw4_blob.json', 'r') as f:
        data_blob = json.load(f)

    mu_all = {}
    cov_all = {}

    print('GMM clustering')
    for i in range(5):
        np.random.seed(i)
        mu, cov = gmm_clustering(data_blob, K=3)
        mu_all[i] = mu
        cov_all[i] = cov

        print('\nrun' + str(i) + ':')
        print('mean')
        print(np.array_str(np.array(mu), precision=4))
        print('\ncov')
        print(np.array_str(np.array(cov), precision=4))

    with open('gmm.json', 'w') as f_json:
        json.dump({'mu': mu_all, 'cov': cov_all}, f_json)


if __name__ == "__main__":
    main()