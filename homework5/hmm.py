from __future__ import print_function
import json
import numpy as np
import sys

def forward(pi, A, B, O):
  """
  Forward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)
  """
  S = len(pi)
  N = len(O)
  alpha = np.zeros([S, N])

  pi_1 = pi[0]
  pi_2 = pi[1]
  state_1 = 0
  state_2 = 1

  for i in range(N):
    alpha[state_1, i] = pi_1 * B[state_1, O[i]]
    alpha[state_2, i] = pi_2 * B[state_2, O[i]]

    pi_1 = alpha[state_1, i] * A[state_1, state_1] + alpha[state_2, i] * A[state_2, state_1]
    pi_2 = alpha[state_1, i] * A[state_1, state_2] + alpha[state_2, i] * A[state_2, state_2]


  return alpha


def backward(pi, A, B, O):
  """
  Backward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  """
  S = len(pi)
  N = len(O)
  beta = np.zeros([S, N])

  state_1 = 0
  state_2 = 1
  state_1_beta = 1
  state_2_beta = 1

  beta[state_1, N - 1] = state_1_beta
  beta[state_2, N - 1] = state_2_beta

  for i in range(N - 2, -1, -1):
    beta[state_1, i] = state_1_beta * A[state_1, state_1] * B[state_1, O[i + 1]] + state_2_beta * A[state_1, state_2]  * B[state_2, O[i + 1]]
    beta[state_2, i] = state_1_beta * A[state_2, state_1] * B[state_1, O[i + 1]] + state_2_beta * A[state_2, state_2] * B[state_2, O[i + 1]]
    state_1_beta = beta[state_1, i]
    state_2_beta = beta[state_2, i]

  return beta

def seqprob_forward(alpha):
  """
  Total probability of observing the whole sequence using the forward algorithm

  Inputs:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  ###################################################
  N = alpha.shape[1]
  prob = np.sum(alpha[:, N - 1])
  
  return prob


def seqprob_backward(beta, pi, B, O):
  """
  Total probability of observing the whole sequence using the backward algorithm

  Inputs:
  - beta: A numpy array beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence
      (in terms of the observation index, not the actual symbol)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  state_1 = 0
  state_2 = 1
  pi_1 = pi[0]
  pi_2 = pi[1]

  prob = beta[state_1, 0] * pi_1 * B[state_1, O[0]] + beta[state_2, 0] * pi_2 * B[state_2, O[0]]
  
  return prob

def viterbi(pi, A, B, O):
  """
  Viterbi algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - path: A list of the most likely hidden state path k* (in terms of the state index)
    argmax_k P(s_k1:s_kT | x_1:x_T)
  """
  S = len(pi)
  N = len(O)
  alpha = np.zeros([S, N])

  pi_1 = pi[0]
  pi_2 = pi[1]
  state_1 = 0
  state_2 = 1
  path_s1 = []
  path_s2 = []

  for i in range(N):
    state_1_alpha = pi_1 * B[state_1, O[i]]
    state_2_alpha = pi_2 * B[state_2, O[i]]

    if i == 0:
      path_s1.append(state_1)
      path_s2.append(state_2)

      # print ("Path1: state1 is: ", state_1_alpha, path_s1)
      # print ("Path2: state2 is: ", state_2_alpha, path_s2)

    alpha[state_1, i] = state_1_alpha
    alpha[state_2, i] = state_2_alpha
    if i != N - 1:
      if state_1_alpha * A[state_1, state_1] < state_2_alpha * A[state_2, state_1]:
        pi_1 = state_2_alpha * A[state_2, state_1]
        path_s1.append(state_2)
      else:
        pi_1 = state_1_alpha * A[state_1, state_1]
        path_s1.append(state_1)
      # print ("Path1: state1 is: ", state_1_alpha * A[state_1, state_1] * B[state_1, O[i+1]], " state2 is: ", state_2_alpha * A[state_2, state_1] * B[state_1, O[i+1]], path_s1)

      if state_1_alpha * A[state_1, state_2] < state_2_alpha * A[state_2, state_2]:
        pi_2 = state_2_alpha * A[state_2, state_2]
        path_s2.append(state_2)
      else: 
        pi_2 = state_1_alpha * A[state_1, state_2]
        path_s2.append(state_1) 
      # print ("Path2: state1 is: ", state_1_alpha * A[state_1, state_2] * B[state_2, O[i+1]], " state2 is: ", state_2_alpha * A[state_2, state_2] * B[state_2, O[i+1]], path_s2)

  if alpha[state_1, N - 1] < alpha[state_2, N - 1]:
    path = path_s2
  else:
    path = path_s1

  return path


##### DO NOT MODIFY ANYTHING BELOW THIS ###################
def main():
  model_file = sys.argv[1]
  Osymbols = sys.argv[2]

  #### load data ####
  with open(model_file, 'r') as f:
    data = json.load(f)
  A = np.array(data['A'])
  B = np.array(data['B'])
  pi = np.array(data['pi'])
  #### observation symbols #####
  obs_symbols = data['observations']
  #### state symbols #####
  states_symbols = data['states']

  N = len(Osymbols)
  O = [obs_symbols[j] for j in Osymbols]

  alpha = forward(pi, A, B, O)
  beta = backward(pi, A, B, O)

  prob1 = seqprob_forward(alpha)
  prob2 = seqprob_backward(beta, pi, B, O)
  print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

  viterbi_path = viterbi(pi, A, B, O)

  print('Viterbi best path is ')
  for j in viterbi_path:
    print(states_symbols[j], end=' ')

if __name__ == "__main__":
  main()