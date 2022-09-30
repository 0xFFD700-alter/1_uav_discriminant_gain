#%%
from IPython import get_ipython
get_ipython().magic('reset -sf')

import pickle
import numpy as np
import cvxpy as cp
# import matplotlib.pyplot as plt

from sklearn import metrics

#%%
num_train_per_class = 800
num_test_per_class = 200

mu = np.load('./data/inference/mean.npy', allow_pickle=True) # mean value
sigma = np.load('./data/inference/var.npy', allow_pickle=True) # ground-true var
delta = np.load('./data/inference/var_sensor.npy', allow_pickle=True) # distortion var
w = np.load('./data/inference/trajectory_sensor.npy', allow_pickle=True) # trajectory of sensors

z = np.load('./data/inference/data_test_noise.npy', allow_pickle=True) # distorted data
label_test = np.load('./data/inference/label_test.npy', allow_pickle=True) # label of test data

L = mu.shape[0]
N = mu.shape[-1]
K = z.shape[1]
rng = np.random.default_rng(2022)

# TODO: how to specify peak power constraints
num_a = int(K * 0.3) + 1
num_b = K - num_a
P_list = np.array([0.05] * num_a + [0.03] * num_b)
P_list *= 1e3

H = 100.0                                       # m
T = 30.0                                        # s
slot = T / N                                    # s
Vm = 20.0                                       # m/s
q_init = (200.0, 0.0)                           # m
delta_0 = 1e-11                                 # W
P = np.full((N, K), P_list).T                   # W
P_bar = (P_list / 2).T                          # W
L_0 = 1e-4                                      # dB
sca_momentum = 1 - 1e-1

# mu[mu > 0] = np.log(1 + mu[mu > 0]) / np.log(100)
# mu[mu <= 0] = -np.log(1 - mu[mu <= 0]) / np.log(100)

# def normalize(data): # TODO
#     while np.abs(data).max() / np.abs(data).min() >= 500:
#         index = np.where(np.abs(data) == np.abs(data).max())
#         data[index] /= 2
#     return data

#%%
E = delta + sigma + np.average(mu ** 2, axis=0)
c_iter = np.sqrt(P * L_0 / E / 6.6e5)
# c_iter = normalize(c_iter)

u = np.zeros(N)
for l1 in range(L - 1):
    for l2 in range(l1 + 1, L):
        u += (mu[l1] - mu[l2]) ** 2
u *= 2 / L / (L - 1)

alpha_iter = np.sum(c_iter, axis=0) ** 2 * u / (np.sum(c_iter, axis=0) ** 2 * sigma + np.sum(delta * c_iter ** 2, axis=0) + delta_0) / 1e1
# alpha_iter = normalize(alpha_iter)

#%%
q = cp.Variable((N, 2))
c = cp.Variable((K, N), nonneg=True)
alpha = cp.Variable(N, nonneg=True)

epsilon = 1e-3
epsilon_sca = 1e-1
gain = 0
gain_list = [np.sum(alpha_iter)]

svm_model = './svm_model/svm_model.pkl'
with open(svm_model, 'rb') as file:
    classifier = pickle.load(file)
    
z_hat = (np.sum(z * c_iter, axis=1) + np.random.normal(0, delta_0, N))
predicted = classifier.predict(z_hat)
accuracy = metrics.accuracy_score(label_test, predicted)
accuracy_list = [accuracy]

while True:
    objective_1 = cp.Minimize(
        cp.sum(
            cp.multiply(
                cp.vstack([cp.sum(cp.square(q - w[k]), axis=1) for k in range(K)]),
                c_iter ** 2 * E
                )
            )
        )
    
    constraints_1 = [
        cp.multiply(
            cp.vstack([cp.sum(cp.square(q - w[k]), axis=1) for k in range(K)]) + H ** 2,
            c_iter ** 2 * E
            ) <= P * L_0,  
        cp.sum(
            cp.multiply(
                cp.vstack([cp.sum(cp.square(q - w[k]), axis=1) for k in range(K)]) + H ** 2,
                c_iter ** 2 * E
                ), axis=1
            ) <= P_bar * L_0 * N,    
        cp.norm(q[0] - q_init) <= slot * Vm
        ] + [cp.norm(q[n + 1] - q[n]) <= slot * Vm for n in range(N - 1)]
    
    prob_1 = cp.Problem(objective_1, constraints_1)
    prob_1.solve(solver=cp.MOSEK, verbose=True)
    assert prob_1.status == 'optimal'
    q_iter = q.value
    
    gain_list_sca = []
    while True:
        objective_2 = cp.Maximize(cp.sum(alpha))
        constraints_2 = [
            cp.multiply(
                cp.square(c),
                (np.vstack([np.sum((q_iter - w[k]) ** 2, axis=1) for k in range(K)]) + H ** 2) * E
                ) <= P * L_0,
            cp.sum(
                cp.multiply(
                    cp.square(c),
                    (np.vstack([np.sum((q_iter - w[k]) ** 2, axis=1) for k in range(K)]) + H ** 2) * E
                    ), axis=1
                ) <= P_bar * L_0 * N,
            u / alpha_iter * np.sum(c_iter, axis=0) ** 2 \
            + cp.multiply(cp.sum(c - c_iter, axis=0), 2 * u / alpha_iter * np.sum(c_iter, axis=0)) \
            - cp.multiply(alpha - alpha_iter, u / alpha_iter ** 2 * np.sum(c_iter, axis=0) ** 2) \
            - cp.multiply(cp.square(cp.sum(c, axis=0)), sigma) \
            >= cp.sum(cp.multiply(cp.square(c), delta), axis=0) + delta_0
            ]
        
        # function value
        # u / alpha_iter * np.sum(c_iter, axis=0) ** 2

        # linear term
        # + cp.multiply(cp.sum(c - c_iter, axis=0), 2 * u / alpha_iter * np.sum(c_iter, axis=0))
        # - cp.multiply(alpha - alpha_iter, u / alpha_iter ** 2 * np.sum(c_iter, axis=0) ** 2)

        # - cp.multiply(cp.square(cp.sum(c, axis=0)), sigma)
        # >= cp.sum(cp.multiply(cp.square(c), delta), axis=0) + delta_0
        
        prob_2 = cp.Problem(objective_2, constraints_2)
        gain_iter = gain
        gain = prob_2.solve(solver=cp.MOSEK, verbose=True)
        assert prob_2.status == 'optimal'
        
        # if gain < gain_iter or gain_iter != 0 and gain / gain_iter > 5:
        #     gain = gain_iter
        #     break
        
        c_iter = sca_momentum * c_iter + (1 - sca_momentum) * c.value 
        alpha_iter = sca_momentum * alpha_iter + (1 - sca_momentum) * alpha.value
        gain_list_sca.append(gain)
        
        if np.abs(gain - gain_iter) < epsilon_sca:
            break
    
    gain_list.append(gain)
    z_hat = (np.sum(z * c_iter, axis=1) + np.random.normal(0, delta_0, N))
    predicted = classifier.predict(z_hat)
    accuracy = metrics.accuracy_score(label_test, predicted)
    accuracy_list.append(accuracy)
    
    if np.abs(gain - gain_iter) < epsilon:
        break

# gain_log = []
# gain_log.append(alpha_iter)


# if iter_count == 0:
#     sca_gain_list.append(gain)
#     z_hat = (np.sum(z * c_iter, axis=1) + np.random.normal(0, delta_0, N))
#     predicted = classifier.predict(z_hat)
#     accuracy = metrics.accuracy_score(label_test, predicted)
#     sca_accuracy_list.append(accuracy)
    
#     gain_log.append(alpha_iter)



# c_iter
# q_iter

# #%%
# fig = plt.figure(dpi=600, figsize=(6, 8))
# # plt.title('outter loop')
# ax = fig.add_subplot(211)
# # ax.set_xlim([0, 400])
# # ax.set_ylim([0, 400])
# ax.set_title('discriminant gain')
# plt.plot(gain_list)

# ax = fig.add_subplot(212)
# # ax.set_xlim([0, 400])
# # ax.set_ylim([0, 400])
# ax.set_title('accuracy')
# plt.plot(accuracy_list)

# #%%
# fig = plt.figure(dpi=600, figsize=(6, 8))
# # plt.title('inner sca loop')
# ax = fig.add_subplot(211)
# # ax.set_xlim([0, 400])
# # ax.set_ylim([0, 400])
# ax.set_title('discriminant gain')
# plt.plot(sca_gain_list)

# ax = fig.add_subplot(212)
# # ax.set_xlim([0, 400])
# # ax.set_ylim([0, 400])
# ax.set_title('accuracy')
# plt.plot(sca_accuracy_list)

# #%%
# fig = plt.figure(dpi=600)
# ax = fig.add_subplot(111)
# ax.set_title('trajectory')
# ax.set_xlim([0, 400])
# ax.set_ylim([0, 400])
# plt.grid()

# start_a, end_a = (50.0, 150.0), (50.0, 350.0)
# start_b, end_b = (350.0, 150.0), (250.0, 325.0)
# center_a = np.linspace(start_a, end_a, num=N)
# center_b = np.linspace(start_b, end_b, num=N)

# init_a, fin_a = w[:4, 0], w[:4, -1]
# init_b, fin_b = w[4:, 0], w[4:, -1]

# plt.plot(center_a[:, 0], center_a[:, 1], 'r')
# plt.plot(center_b[:, 0], center_b[:, 1], 'r')
# plt.plot(q_iter[:, 0], q_iter[:, 1], color='b')

# plt.legend(['clutter A', 'clutter B', 'UAV'], loc='best')

# plt.scatter(init_a[:, 0], init_a[:, 1], c='none', edgecolors='k')
# plt.scatter(init_b[:, 0], init_b[:, 1], c='none', edgecolors='k')
# plt.scatter(fin_a[:, 0], fin_a[:, 1], c='none', edgecolors='k')
# plt.scatter(fin_b[:, 0], fin_b[:, 1], c='none', edgecolors='k')

#%%
# fig = plt.figure(figsize=(5, 75))
# for i, gains in enumerate(gain_log):
#     ax = plt.subplot(52, 1, i + 1)
#     ax.set_ylim(0, 50)
#     plt.bar([k for k in range(N)], gains)
    
# fig = plt.figure(figsize=(20, 6), dpi=600)
# ax = plt.subplot(1, 1, 1)
# ax.set_ylim(0, 80)
# plt.bar([k for k in range(N)], alpha_iter)

#%%

# s = [1, 1, 1, 1, 1, 2, 2, 2, 2]


# def invoke(a, b):
#     return a == b

# def solution(n, s):
#     '''
#     returns:
#     m - # of cards that are equivalent
#     c - a card that has m copies
#     '''
#     if n == 1:
#         return s[0]
#     c1 = solution(n // 2, s[:n // 2])
#     c2 = solution(n - n // 2, s[n // 2:])
    
#     return merge(c1, c2, n, s)


# def merge(c1, c2, n, s):
#     '''
#     invoke(a, b):
#     test if card a and card b are equivalent
#     Python list object:
#     s1 + s2 do the same thing as s1 \cup s2
#     '''
#     count_c1 = 0
#     count_c2 = 0
#     for c in s:
#         if c1 and invoke(c1, c) == True:
#             count_c1 += 1
#         if c2 and invoke(c2, c) == True:
#             count_c2 += 1
    
#     if count_c1 > n / 2:
#         return c1
#     if count_c2 > n / 2:
#         return c2
    
#     return None
    
# ans = solution(len(s), s)