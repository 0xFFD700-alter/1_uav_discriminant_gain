import numpy as np
from numpy.random import default_rng
from sklearn import datasets, svm, metrics
import pickle


def FC(power_tx, num_device, PCA_dim):
    # parameters
    num_antenna = 8  # number of antenna, N
    num_class = 4
    var_dist_scale = 0.4
    var_comm_noise = 1  # communication noise, sigma_{0}^{2}
    rng = default_rng(1555)
    bandwidth = 1.5 * 10**6

    var_dist = rng.uniform(0, var_dist_scale, (num_device, PCA_dim))

    radius = 500  # BS in the center of circle of radius = 500 m
    radius_inner = 450  # Distance between device circle and the server
    chl_shadow_std_db = 8  # shadow fading standard deviation = 8 dB
    user_dist = (radius - radius_inner) * rng.uniform(0,1,(num_device, 1)) + radius_inner
    user_pl_db = 128.1 + 37.6 * np.log10(user_dist / 1e3)  # path loss in dB
    user_pl_db = user_pl_db - chl_shadow_std_db
    user_pl = 10 ** (-user_pl_db / 10)
    # rayli_fading_real = rng.normal(0, 1, (1, num_antenna))  # rayleigh fading ~ CN(0,1)
    # rng3 = default_rng(999)
    # rayli_fading_img = rng3.normal(0, 1, (1, num_antenna))
    # for i in range(1,num_device):
    #     rng2 = default_rng(i+1000)
    #     rng3 = default_rng(2000-i)
    #     rayli_fading_real = np.vstack((rayli_fading_real,rng2.normal(0, 1, (1, num_antenna))))   # rayleigh fading ~ CN(0,1)
    #     rayli_fading_img = np.vstack((rayli_fading_img,rng3.normal(0, 1, (1, num_antenna))))
    rayli_fading_real = rng.normal(0, 1, (num_device, num_antenna))  # rayleigh fading ~ CN(0,1)
    rayli_fading_img = rng.normal(0, 1, (num_device, num_antenna))
    rayli_fading_gain = rayli_fading_real ** 2 + rayli_fading_img ** 2
    noise_power = 10**(-17.4) * bandwidth  # from You's paper
    channel_gain = user_pl * np.ones((1, num_antenna)) * np.sqrt(rayli_fading_gain) / noise_power


    def add_noise_to_normed_pca(data_test_pca_normed, A, B):
        data_test_pca_add_noise = (A.T @ channel_gain.T @ B) * data_test_pca_normed + np.sum(A * var_comm_noise)
        return data_test_pca_add_noise

    def model_inference(data, label, model):
        predicted = model.predict(data)
        accuracy = metrics.accuracy_score(label, predicted)
        return accuracy


    Sigma = np.zeros((num_antenna,num_antenna))
    for i in range(num_device):
        Sigma += channel_gain.T @ channel_gain

    U, s, V = np.linalg.svd(Sigma)
    F = U[:,0].reshape(-1,1)

    eta = 0
    for i in range(num_device):
        h = channel_gain[i].reshape(-1,1)
        temp = np.trace( 1/( F.T @ h @ h.T @ F ) ) / power_tx
        if temp>eta:
            eta = temp

    A = np.sqrt(eta) * F

    B = (A.T @ channel_gain.T).T / ( A.T @ channel_gain.T @ channel_gain @ A)

    mean_class_dir = './save_model/save_mean_variance/mean_class_{}dim.npy'.format(PCA_dim)
    var_class_dir = './save_model/save_mean_variance/var_class_{}dim.npy'.format(PCA_dim)
    data_test_pca_normed_dir = './save_model/save_mean_variance/data_test_pca_normed_{}dim.npy'.format(PCA_dim)
    label_test_dir = './save_model/save_mean_variance/label_test_{}dim.npy'.format(PCA_dim)

    mean_class = np.load(mean_class_dir, allow_pickle=True)
    var_class = np.load(var_class_dir, allow_pickle=True)
    data_test_pca_normed = np.load(data_test_pca_normed_dir, allow_pickle=True)
    label_test = np.load(label_test_dir, allow_pickle=True)

    data_test_pca_add_noise = add_noise_to_normed_pca(data_test_pca_normed, A, B)

    svm_model_file = './save_model/svm_model_{}dimension.pkl'.format(PCA_dim)
    mlp_model_file = './save_model/mlp_model_{}dimension.pkl'.format(PCA_dim)

    with open(svm_model_file, 'rb') as file:
        svm_model = pickle.load(file)
    with open(mlp_model_file, 'rb') as file:
        mlp_model = pickle.load(file)

    svm_accuracy = model_inference(data_test_pca_add_noise, label_test, svm_model)
    mlp_accuracy = model_inference(data_test_pca_add_noise, label_test, mlp_model)

    disc_init = 0
    for idx in range(0, int(PCA_dim / 2)):
        alpha_init = np.zeros((int(num_class * (num_class - 1) / 2), 2))
        for idx_class_a in range(num_class):
            for idx_class_b in range(idx_class_a):
                for idx_m in range(2):
                    alpha_init[int(idx_class_a * (idx_class_a-1) / 2) + idx_class_b, idx_m] = ((mean_class[idx_class_a, 2 * idx + idx_m] - mean_class[idx_class_b, 2 * idx + idx_m]) ** 2) / ((np.sum(var_dist[:, 2 * idx + idx_m].reshape(1,num_device) @ (B ** 2)) + var_comm_noise * np.sum(A ** 2)) / ((np.sum(B)) ** 2) + var_class[2 * idx + idx_m])

        disc_init += 2*alpha_init.sum() / num_class / (num_class-1)


    noise = np.sum(A * var_comm_noise)

    return svm_accuracy, mlp_accuracy, disc_init, noise, data_test_pca_add_noise, data_test_pca_normed

PCA_dim = 12  # PCA dimension, M
num_device = 3  # number of devices, K
power_mdB = 12
power_tx = 10 ** ( (power_mdB-30) / 10 )  # transmit power, P_{k}

#####the below is computating the accuracy with the change of power#####
power_mdB = np.linspace(0,20,11)
power_list = 10 ** ( (power_mdB-30) / 10 )

np.save('./save_model/save_results/power_list.npy', power_mdB)
svm_accuracy_list = np.zeros((len(power_list), 1))
mlp_accuracy_list = np.zeros((len(power_list), 1))
discriminant_gain_list = np.zeros((len(power_list), 1))

for i in range(len(power_list)):
    power_tx = power_list[i]
    svm_accuracy_list[i], mlp_accuracy_list[i], discriminant_gain_list[i], noise, data_add_noise, data_normed = FC(power_tx,num_device,PCA_dim)

np.save('./save_model/save_results/svm_accuracy_with_power_FC.npy', svm_accuracy_list)
np.save('./save_model/save_results/mlp_accuracy_with_power_FC.npy', mlp_accuracy_list)
np.save('./save_model/save_results/discriminant_gain_with_power_FC.npy', discriminant_gain_list)
######################################################################################



# #####the below is computating the accuracy with the change of number sizes.#####
# num_device_list = np.linspace(1,13,7)
# np.save('./save_model/save_results/num_device_list.npy', num_device_list)
# svm_accuracy_list = np.zeros((len(num_device_list), 1))
# mlp_accuracy_list = np.zeros((len(num_device_list), 1))
# discriminant_gain_list = np.zeros((len(num_device_list), 1))
#
# for i in range(len(num_device_list)):
#     num_device = int(num_device_list[i])
#     svm_accuracy_list[i], mlp_accuracy_list[i], discriminant_gain_list[i], noise, data_add_noise, data_normed  = FC(power_tx,num_device,PCA_dim)
#
# np.save('./save_model/save_results/svm_accuracy_with_num_device_FC.npy', svm_accuracy_list)
# np.save('./save_model/save_results/mlp_accuracy_with_num_device_FC.npy', mlp_accuracy_list)
# np.save('./save_model/save_results/discriminant_gain_with_num_device_FC.npy', discriminant_gain_list)
# ######################################################################################
