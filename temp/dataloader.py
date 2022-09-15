#%%
from IPython import get_ipython
get_ipython().magic('reset -sf')

from scipy import io as sio
from sklearn.decomposition import PCA

import numpy as np
import pickle
import os

classes = [str(i) for i in range(1, 6)]
exclude_classes = ['5']
num_class = len(classes) - len(exclude_classes)
num_sensor = 5
rng = np.random.default_rng(2022)

pca_dim = 30
pca_dim_sensor = 50
# training set per class: -> first num_train_per_class samples of each class
num_per_class = 1000
num_train_per_class = 800
num_test_per_class = 200

data_dir = './data/THREE_RADAR_STFT_MAT'
data_dir_list = os.listdir(data_dir)

model_dir = './pca_model'
model_dir_list = [f'pca_model_radar{i}.pkl' for i in range(1, len(data_dir_list) + 1)]

#%%
# process data from 3 radar, generate mean and variance

def raw_resample_real_whitening(mat, n_samples=10):
    r'''
    per radar per class
    read data from training set or test set respectively
    '''
    # resample: 400 * 38 --> 40 * 38 --> 1520 * 1
    mat_resample = mat[:, 0::n_samples]
    samples = np.abs(mat_resample.real).reshape(mat.shape[0], -1) + np.abs(mat_resample.imag).reshape(mat.shape[0], -1)
    mean = samples.mean(axis=1, keepdims=True)
    std = np.maximum(samples.std(axis=1, keepdims=True), 1.0 / np.sqrt(samples.shape[1]))
    return (samples - mean) / std

def data_from_mat(mat_dir):
    r'''
    per radar
    return both training set and test set
    '''
    data_train = []
    data_test = []
    label_train = []
    label_test = []
    dir_list = sorted(os.listdir(mat_dir))
    for mat_name in dir_list:
        class_id = mat_name[mat_name.rindex('_') + 1 : mat_name.rindex('.')]
        if class_id in exclude_classes:
            continue
        mat_fname = os.path.join(mat_dir, mat_name)
        mat_contents = sio.loadmat(mat_fname)
        mat_resample_whitened = raw_resample_real_whitening(mat_contents['mat'])
        data_train.append(mat_resample_whitened[:num_train_per_class])
        data_test.append(mat_resample_whitened[num_train_per_class:])
        label_train.append([class_id] * num_train_per_class)
        label_test.append([class_id] * num_test_per_class)
    return np.stack(data_train), np.stack(data_test), np.stack(label_train), np.stack(label_test)

def gen_pca_model(data, pca_model, pca_dim):
    r'''
    generate pca model using training data (per radar)
    '''
    if os.path.exists(pca_model) is False:
        pca_spec_raw = PCA(n_components=pca_dim)
        pca_spec_raw.fit(data)
        with open(pca_model, 'wb') as file:
            pickle.dump(pca_spec_raw, file)
    else:
        with open(pca_model, 'rb') as file:
            pca_spec_raw = pickle.load(file)
    return pca_spec_raw

def pca_preprocess(data_dir_list, model_dir_list):
    r'''
    warpper
    '''
    data_train_list = []
    data_test_list = []
    dir_list = zip(data_dir_list, model_dir_list)
    for data, model in dir_list:
        data_train, data_test, label_train, label_test = data_from_mat(os.path.join(data_dir, data))
        
        # PCA
        data_train = data_train.reshape(-1, data_train.shape[-1])
        data_test = data_test.reshape(-1, data_test.shape[-1]) 
        pca_model = gen_pca_model(data_train, os.path.join(model_dir, model), pca_dim_sensor)
        data_train_pca = pca_model.transform(data_train)
        data_test_pca = pca_model.transform(data_test)
        data_train_list.append(data_train_pca.reshape(num_class, -1, pca_dim_sensor))
        data_test_list.append(data_test_pca.reshape(num_class, -1, pca_dim_sensor))
        
        # raw data
        # data_train_list.append(data_train)
        # data_test_list.append(data_test)
    return np.stack(data_train_list, axis=2), np.stack(data_test_list, axis=2), label_train, label_test

#%%
# add distrotion to original data
# generate trajectory of sensors

def add_distortion(data_true, var_sensor):
    data_noise = []
    for k in range(num_sensor):
        data_noise.append(
            data_true + np.hstack([
                rng.normal(0, np.sqrt(var_sensor[k, n]), (num_test_per_class * num_class, 1)) for n in range(data_true.shape[-1])
                ])
            )
    return np.stack(data_noise, axis=1)

# TODO: only init locations are random
def generate_trajectory(num, start_a, end_a, start_b, end_b, radius=30, ratio=0.3):
    num_a = int(num_sensor * ratio) + 1
    num_b = num_sensor - num_a
    
    rad_a = rng.uniform(0, 2 * np.pi, num_a)
    dist_a = rng.uniform(0, radius, num_a)
    rad_b = rng.uniform(0, 2 * np.pi, num_b)
    dist_b = rng.uniform(0, radius, num_b)
    
    init_a = start_a + np.stack([dist_a * np.cos(rad_a), dist_a * np.sin(rad_a)]).T
    init_b = start_b + np.stack([dist_b * np.cos(rad_b), dist_b * np.sin(rad_b)]).T
    init_a = np.expand_dims(init_a, 1)
    init_b = np.expand_dims(init_b, 1)
    
    center_a = np.linspace(start_a, end_a, num=num)
    center_b = np.linspace(start_b, end_b, num=num)
    step_a = np.cumsum(center_a - np.vstack((center_a[0], center_a[:num - 1])), axis=0)
    step_b = np.cumsum(center_b - np.vstack((center_b[0], center_b[:num - 1])), axis=0)
    
    return np.vstack([init_a + step_a, init_b + step_b])

#%%
def normalize(data): # TODO
    while np.abs(data).max() / np.abs(data).min() >= 500:
        index = np.where(np.abs(data) == np.abs(data).max())
        data[index] /= 2
    return data

#%%
# 1. normalize -> extract mean and var -> rng fake data
# data = np.stack([rng.normal(mean[i], np.sqrt(var[i]), (num_per_class, len(var[i]))) for i in range(num_class)])
# 2. pca to 30 -> normalize

if __name__ == '__main__':
    # data_train_pca, data_test_pca: -> (class, sample, radar, pca_dim)
    data_train_pca, data_test_pca, label_train, label_test = pca_preprocess(data_dir_list, model_dir_list)
    
    # normalize the variance of data_pca to 1
    data_pca = np.concatenate((data_train_pca, data_test_pca), axis=1)
    data_pca = data_pca.reshape(num_class, num_per_class, -1)
    label = np.concatenate((label_train, label_test), axis=1)
    
    # TODO: fake data
    # mean: -> (class, radar * pca_dim)  var: -> (radar * pca_dim)
    data_pca = data_pca.reshape(num_class * num_per_class, -1)
    pca_model = gen_pca_model(data_pca[:, :num_train_per_class], os.path.join(model_dir, 'pca_model.pkl'), pca_dim)
    # data = data_pca.reshape(num_class, -1, 150)
    data = pca_model.transform(data_pca).reshape(num_class, -1, pca_dim)
    
#%%
    # var = data.var(axis=1).mean(axis=0)
    # var = (var - var.min()) / (var.max() - var.min())
    # std = data.std(axis=1, keepdims=True)
    # mean = data.mean(axis=(0, 1), keepdims=True)
    # mean = (data - mean).mean(axis=1)
    
    mean = data.mean(axis=(0, 1), keepdims=True)
    std = data.std(axis=1, keepdims=True)
    data_normed = (data - mean) / std
    mean = data_normed.mean(axis=1)
    var = data_normed.var(axis=1).mean(axis=0)
    
    np.save('./data/training/data.npy', data_normed)
    np.save('./data/training/label.npy', label)
    np.save('./data/inference/mean.npy', mean)
    np.save('./data/inference/var.npy', var)
    
    sio.savemat('../temp_m/data/training/data.mat', {'data':data_normed})
    sio.savemat('../temp_m/data/training/label.mat', {'label':label})
    sio.savemat('../temp_m/data/inference/mean.mat', {'mu':mean})
    sio.savemat('../temp_m/data/inference/var.mat', {'sigma':var})
    
    # data_test_true -> (num_class * num_test_per_class, radar * pca_dim)
    var_list = rng.uniform(0.1, 0.6, num_sensor)
    data_test_true = data_normed[:, num_train_per_class:].reshape(num_class * num_test_per_class, -1)
    # TODO: different delta for different dim
    var_sensor = np.full((data_test_true.shape[-1], num_sensor), var_list).T
    data_test_noise = add_distortion(data_test_true, var_sensor)
    label_test = label[:, num_train_per_class:].reshape(-1)
    
    start_a, end_a = (50.0, 150.0), (50.0, 350.0) # TODO: not staright trajectory
    start_b, end_b = (350.0, 150.0), (250.0, 325.0) # TODO: not staright trajectory
    trajectory_sensor = generate_trajectory(data_test_true.shape[-1], start_a, end_a, start_b, end_b)
    
    np.save('./data/inference/var_sensor.npy', var_sensor)
    np.save('./data/inference/data_test_noise.npy', data_test_noise)
    np.save('./data/inference/label_test.npy', label_test)
    np.save('./data/inference/trajectory_sensor.npy', trajectory_sensor)
    
    sio.savemat('../temp_m/data/inference/var_sensor.mat', {'delta':var_sensor})
    sio.savemat('../temp_m/data/inference/data_test_noise.mat', {'z':data_test_noise})
    sio.savemat('../temp_m/data/inference/label_test.mat', {'label_test':label_test})
    sio.savemat('../temp_m/data/inference/trajectory_sensor.mat', {'w':trajectory_sensor})

