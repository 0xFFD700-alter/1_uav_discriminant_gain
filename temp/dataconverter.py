#%%
from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import scipy.io as sio

mean_class = np.load('Aircomp/save_model/save_mean_variance/mean_class_12dim.npy', allow_pickle=True)
var_class = np.load('Aircomp/save_model/save_mean_variance/var_class_12dim.npy', allow_pickle=True)
data_test_pca_normed = np.load('Aircomp/save_model/save_mean_variance/data_test_pca_normed_12dim.npy', allow_pickle=True)
label_test = np.load('Aircomp/save_model/save_mean_variance/label_test_12dim.npy', allow_pickle=True)

sio.savemat('../temp_m/data/inference/mean.mat', {'mu':mean_class})
sio.savemat('../temp_m/data/inference/var.mat', {'sigma':var_class})
sio.savemat('../temp_m/data/inference/data_test.mat', {'x':data_test_pca_normed})
sio.savemat('../temp_m/data/inference/label_test.mat', {'label_test':label_test})


