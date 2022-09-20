import random

import numpy as np
from numpy.random import default_rng
import cvxpy as cp
from sklearn import datasets, svm, metrics
import pickle

num_antenna = 3
num_device = 3
rng = default_rng(1999)
var_dist = rng.uniform(0, 0.4, (num_device, 2))
# radius = 500  # BS in the center of circle of radius = 500 m
# radius_inner = 450  # Distance between device circle and the server
# chl_shadow_std_db = 8  # shadow fading standard deviation = 8 dB
# user_dist = (radius - radius_inner) * rng.uniform(0,1,(num_device, 1)) + radius_inner
# user_pl_db = 128.1 + 37.6 * np.log10(user_dist / 1e3)  # path loss in dB
# user_pl_db = user_pl_db - chl_shadow_std_db
# user_pl = 10 ** (-user_pl_db / 10)
# rayli_fading_real = rng.normal(0, 1, (1, num_antenna))  # rayleigh fading ~ CN(0,1)
# rng3 = default_rng(999)
# rayli_fading_img = rng3.normal(0, 1, (1, num_antenna))
# for i in range(1, num_device):
#     rng2 = default_rng(i)
#     rng3 = default_rng(100 - i)
#     rayli_fading_real = np.vstack((rayli_fading_real, rng2.normal(0, 1, (1, num_antenna))))  # rayleigh fading ~ CN(0,1)
#     rayli_fading_img = np.vstack((rayli_fading_img, rng3.normal(0, 1, (1, num_antenna))))
# rayli_fading_gain = rayli_fading_real ** 2 + rayli_fading_img ** 2
b = np.log10(2)

power_mdB = np.linspace(-10,13,40)
power_list = 10 ** ( (power_mdB-30) / 10 )

a = power_list[::2]