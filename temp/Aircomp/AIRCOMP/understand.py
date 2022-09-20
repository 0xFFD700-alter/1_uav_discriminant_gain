import numpy as np
from numpy.random import default_rng
import cvxpy as cp
from sklearn import datasets, svm, metrics
import pickle

a = [1,2,3,4,5]
b = [5,4,3,2,1]
c = a[0:2] * 3