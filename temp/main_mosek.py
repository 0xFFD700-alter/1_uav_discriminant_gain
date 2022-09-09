#%%
from IPython import get_ipython
get_ipython().magic('reset -sf')

import pickle
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import sys, os, mosek

from sklearn import metrics

#%%

# Since the actual value of Infinity is ignored, we define it solely for symbolic purposes:
inf = 0.0

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

