#%%
from IPython import get_ipython
get_ipython().magic('reset -sf')

from sklearn import svm, metrics

import numpy as np
import pickle

#%%
if __name__ == '__main__':
    num_class = 4
    num_train_per_class = 800
    num_test_per_class = 200
    
    data = np.load('./data/training/data.npy', allow_pickle=True)
    label = np.load('./data/training/label.npy', allow_pickle=True)
    
    data_train = data[:, :num_train_per_class].reshape(num_class * num_train_per_class, -1)
    data_test = data[:, num_train_per_class:].reshape(num_class * num_test_per_class, -1)
    label_train = label[:, :num_train_per_class].reshape(-1)
    label_test = label[:, num_train_per_class:].reshape(-1)
    
    svm_model = './svm_model/svm_model.pkl'
    
    # if os.path.exists(svm_model) is False:
    #     classifier = svm.SVC(C=1, gamma=0.001)
    #     classifier.fit(data_train, label_train)
    #     with open(svm_model, 'wb') as file:
    #         pickle.dump(classifier, file)
    # else:
    #     with open(svm_model, 'rb') as file:
    #         classifier = pickle.load(file)
    
    classifier = svm.SVC(C=1, gamma=0.001)
    classifier.fit(data_train, label_train)
    with open(svm_model, 'wb') as file:
        pickle.dump(classifier, file)
    
    predicted = classifier.predict(data_test)
    accuracy = metrics.accuracy_score(label_test, predicted)
