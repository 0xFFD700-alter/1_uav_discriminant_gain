import matplotlib.pyplot as plt
import numpy as np

PCA_dim_list = np.load('./save_model/save_results/PCA_dim_list.npy', allow_pickle=True)
svm_accuracy = np.load('./save_model/save_results/svm_accuracy_with_PCA_dim.npy', allow_pickle=True)
mlp_accuracy = np.load('./save_model/save_results/mlp_accuracy_with_PCA_dim.npy', allow_pickle=True)
discriminant_gain = np.load('./save_model/save_results/discriminant_gain_with_PCA_dim.npy', allow_pickle=True)

#baseline
svm_baseline_accuracy = np.load('./save_model/save_results/svm_accuracy_with_PCA_dim_baseline.npy', allow_pickle=True)
mlp_baseline_accuracy = np.load('./save_model/save_results/mlp_accuracy_with_PCA_dim_baseline.npy', allow_pickle=True)
discriminant_baseline_gain = np.load('./save_model/save_results/discriminant_gain_with_PCA_dim_baseline.npy', allow_pickle=True)
#####

##########FC################
svm_FC_accuracy = np.load('./save_model/save_results/svm_accuracy_with_PCA_dim_FC.npy', allow_pickle=True)
mlp_FC_accuracy = np.load('./save_model/save_results/mlp_accuracy_with_PCA_dim_FC.npy', allow_pickle=True)
discriminant_gain_FC = np.load('./save_model/save_results/discriminant_gain_with_PCA_dim_FC.npy', allow_pickle=True)
############################


fig = plt.figure()
label_optimal = 'Our proposal'
label_equal = 'Baseline'
label_FC = 'Weighted subspace centroid'


plt.plot(PCA_dim_list[1:], svm_accuracy[1:] * 100, '-k', marker='o', markersize=8, label=label_optimal, linewidth=2.0)
plt.plot(PCA_dim_list[1:], svm_baseline_accuracy[1:] * 100, '-.r', marker='v', markersize=8, label=label_equal, linewidth=2.0)
plt.plot(PCA_dim_list[1:], svm_FC_accuracy[1:] * 100, ':m', marker='s', markersize=8, label=label_FC, linewidth=2.0)
plt.legend(fontsize=14)
plt.xlabel('PCA dimension', fontsize=14)
plt.ylabel('SVM Accuracy (%)', fontsize=14)
plt.grid(True, which="both", ls="-.")
plt.savefig('./figures/svm_PCA_dim_accuracy.pdf', dpi=600, format='pdf')
plt.show()

##################MLP#####################
fig = plt.figure()
plt.plot(PCA_dim_list[1:], mlp_accuracy[1:] * 100, '-k', marker='o', markersize=8, label=label_optimal, linewidth=2.0)
plt.plot(PCA_dim_list[1:], mlp_baseline_accuracy[1:] * 100, '-.r', marker='v', markersize=8, label=label_equal, linewidth=2.0)
plt.plot(PCA_dim_list[1:], mlp_FC_accuracy[1:] * 100, ':m', marker='s', markersize=8, label=label_FC, linewidth=2.0)
plt.legend(fontsize=14)
plt.xlabel('PCA dimension', fontsize=14)
plt.ylabel('MLP Accuracy (%)', fontsize=14)
plt.grid(True, which="both", ls="-.")
plt.savefig('./figures/mlp_PCA_dim_accuracy.pdf', dpi=600, format='pdf')
plt.show()



##################DIS#####################
fig = plt.figure()
plt.plot(PCA_dim_list[1:], discriminant_gain[1:], '-k', marker='o', markersize=8, label=label_optimal, linewidth=2.0)
plt.plot(PCA_dim_list[1:], discriminant_baseline_gain[1:], '-.r', marker='v', markersize=8, label=label_equal, linewidth=2.0)
plt.plot(PCA_dim_list[1:], discriminant_gain_FC[1:], ':m', marker='s', markersize=8, label=label_FC, linewidth=2.0)
plt.legend(fontsize=14)
plt.xlabel('PCA dimension', fontsize=14)
plt.ylabel('Discriminant Gain', fontsize=14)
plt.grid(True, which="both", ls="-.")
plt.savefig('./figures/discriminant_gain_PCA_dim.pdf', dpi=600, format='pdf')
plt.show()