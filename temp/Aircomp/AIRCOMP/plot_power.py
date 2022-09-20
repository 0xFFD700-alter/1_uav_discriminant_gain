import matplotlib.pyplot as plt
import numpy as np

power_list = np.load('./save_model/save_results/power_list.npy', allow_pickle=True)
svm_accuracy = np.load('./save_model/save_results/svm_accuracy_with_power.npy', allow_pickle=True)
mlp_accuracy = np.load('./save_model/save_results/mlp_accuracy_with_power.npy', allow_pickle=True)
discriminant_gain = np.load('./save_model/save_results/discriminant_gain_with_power.npy', allow_pickle=True)

#baseline
svm_baseline_accuracy = np.load('./save_model/save_results/svm_accuracy_with_power_baseline.npy', allow_pickle=True)
mlp_baseline_accuracy = np.load('./save_model/save_results/mlp_accuracy_with_power_baseline.npy', allow_pickle=True)
discriminant_baseline_gain = np.load('./save_model/save_results/discriminant_gain_with_power_baseline.npy', allow_pickle=True)
#####

##########FC################
svm_FC_accuracy = np.load('./save_model/save_results/svm_accuracy_with_power_FC.npy', allow_pickle=True)
mlp_FC_accuracy = np.load('./save_model/save_results/mlp_accuracy_with_power_FC.npy', allow_pickle=True)
discriminant_gain_FC = np.load('./save_model/save_results/discriminant_gain_with_power_FC.npy', allow_pickle=True)
############################


fig = plt.figure()
label_optimal = 'Optimal allocation (our proposal)'
label_equal = 'Baseline'
label_FC = 'Weighted subspace centroid'

################SVM######################
plt.plot(power_list, svm_accuracy * 100, '-k', marker='o', markersize=8, label=label_optimal, linewidth=2.0)
plt.plot(power_list, svm_baseline_accuracy * 100, '-.r', marker='v', markersize=8, label=label_equal, linewidth=2.0)
plt.plot(power_list, svm_FC_accuracy * 100, ':m', marker='s', markersize=8, label=label_FC, linewidth=2.0)
plt.legend(fontsize=14)
plt.xlabel('Power(mdB)', fontsize=14)
plt.ylabel('SVM Accuracy (%)', fontsize=14)
plt.grid(True, which="both", ls="-.")
plt.xlim([min(power_list), max(power_list)])
plt.savefig('./figures/svm_power_accuracy.pdf', dpi=600, format='pdf')
plt.show()

##################MLP#####################
fig = plt.figure()
plt.plot(power_list, mlp_accuracy * 100, '-k', marker='o', markersize=8, label=label_optimal, linewidth=2.0)
plt.plot(power_list, mlp_baseline_accuracy * 100, '-.r', marker='v', markersize=8, label=label_equal, linewidth=2.0)
plt.plot(power_list, mlp_FC_accuracy * 100, ':m', marker='s', markersize=8, label=label_FC, linewidth=2.0)
plt.legend(fontsize=14, loc='lower right')
plt.xlabel('Power(mdB)', fontsize=14)
plt.ylabel('MLP Accuracy (%)', fontsize=14)
plt.grid(True, which="both", ls="-.")
plt.xlim([min(power_list), max(power_list)])
plt.savefig('./figures/mlp_power_accuracy.pdf', dpi=600, format='pdf')
plt.show()


##################DIS#####################
fig = plt.figure()
plt.plot(power_list, discriminant_gain, '-k', marker='o', markersize=8, label=label_optimal, linewidth=2.0)
plt.plot(power_list, discriminant_baseline_gain, '-.r', marker='v', markersize=8, label=label_equal, linewidth=2.0)
plt.plot(power_list, discriminant_gain_FC, ':m', marker='s', markersize=8, label=label_FC, linewidth=2.0)
plt.legend(fontsize=14)
plt.xlabel('Power(mdB)', fontsize=14)
plt.ylabel('Discriminant Gain', fontsize=14)
plt.grid(True, which="both", ls="-.")
plt.xlim([min(power_list), max(power_list)])
plt.savefig('./figures/discriminant_gain_power_accuracy.pdf', dpi=600, format='pdf')
plt.show()