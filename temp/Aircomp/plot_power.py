# import matplotlib.pyplot as plt
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


# fig = plt.figure()
# label_optimal = 'Our proposal'
# label_equal = 'Baseline'
# label_FC = 'Weighted subspace centroid'
#
# ################SVM######################
# plt.plot(power_list[0:7], svm_accuracy[0:7] * 100, '-k', marker='o', markersize=8, label=label_optimal, linewidth=2.0)
# plt.plot(power_list[0:7], svm_baseline_accuracy[0:7] * 100, '-.r', marker='v', markersize=8, label=label_equal, linewidth=2.0)
# plt.plot(power_list[0:7], svm_FC_accuracy[0:7] * 100, ':m', marker='s', markersize=8, label=label_FC, linewidth=2.0)
# plt.legend(fontsize=14, loc='lower left')
# plt.xlabel('Power(mdB)', fontsize=14)
# plt.ylabel('SVM Accuracy (%)', fontsize=14)
# plt.grid(True, which="both", ls="-.")
# plt.xticks(power_list[0:7])
# plt.savefig('./figures/svm_power_accuracy.pdf', dpi=600, format='pdf')
# plt.show()
#
# ##################MLP#####################
# fig = plt.figure()
# plt.plot(power_list[0:7], mlp_accuracy[0:7] * 100, '-k', marker='o', markersize=8, label=label_optimal, linewidth=2.0)
# plt.plot(power_list[0:7], mlp_baseline_accuracy[0:7] * 100, '-.r', marker='v', markersize=8, label=label_equal, linewidth=2.0)
# plt.plot(power_list[0:7], mlp_FC_accuracy[0:7] * 100, ':m', marker='s', markersize=8, label=label_FC, linewidth=2.0)
# plt.legend(fontsize=14, loc='lower left')
# plt.xlabel('Power(mdB)', fontsize=14)
# plt.ylabel('MLP Accuracy (%)', fontsize=14)
# plt.grid(True, which="both", ls="-.")
# plt.xticks(power_list[0:7])
# plt.savefig('./figures/mlp_power_accuracy.pdf', dpi=600, format='pdf')
# plt.show()
#
#
# ##################DIS#####################
# fig = plt.figure()
# plt.plot(power_list[0:7], discriminant_gain[0:7], '-k', marker='o', markersize=8, label=label_optimal, linewidth=2.0)
# plt.plot(power_list[0:7], discriminant_baseline_gain[0:7], '-.r', marker='v', markersize=8, label=label_equal, linewidth=2.0)
# plt.plot(power_list[0:7], discriminant_gain_FC[0:7], ':m', marker='s', markersize=8, label=label_FC, linewidth=2.0)
# plt.legend(fontsize=14)
# plt.xlabel('Power(mdB)', fontsize=14)
# plt.ylabel('Discriminant Gain', fontsize=14)
# plt.grid(True, which="both", ls="-.")
# plt.xticks(power_list[0:7])
# plt.savefig('./figures/discriminant_gain_power.pdf', dpi=600, format='pdf')
# plt.show()