import matplotlib.pyplot as plt
import numpy as np

svm_accuracy = np.load('./save_model/save_results/svm_acc_dis.npy', allow_pickle=True)
mlp_accuracy = np.load('./save_model/save_results/mlp_acc_dis.npy', allow_pickle=True)
discriminant_gain = np.load('./save_model/save_results/discriminant_gain_accuracy.npy', allow_pickle=True)



fig = plt.figure()
plt.plot(discriminant_gain[::8], svm_accuracy[::8] * 100, '-k', label='SVM', linewidth=2.0)
plt.plot(discriminant_gain[::8], mlp_accuracy[::8] * 100, '--k', label='Neural network', linewidth=2.0)
plt.legend(fontsize=14)
plt.xlabel('Discriminant gain', fontsize=14)
plt.ylabel('Inference accuracy (%)', fontsize=14)
plt.grid(True, which="both", ls="-.")
plt.xlim([min(discriminant_gain), max(discriminant_gain)])
plt.ylim([25, 100])
plt.savefig('./figures/discriminant_gain_accuracy.pdf', dpi=600, format='pdf')
plt.show()
