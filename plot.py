import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')

acc_training = np.load('./result/train_acc.npy')
acc_valid = np.load('./result/valid_acc.npy')

loss_training = np.load('./result/cross_entropy.npy')
loss_valid = np.load('./result/valid_loss.npy')

plt.plot(np.arange(2000), loss_training, c='r',linestyle='--',marker='^', markersize='3',label="Training")
plt.plot(np.arange(2000), loss_valid, c='b',linestyle=':',marker='*', markersize='3',label="Validation")

plt.legend(loc=0,fontsize=25)
plt.xticks([0,500,1000,1500,2000],fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('Epoch',fontsize=30)
# plt.ylabel('Recognition accuracy',fontsize=30)
plt.ylabel('Recognition loss',fontsize=30)
plt.tight_layout()
plt.savefig('./result/loss.png', format='png')
plt.show()

plt.plot(np.arange(2000), acc_training, 'r', label="Training")
plt.plot(np.arange(2000), acc_valid, 'b', label="Validation")
plt.legend(loc=0,fontsize=25)
plt.xticks([0,500,1000,1500,2000],fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('Epoch',fontsize=30)
plt.ylabel('Recognition accuracy',fontsize=30)
plt.tight_layout()
plt.savefig('./result/accuracy.png', format='png')
plt.show()
