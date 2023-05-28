import numpy as np
import matplotlib.pyplot as plt
from openpyxl import Workbook

epoch_list = []
loss_list = []
# sketch_acc_list = []
acc_list = []

log_file = './saved_model/saved_model1673318065/epoch.log'

with open(log_file, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        epoch, loss, sketch_acc, acc, time = line.strip().split(', ')
        epoch_list.append(int(epoch.replace('Epoch: ', '')) - 1)
        loss_list.append(float(loss.replace('Loss: ', '')))
        # sketch_acc_list.append(float(sketch_acc.replace('Sketch Acc: ', '')))
        acc_list.append(float(acc.replace('Acc: ', '')))

# print(epoch_list)
# print(loss_list)
# print(sketch_acc_list)
# print(acc_list)
epoch_list = np.array(epoch_list)
loss_list = np.array(loss_list)
# sketch_acc_list = np.array(sketch_acc_list)
acc_list = np.array(acc_list)

# plt.plot(epoch_list, loss_list, 'xkcd:blue')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.savefig('irnet_loss.jpg')

# plt.plot(epoch_list, acc_list, 'xkcd:orange')
# plt.xlabel('Epoch')
# plt.ylabel('Exact Match Accuracy(%)')
# plt.savefig('irnet_acc.jpg')

wb = Workbook()
ws = wb.active
# for e, l in zip(epoch_list, loss_list):
#     ws.append([int(e), float(l)])
# wb.save('irnet_loss.xlsx')

for e, a in zip(epoch_list, acc_list):
    ws.append([int(e), float(float(a) * 100)])
wb.save('irnet_acc.xlsx')