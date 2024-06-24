
import numpy as np

pred = np.load('data/vad_ego_prediction.npy')
label = np.load('data/val_label.npy')
val_input = np.load('data/val.npy')

L2_1s_list = []
L2_2s_list = []
L2_3s_list = []


for i in range(len(pred)):
    traj_pred = pred[i]
    traj_label = label[i]
    # x_diff = traj_label[10]-traj_label[0]
    # if x_diff<=-2 or x_diff>=2:
    #     continue
    #else:
    #    continue
    L2_1s = np.sqrt(((traj_pred[2:4] - traj_label[2:4]) ** 2).sum())
    L2_2s = np.sqrt(((traj_pred[6:8] - traj_label[6:8]) ** 2).sum())
    L2_3s = np.sqrt(((traj_pred[10:12] - traj_label[10:12]) ** 2).sum())

    L2_1s_list.append(L2_1s)
    L2_2s_list.append(L2_2s)
    L2_3s_list.append(L2_3s)

print('total number of data counted:',len(L2_3s_list))

print('1s:',np.mean(L2_1s_list))
print('2s:',np.mean(L2_2s_list))
print('3s:',np.mean(L2_3s_list))



