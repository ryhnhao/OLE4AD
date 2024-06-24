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
    x_diff = traj_label[10]-traj_label[0]
    if x_diff<=-2 or x_diff>=2:
        continue
    #else:
    #    continue
    L2_1s = np.sqrt(((traj_pred[2:3] - traj_label[2:3]) ** 2).sum())
    L2_2s = np.sqrt(((traj_pred[6:7] - traj_label[6:7]) ** 2).sum())
    L2_3s = np.sqrt(((traj_pred[10:11] - traj_label[10:11]) ** 2).sum())

    L2_1s_list.append(L2_1s)
    L2_2s_list.append(L2_2s)
    L2_3s_list.append(L2_3s)

print('ST samples')
print('total number of data counted:',len(L2_3s_list))

print('1s:',np.mean(L2_1s_list))
print('2s:',np.mean(L2_2s_list))
print('3s:',np.mean(L2_3s_list))



L2_1s_list = []
L2_2s_list = []
L2_3s_list = []


for i in range(len(pred)):
    traj_pred = pred[i]
    traj_label = label[i]
    x_diff = traj_label[10]-traj_label[0]
    if not(x_diff<=-2 or x_diff>=2):
        continue
    #else:
    #    continue
    L2_1s = np.sqrt(((traj_pred[2:3] - traj_label[2:3]) ** 2).sum())
    L2_2s = np.sqrt(((traj_pred[6:7] - traj_label[6:7]) ** 2).sum())
    L2_3s = np.sqrt(((traj_pred[10:11] - traj_label[10:11]) ** 2).sum())

    L2_1s_list.append(L2_1s)
    L2_2s_list.append(L2_2s)
    L2_3s_list.append(L2_3s)

print('LR samples')
print('total number of data counted:',len(L2_3s_list))

print('1s:',np.mean(L2_1s_list))
print('2s:',np.mean(L2_2s_list))
print('3s:',np.mean(L2_3s_list))
