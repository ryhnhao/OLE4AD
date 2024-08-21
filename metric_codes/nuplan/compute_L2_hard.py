

import numpy as np
import json

f = open('traj_pdm_closed.json','rb')
d = json.load(f)
d = {key: d[key] for key in sorted(d.keys())}


f_scene = open('scene.json','rb')
scene = json.load(f_scene)

center_list = np.load('kmeans_center.npy')
keep_stationary = np.zeros((1,12))
center_list = np.vstack((center_list,keep_stationary))

pred_traj_list = []
gt_traj_list = []
gt_check_list = []
for token in d:
    hist = d[token]['simulation_his']
    for i in range(len(hist)):
        center = np.array(hist[i]['ego_centor'])
        pred_traj = np.array(hist[i]['pred_traj'])
        gt_traj = np.array(hist[i]['gt_traj'])
        if pred_traj.shape!=gt_traj.shape:
            continue
        if gt_traj[0][2]<0:
            for j in range(len(pred_traj)):
                pred_traj[j][1] = -pred_traj[j][1]
                gt_traj[j][1] = -gt_traj[j][1]
                pred_traj[j][0] = -pred_traj[j][0]
                gt_traj[j][0] = -gt_traj[j][0]
            center[1] = -center[1]
            center[0] = -center[0]
        pred_traj = np.delete(pred_traj,2,1)
        gt_traj = np.delete(gt_traj,2,1)
        pred_traj -= center
        gt_traj -= center
        pred_traj_list.append(pred_traj.reshape(-1))
        gt_traj_list.append(gt_traj.reshape(-1))

L2_1s_list = []
L2_2s_list = []
L2_3s_list = []

cnt = 0
correct = [0 for i in range(21)]

pred = [0 for i in range(21)]
label = [0 for i in range(21)]

easy_list = np.load('easy_list.npy')
for i in range(len(pred_traj_list)):
    if i in easy_list:
        continue
    traj_pred = pred_traj_list[i]
    traj_label = gt_traj_list[i]
    
    pred_decision = np.argmin(np.linalg.norm(center_list - traj_pred,axis=1))
    label_decision = np.argmin(np.linalg.norm(center_list - traj_label,axis=1))

    if pred_decision == label_decision:
        correct[label_decision]+=1
    pred[pred_decision]+=1
    label[label_decision]+=1

    L2_1s = np.sqrt(((traj_pred[2:4] - traj_label[2:4]) ** 2).sum())
    L2_2s = np.sqrt(((traj_pred[6:8] - traj_label[6:8]) ** 2).sum())
    L2_3s = np.sqrt(((traj_pred[10:12] - traj_label[10:12]) ** 2).sum())


    L2_1s_list.append(L2_1s)
    L2_2s_list.append(L2_2s)
    L2_3s_list.append(L2_3s)

print('number of cases failed:',cnt)
print(pred)
print(label)
print('num correct:',correct)
print(sum(correct))
print('total number of data counted:',len(L2_3s_list))

print('1s:',np.mean(L2_1s_list))
print('2s:',np.mean(L2_2s_list))
print('3s:',np.mean(L2_3s_list))





