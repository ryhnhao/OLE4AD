

import numpy as np
import json

f = open('traj_pdm_open.json','rb')
d = json.load(f)


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
            center[1] = -center[1]
        pred_traj = np.delete(pred_traj,2,1)
        gt_traj = np.delete(gt_traj,2,1)
        pred_traj -= center
        gt_traj -= center
        pred_traj_list.append(pred_traj.reshape(-1)[:8])
        gt_traj_list.append(gt_traj.reshape(-1)[:8])

L2_1s_list = []
L2_2s_list = []
L2_3s_list = []


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=0, n_init=1).fit(gt_traj_list)
labels = kmeans.cluster_centers_

np.save('kmeans_center.npy',labels)


