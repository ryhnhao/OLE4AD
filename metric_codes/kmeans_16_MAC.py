import numpy as np
from sklearn.cluster import KMeans

pred_mlp = np.load('data/val_mlp_prediction.npy')
pred_uniad = np.load('data/uniad_prediction.npy')
pred_llama = np.load('data/llama_prediction.npy')
pred_llama_no_ego = np.load('data/llama_no_ego_prediction.npy')
pred_vad_ego = np.load('data/vad_ego_prediction.npy')
pred_vad_no_ego = np.load('data/vad_no_ego_prediction.npy')
val_input = np.load('data/val.npy')

val_label = np.load('data/val_label.npy')



kmeans = KMeans(n_clusters=15, random_state=0, n_init=1).fit(val_label)

labels = kmeans.labels_

center_list = kmeans.cluster_centers_
keep_stationary = np.zeros((1,12))
center_list = np.vstack((center_list,keep_stationary))
# np.save('center_list_16.npy',center_list)

labels = []
meta_list = [[] for i in range(16)]
for i in range(len(val_label)):
    traj_label = val_label[i]
    label_decision = np.argmin(np.linalg.norm(center_list - traj_label,axis=1))
    labels.append(label_decision)

easy_list = []
hard_list = []
for i in range(len(pred_llama)):
    traj_pred = pred_llama[i]

    pred_decision = np.argmin(np.linalg.norm(center_list - traj_pred,axis=1))

    if pred_decision == labels[i]:
        easy_list.append(i)
    else:
        hard_list.append(i)
        meta_list[labels[i]].append(i)


def check_correct(pred_traj_list):
    correct_list = [0 for i in range(16)]
    cnt = 0
    for i in range(len(pred_traj_list)):
        # if i in easy_list:
        #     continue
        traj_pred = pred_traj_list[i]

        pred_decision = np.argmin(np.linalg.norm(center_list - traj_pred,axis=1))
        if pred_decision == labels[i]:
            cnt += 1
            correct_list[labels[i]]+=1
    print(correct_list)
    return cnt


print('uniad')
uniad = check_correct(pred_uniad)
print('number of correct samples')
print(uniad)
print()

print('llama')
llama = check_correct(pred_llama)
print('number of correct samples')
print(llama)
print()

print('llama_no_ego')
llama_no_ego = check_correct(pred_llama_no_ego)
print('number of correct samples')
print(llama_no_ego)
print()

print('vad_ego')
vad_ego = check_correct(pred_vad_ego)
print('number of correct samples')
print(vad_ego)
print()

print('vad_no_ego')
vad_no_ego = check_correct(pred_vad_no_ego)
print('number of correct samples')
print(vad_no_ego)
print()

print('mlp')
mlp = check_correct(pred_mlp)
print('number of correct samples')
print(mlp)
