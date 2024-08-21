import json
import pickle
import ast
import numpy as np


split = json.load(open('/data/GPT-Driver/split.json', 'r'))
val_tokens = split["val"]


def process_data(data_path):
    f = open(data_path)
    user = []
    assistant = []
    for line in f:
        l = json.loads(line)
        for m in l['messages']:
            if m['role']=='user':
                user.append({'user': m['content']})
            if m['role']=='assistant':
                assistant.append(m['content'])
    return user,assistant


user, assistant = process_data('/GPT-Driver/data/val.json')



L2_1s_list = []
L2_2s_list = []
L2_3s_list = []

res = {}
uniad_pred = []

uniad_results_f = open('uniad_traj.pkl','rb')
uniad_results = pickle.load(uniad_results_f)

x = 0
y = 0

for i in range(len(assistant)):

    traj_label = assistant[i].split("\n")[-1]
    try:
        traj_label = ast.literal_eval(traj_label)
        traj_label = np.array(traj_label)
    except:
        print('invalid_label:', traj_label)
        continue
    traj_pred = uniad_results[val_tokens[i]].numpy()

    L2_1s = np.sqrt(((traj_pred[1:2,0:2] - traj_label[1:2,0:2]) ** 2).sum())
    L2_2s = np.sqrt(((traj_pred[3:4,0:2] - traj_label[3:4, 0:2]) ** 2).sum())
    L2_3s = np.sqrt(((traj_pred[5:6, 0:2] - traj_label[5:6, 0:2]) ** 2).sum())

    res[val_tokens[i]] = {'traj_pred':traj_pred,'traj_label':traj_label}

    L2_1s_list.append(L2_1s)
    L2_2s_list.append(L2_2s)
    L2_3s_list.append(L2_3s)

    uniad_pred.append(traj_pred.reshape(12))

print('total number of data counted:',len(L2_3s_list))

print('1s:',np.mean(L2_1s_list))
print('2s:',np.mean(L2_2s_list))
print('3s:',np.mean(L2_3s_list))

# print('1s:',np.max(L2_1s_list))
# print('2s:',np.max(L2_2s_list))
# print('3s:',np.max(L2_3s_list))

with open('output_data.pkl', 'wb') as f:
    pickle.dump(res, file=f)

uniad_pred = np.array(uniad_pred)
with open('uniad_prediction.npy', 'wb') as f1:
    np.save(f1,uniad_pred)

