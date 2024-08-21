import json
import pickle
import numpy as np
data = pickle.load(open('cached_nuscenes_info.pkl', 'rb'))
split = json.load(open('/GPT-Driver/data/split.json', 'r'))
train_tokens = split["val"]

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


res = []
for token in train_tokens:
    data_dict = data[token]
    x1 = data_dict['gt_ego_fut_trajs'][1][0]
    x2 = data_dict['gt_ego_fut_trajs'][2][0]
    x3 = data_dict['gt_ego_fut_trajs'][3][0]
    x4 = data_dict['gt_ego_fut_trajs'][4][0]
    x5 = data_dict['gt_ego_fut_trajs'][5][0]
    x6 = data_dict['gt_ego_fut_trajs'][6][0]
    y1 = data_dict['gt_ego_fut_trajs'][1][1]
    y2 = data_dict['gt_ego_fut_trajs'][2][1]
    y3 = data_dict['gt_ego_fut_trajs'][3][1]
    y4 = data_dict['gt_ego_fut_trajs'][4][1]
    y5 = data_dict['gt_ego_fut_trajs'][5][1]
    y6 = data_dict['gt_ego_fut_trajs'][6][1]
    temp = [x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6]

    for i in range(len(temp)):
        temp[i] = round(temp[i],2)
    res.append(temp)

res = np.array(res)

with open('val_label.npy', 'wb') as f:
    np.save(f,res)

