import json
import pickle
import ast
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
    vx = data_dict['gt_ego_lcf_feat'][0]*0.5
    vy = data_dict['gt_ego_lcf_feat'][1]*0.5
    v_yaw = data_dict['gt_ego_lcf_feat'][4]
    ax = data_dict['gt_ego_his_diff'][-1, 0] - data_dict['gt_ego_his_diff'][-2, 0]
    ay = data_dict['gt_ego_his_diff'][-1, 1] - data_dict['gt_ego_his_diff'][-2, 1]
    cx = data_dict['gt_ego_lcf_feat'][2]
    cy = data_dict['gt_ego_lcf_feat'][3]
    vhead = data_dict['gt_ego_lcf_feat'][7]*0.5
    steeling = data_dict['gt_ego_lcf_feat'][8]
    xh1 = data_dict['gt_ego_his_trajs'][0][0]
    yh1 = data_dict['gt_ego_his_trajs'][0][1]
    xh2 = data_dict['gt_ego_his_trajs'][1][0]
    yh2 = data_dict['gt_ego_his_trajs'][1][1]
    xh3 = data_dict['gt_ego_his_trajs'][2][0]
    yh3 = data_dict['gt_ego_his_trajs'][2][1]
    xh4 = data_dict['gt_ego_his_trajs'][3][0]
    yh4 = data_dict['gt_ego_his_trajs'][3][1]
    temp = [vx,vy,v_yaw,ax,ay,cx,cy,vhead,steeling,xh1,yh1,xh2,xh2,xh3,yh3,xh4,yh4]
    for i in range(len(temp)):
        temp[i] = round(temp[i],2)

    cmd_vec = data_dict['gt_ego_fut_cmd']
    right, left, forward = cmd_vec
    if right > 0:
        mission_goal = [1,0,0]
    elif left > 0:
        mission_goal = [0,0,1]
    else:
        assert forward > 0
        mission_goal = [0,1,0]
    temp += mission_goal

    res.append(temp)

res = np.array(res)

with open('val.npy', 'wb') as f:
    np.save(f,res)

