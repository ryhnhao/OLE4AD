import pickle
import ndjson
import json
import tiktoken
import numpy as np
from prompt_message import system_message, generate_user_message, generate_assistant_message

def jsonl_to_dict(filename):
    data_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            token = json_obj['token']
            data_dict[token] = json_obj['detections']
    return data_dict

data = pickle.load(open('data/cached_nuscenes_info.pkl', 'rb'))
uniad_perceptions = jsonl_to_dict('data/detection_motion_result_trainval.jsonl')
split = json.load(open('data/split.json', 'r'))

train_tokens = split["train"]
val_tokens = split["val"]
num_train_samples = len(train_tokens)
train_ratio = 1

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

num_language_tokens = 0
num_system_tokens = 0
num_user_tokens = 0
num_assistant_tokens = 0

traj_only = False

train_messages = []
for token_i, token in enumerate(train_tokens):
    if token_i >= train_ratio * num_train_samples:
        break

    uniad_perception = uniad_perceptions[token] # list of dicts
    uniad_boxes, uniad_names, uniad_trajs = [], [], []
    for obj in uniad_perception:
        uniad_names.append(obj['name'])
        uniad_boxes.append(obj['box'])
        box_center = np.array(obj['box'][:2]) # (2)
        full_traj = np.array(obj['traj'][:6]) # [6, 2]
        rel_traj = full_traj - box_center[None,:]
        rel_traj = np.concatenate([np.zeros((1,2)), rel_traj], axis=0) # [7, 2]
        rel_diff_traj = rel_traj[1:] - rel_traj[:-1] # [6, 2]
        uniad_trajs.append(rel_diff_traj)
    if len(uniad_trajs) == 0:
        data[token]['gt_agent_fut_trajs'] = np.zeros((0,6,2)) # [num_objs, 6, 2]
        data[token]['gt_boxes'] = np.zeros((0,9))
        data[token]['gt_names'] = np.array([])
    else:
        data[token]['gt_boxes'] = np.array(uniad_boxes)
        data[token]['gt_names'] = np.array(uniad_names)
        data[token]['gt_agent_fut_trajs'] = np.stack(uniad_trajs, axis=0) # [num_objs, 6, 2]
    data[token]['gt_agent_fut_masks'] = np.ones((len(uniad_boxes), 6)) # [num_objs, 6]

    user_message = generate_user_message(data, token)
    assitant_message = generate_assistant_message(data, token, traj_only=traj_only)
    if len(assitant_message.split("\n")) > 6:
        print()
        print(token)
        print(system_message)
        print(user_message)
        print(assitant_message)
    num_language_tokens += len(encoding.encode(system_message))
    num_system_tokens += len(encoding.encode(system_message))
    num_language_tokens += len(encoding.encode(user_message))
    num_user_tokens += len(encoding.encode(user_message))
    num_language_tokens += len(encoding.encode(assitant_message))
    num_assistant_tokens += len(encoding.encode(assitant_message))


    train_message = {"messages": 
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}, 
            {"role": "assistant", "content": assitant_message}
        ]
    }
    train_messages.append(train_message)

print("#### Cost Summarization ####")
print(f"Number of system tokens: {num_system_tokens}")
print(f"Number of user tokens: {num_user_tokens}")
print(f"Number of assistant tokens: {num_assistant_tokens}")
print(f"Number of total tokens: {num_language_tokens}")

with open("data/train_uniad.json", "w") as f:
    ndjson.dump(train_messages, f)