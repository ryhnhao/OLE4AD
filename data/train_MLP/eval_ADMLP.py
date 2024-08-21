import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Set random seed for reproducibility
torch.manual_seed(42)

# Define input size, hidden size, and output size
input_size =  20 # replace with the actual input size
hidden_size = 512  # choose an appropriate size
output_size = 12  # number of output nodes

# Create an instance of the MLP model
model = MLP(input_size, hidden_size, output_size)

model = torch.load('ADMLP_full.pt',map_location='cpu')

model.eval()

def numpy_to_tensor(array):
    return torch.from_numpy(array).float()
# Generate example data (replace with your actual data)
with open('val.npy', 'rb') as f:
    input_data = np.load(f)
with open('val_label.npy', 'rb') as g:
    target_data = np.load(g)

# Convert input and target data to PyTorch tensors
inputs = numpy_to_tensor(input_data)
targets = numpy_to_tensor(target_data)

L2_1s_list = []
L2_2s_list = []
L2_3s_list = []

res = []
for i in range(len(inputs)):
    new_inputs = inputs[i]
    with torch.no_grad():
        predictions = model(new_inputs)
    # print(predictions)
    predictions_np = predictions.numpy()

    # Round the numbers to two decimal places
    traj_pred = np.round(predictions_np, decimals=2)
    traj_label = target_data[i]

    L2_1s = np.sqrt(((traj_pred[2:4] - traj_label[2:4]) ** 2).sum())
    L2_2s = np.sqrt(((traj_pred[6:8] - traj_label[6:8]) ** 2).sum())
    L2_3s = np.sqrt(((traj_pred[10:12] - traj_label[10:12]) ** 2).sum())

    L2_1s_list.append(L2_1s)
    L2_2s_list.append(L2_2s)
    L2_3s_list.append(L2_3s)
    
    res.append(traj_pred)

print('total number of data counted:',len(L2_1s_list))

print('1s:',np.mean(L2_1s_list))
print('2s:',np.mean(L2_2s_list))
print('3s:',np.mean(L2_3s_list))

print('1s:',np.max(L2_1s_list))
print('2s:',np.max(L2_2s_list))
print('3s:',np.max(L2_3s_list))


res = np.array(res)
with open('val_mlp_prediction.npy', 'wb') as f:
    np.save(f,res)
