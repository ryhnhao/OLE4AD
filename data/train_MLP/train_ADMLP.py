import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

# Define the two-layer MLP model
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
#torch.manual_seed(42)

# Define input size, hidden size, and output size
input_size =  20 # replace with the actual input size
hidden_size = 512  # choose an appropriate size
output_size = 12  # number of output nodes

# Create an instance of the MLP model
model = MLP(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.L1Loss()
# criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(),lr=4e-6)
scheduler = MultiStepLR(optimizer,[25],gamma=0.1)
# Convert numpy arrays to PyTorch tensors
def numpy_to_tensor(array):
    return torch.from_numpy(array).float()

# Generate example data (replace with your actual data)
with open('train.npy', 'rb') as f:
    input_data = np.load(f)
with open('train_label.npy', 'rb') as g:
    target_data = np.load(g)

# Convert input and target data to PyTorch tensors
inputs = numpy_to_tensor(input_data)
targets = numpy_to_tensor(target_data)

# Create a DataLoader for batching
batch_size = 32
dataset = TensorDataset(inputs, targets)
print(len(dataset))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop with batching
num_epochs = 30  # adjust as needed
model = model.cuda()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i,(batch_inputs, batch_targets) in enumerate(dataloader):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_inputs.cuda())

        # Calculate the loss
        loss = criterion(outputs, batch_targets.cuda())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()
    mean_epoch_loss = epoch_loss / len(dataloader)
    print(f'Epoch [{epoch}], Loss: {mean_epoch_loss:.4f}')

torch.save(model,'./ADMLP_full.pt')


