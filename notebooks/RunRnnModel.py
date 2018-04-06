# Import all required packages.

from IPython.display import display
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from DataProcessorScript import generate_variables
import sys
# Define ModelClass

class Strain(nn.Module):

    def __init__(self, num_features, input_seq_len_list, input_size=2, num_classes=5, hidden_dim = 100):
        super(Strain, self).__init__()

        time_steps = input_seq_len_list[0]
        batch_size = 1
        self.rnn1 = nn.LSTM(input_size, hidden_dim, 1) # 1 is number of layers.
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, indices):
        y_out, _ = self.rnn1(x)
        y_out = torch.index_select(y_out, 0, indices)
        y_out = y_out.squeeze(2).squeeze(1) #y_out.view(y_out.shape[0], -1)
        y_out = self.linear(y_out)

        return y_out

#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features

# input_seq
# target
# indices

input_seq, target, indices = generate_variables(feature_list=["activity_details"], restrict_seqlen=10)["activity_details"]
print('input_seq')
print(input_seq)
print('target')
print(target)
print('indices')
print(indices)
# declaring Network.
net = Strain(num_features=1, input_seq_len_list=[len(input_seq)])
criterion = nn.CrossEntropyLoss(size_average=True)
optimizer = optim.SGD(net.parameters(), 0.001)

# Train the network

for i in range(2):
    print('i')
    print(i)
    optimizer.zero_grad()
    y_hat = net.forward(input_seq, indices=indices)
    print('y_hat')
    print(y_hat)
    print('target')
    print(target)
    loss = criterion(y_hat, target)
    print('loss')
    print(loss)
    sys.stdout.flush()
    loss.backward()
    optimizer.step()
    print(loss)

print(y_hat)

print(target)
