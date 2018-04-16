# Import all required packages.
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from GenerateVariables import generate_variables
import sys

# Set Cuda.
cuda = False

feature_list = [
    "activity_details",
    "sms_details",
    "audio_details",
    "conversation_details",
    "dark_details",
    "phonecharge_details",
    "phonelock_details",
    "gps_details"]

# Getting the input and generating respective sequences.
feature_dict = generate_variables(feature_list=feature_list, restrict_seqlen=2, cuda=cuda)

# return
input_size_list = []
index_list = []
input_list = []

for key in feature_dict.keys():
    input_seq, target, indices = feature_dict[key]
    input_size_list.append(input_seq.shape[2])
    input_list.append(input_seq)
    index_list.append(indices)

# Define ModelClass
class Strain(nn.Module):

    def __init__(self, input_size_list, num_classes=5, hidden_dim = 100):
        super(Strain, self).__init__()

        # since input_size_list contains a list of sizes of each feature, We initialize those many RNN cells, 1 for each feature.
        self.rnns = nn.ModuleList([nn.LSTM(input_size, hidden_dim, 1) for input_size in input_size_list])
        self.batch_norm = nn.BatchNorm2d(len(input_size_list)*100)
        self.linear = nn.Linear(len(input_size_list)*100, num_classes)

    def forward(self, input_list, index_list):

        if len(input_list) != len(index_list):
            print("mistamatching index and targets")
            return

        output_list = []

        for idx, indices in enumerate(index_list):
            y_out, _ = self.rnns[idx](input_list[idx])
            y_out = torch.index_select(y_out, 0, index_list[idx])
            y_out = y_out.view(y_out.shape[0], -1)
            output_list.append(y_out)

        y_out = torch.cat(output_list, dim=1)
        y_out = self.batch_norm(y_out)
        y_out = self.linear(y_out)

        return y_out

# Function to set weigths.
def weights_init(m):
    classname = m.__class__.__name__
#     if classname.find('LSTM') != -1:
#         m.weight.data.normal_(0.0, 0.02)
    if classname.find('Linear') != -1:
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)

# declaring Network.
net = Strain(input_size_list=input_size_list)

if cuda:
    net = net.cuda()

net.apply(weights_init)
criterion = nn.CrossEntropyLoss(size_average=True)
optimizer = optim.SGD(net.parameters(), 0.001)

# Train the network
for i in range(10):
    optimizer.zero_grad()
    y_hat = net.forward(input_list, index_list)
    loss = criterion(y_hat, target)
    loss.backward()
    optimizer.step()
    print(loss.data)
