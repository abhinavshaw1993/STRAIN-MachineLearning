# Import all required packages.
import torch
import torch.nn as nn


# Define ModelClass
class Strain(nn.Module):

    def __init__(self, input_size_list, num_classes=5, hidden_dim = 100):
        super(Strain, self).__init__()

        # since input_size_list contains a list of sizes of each feature, We initialize those many RNN cells, 1 for each
        # feature.
        self.rnns = nn.ModuleList([nn.LSTM(input_size, hidden_dim, 1) for input_size in input_size_list])
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
        y_out = self.linear(y_out)

        return y_out


# Function to set weigths.
def weights_init(m):
    classname = m.__class__.__name__
#     if classname.find('LSTM') != -1:
#         m.weight.data.normal_(0.0, 0.02)
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
