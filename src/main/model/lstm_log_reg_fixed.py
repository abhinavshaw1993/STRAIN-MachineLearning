# Import all required packages.
import torch
import torch.nn as nn


# Define ModelClass
class Strain_log_regression(nn.Module):

    def __init__(self, input_size_list, num_classes=5, hidden_dim=20):
        self.hidden_dim = hidden_dim
        super(Strain_log_regression, self).__init__()

        # since input_size_list contains a list of sizes of each feature, We initialize those many RNN cells, 1 for each
        # feature.
        self.rnns = nn.ModuleList([nn.LSTM(input_size, hidden_dim, 1, batch_first=True) for input_size in input_size_list])
        self.linear = nn.Linear(len(input_size_list)*hidden_dim, num_classes)

    def forward(self, input_list, mask_list, target_len):

        if len(input_list) != len(mask_list):
            print("mistamatching index and targets")
            return

        output_list = []

        for idx in range(len(mask_list)):
            self.rnns[idx].flatten_parameters()
            y_out, _ = self.rnns[idx](input_list[idx])

            mask = torch.unsqueeze(mask_list[idx], dim=2)
            x, y, z = mask.shape
            mask = mask.expand(x, y, self.hidden_dim)

            y_out = torch.masked_select(y_out, mask)
            y_out = y_out.view(target_len, self.hidden_dim)

            output_list.append(y_out)

        y_out = torch.cat(output_list, dim=1)
        # print("Y Out shape:", y_out.shape)
        # print("Concatenated Feature vectors", y_out)
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
