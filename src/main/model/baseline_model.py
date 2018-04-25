# Import all required packages.
import torch.nn as nn


# Define ModelClass
class StrainBaseline(nn.Module):
    """
    This class
    """

    def __init__(self, num_features, num_classes=5):
        super(StrainBaseline, self).__init__()

        # since input_size_list contains a list of sizes of each feature, We initialize those many RNN cells, 1 for each
        # feature.
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, input):
        # print("Concatenated Feature vectors", y_out)
        if (input != input).any():
            print("null exists")

        y_out = self.linear(input)

        return y_out


# Function to set weigths.
def weights_init(m):
    classname = m.__class__.__name__
    #     if classname.find('LSTM') != -1:
    #         m.weight.data.normal_(0.0, 0.02)
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
