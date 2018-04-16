import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable

def generate_variables(feature_list = [
    "activity_details",
    # "dinning_details" ,
    "sms_details",
    "audio_details",
    "conversation_details",
    "dark_details",
    "phonecharge_details",
    "phonelock_details",
    "gps_details"],
    restrict_seqlen=10,
    cuda=True):


    feature_dict = {}

    for feature in feature_list:

        # Read CSV and skip the time columns.
        raw_feature_train_x = pd.read_csv("Data/"+feature+"_train_x.csv", skip_blank_lines=False).iloc[:,1:]
        raw_feature_train_y = pd.read_csv("Data/"+feature+"_train_y.csv", skip_blank_lines=False)
        raw_feature_train_y["stress_level"] += -1
        raw_feature_train_y_indices = pd.read_csv("Data/"+feature+"_train_y_indices.csv", skip_blank_lines=False)

        # If want to truncate training data
        feature_indices = raw_feature_train_y_indices[:restrict_seqlen]

        # Selecting new subset of data.
        last_idx = feature_indices.iloc[-1,0]
        feature_train_x = raw_feature_train_x.iloc[:last_idx+1]
        feature_train_y = raw_feature_train_y.iloc[:last_idx+1]

        # Finding Indexes for the target outputs. Then we convert it into into a tensor.
        # Converting to numpy array.
        np_feature_indices = feature_indices.as_matrix()

        # Train X
        np_feature_train_x =  feature_train_x.as_matrix()

        # Extracting shape for reshaping.
        x, y = np_feature_train_x.shape
        shape = (x, 1, y)
        np_feature_train_x = np_feature_train_x.reshape(shape)

        # Train Y
        np_feature_train_y = feature_train_y.as_matrix()
        np_feature_train_y = np_feature_train_y[np_feature_indices]
        np_feature_train_y = np_feature_train_y.reshape((np_feature_train_y.shape[0],1,1))

        #Initializing Input Seq , Target Seq and Indices.
        input_seq_tensor = torch.from_numpy(np_feature_train_x)
        target_tensor = torch.from_numpy(np_feature_train_y.reshape(-1))
        indices = torch.from_numpy(np_feature_indices.reshape(-1))

        if (cuda):
            input_seq_tensor = input_seq_tensor.cuda()
            target_tensor = target_tensor.cuda()
            indices = indices.cuda()

        # Generating Variables.
        input_seq = Variable(input_seq_tensor, requires_grad=False).float()
        target = Variable(target_tensor, requires_grad=False).long()
        indices = Variable(indices, requires_grad=False)

        # creating everything into a feature dict.
        feature_dict[feature] = (input_seq, target, indices)

    return feature_dict
