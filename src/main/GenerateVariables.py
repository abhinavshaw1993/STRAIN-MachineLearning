import pandas as pd
import torch
from torch.autograd import Variable
import math
import numpy as np


def generate_variables(feature_list=[
    "activity_details",
    # "dinning_details" ,
    "sms_details",
    "audio_details",
    "conversation_details",
    "dark_details",
    "phonecharge_details",
    "phonelock_details",
    "gps_details"],
    restrict_seqlen=-1,
    is_cuda_available=True,
    val_set_size=0.3):


    train_feature_dict = {}
    val_feature_dict = {}

    for feature in feature_list:

        # Read CSV and skip the time columns.
        raw_feature_train_x = pd.read_csv("Data/" + feature + "_train_x.csv", skip_blank_lines=False).iloc[:, 1:]
        raw_feature_train_y = pd.read_csv("Data/" + feature + "_train_y.csv", skip_blank_lines=False)
        raw_feature_train_y["stress_level"] += -1
        raw_feature_train_y_indices = pd.read_csv("Data/" + feature + "_train_y_indices.csv", skip_blank_lines=False)

        # splitting data into test and train splits. Keeping 30% of labels for Val.
        total_y_labels = len(raw_feature_train_y_indices)
        val_samples = total_y_labels * val_set_size
        val_samples = math.floor(val_samples)

        # If want to truncate training data
        if restrict_seqlen != -1:
            feature_indices = raw_feature_train_y_indices[:min(restrict_seqlen, total_y_labels - val_samples)]
        else:
            feature_indices = raw_feature_train_y_indices[:total_y_labels - val_samples]

        # Selecting new subset of data.
        last_idx = feature_indices.iloc[-1, 0]
        feature_train_x = raw_feature_train_x.iloc[:last_idx + 1]
        feature_train_y = raw_feature_train_y.iloc[:last_idx + 1]

        feature_val_indices = raw_feature_train_y_indices[total_y_labels - val_samples:]
        feature_val_start_index = feature_val_indices.iloc[0, 0]

        feature_val_x = raw_feature_train_x.iloc[feature_val_start_index + 1:]
        feature_val_y = raw_feature_train_y  # This is kept different because the the indices will get messed up later.

        # Finding Indexes for the target outputs. Then we convert it into into a tensor.
        # Converting to numpy array.
        np_feature_indices = feature_indices.as_matrix()
        np_feature_val_indices = feature_val_indices.as_matrix()

        # Train X
        np_feature_train_x = feature_train_x.as_matrix()
        np_feature_val_x = feature_val_x.as_matrix()

        # Extracting shape for reshaping.
        x, y = np_feature_train_x.shape
        shape = (x, 1, y)
        np_feature_train_x = np_feature_train_x.reshape(shape)

        # Validation set.
        x, y = np_feature_val_x.shape
        shape = (x, 1, y)
        np_feature_val_x = np_feature_val_x.reshape(shape)

        # Train Y
        np_feature_train_y = feature_train_y.as_matrix()
        np_feature_val_y = feature_val_y.as_matrix()

        # Selecting only those which have a y label associated with it.
        np_feature_train_y = np_feature_train_y[np_feature_indices]
        np_feature_train_y = np_feature_train_y.reshape((np_feature_train_y.shape[0], 1, 1))

        np_feature_val_y = np_feature_val_y[np_feature_val_indices]
        np_feature_val_y = np_feature_val_y.reshape((np_feature_val_y.shape[0], 1, 1))

        # Initializing Input Seq , Target Seq and Indices for Train Set.
        input_seq_tensor = torch.from_numpy(np_feature_train_x)
        target_tensor = torch.from_numpy(np_feature_train_y.reshape(-1))
        indices = torch.from_numpy(np_feature_indices.reshape(-1))

        # Initializing Input Seq , Target Seq and Indices for Val Set.
        val_input_seq_tensor = torch.from_numpy(np_feature_val_x)
        val_target_tensor = torch.from_numpy(np_feature_val_y.reshape(-1))
        val_indices = torch.from_numpy(np_feature_val_indices.reshape(-1))

        ############################# FInal Genereation ##################################

        if is_cuda_available:
            input_seq_tensor = input_seq_tensor.cuda()
            target_tensor = target_tensor.cuda()
            indices = indices.cuda()
            val_input_seq_tensor = val_input_seq_tensor.cuda()
            val_target_tensor = val_target_tensor.cuda()
            val_indices = val_indices.cuda()

        # Generating Variables.
        input_seq = Variable(input_seq_tensor, requires_grad=False).float()
        target = Variable(target_tensor, requires_grad=False).long()
        indices = Variable(indices, requires_grad=False)

        val_input_seq = Variable(val_input_seq_tensor, requires_grad=False).float()
        val_target = Variable(val_target_tensor, requires_grad=False).long()
        val_indices = Variable(val_indices, requires_grad=False)

        # creating everything into a feature dict.
        train_feature_dict[feature] = (input_seq, target, indices)
        val_feature_dict[feature] = (val_input_seq, val_target, val_indices)

    return train_feature_dict, val_feature_dict
