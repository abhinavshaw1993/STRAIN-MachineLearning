import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import math
from sklearn.preprocessing import normalize


def generate_baseline_variables(feature_list=[
    "activity_details",
    # "dinning_details" ,
    "sms_details",
    "audio_details",
    "conversation_details",
    "dark_details",
    "phonecharge_details",
    "phonelock_details",
    "gps_details"],
        val_set_size=.3,
        is_cuda_available=False):
    # Dict Initialization.
    train_feature_dict = {}
    val_feature_dict = {}
    train_x_list = []
    val_x_list = []
    for feature in feature_list:

        # Read CSV and skip the time columns.
        raw_feature_train_x = pd.read_csv("data/" + feature + "_train_x.csv", skip_blank_lines=False).iloc[:, 1:]
        raw_feature_train_y = pd.read_csv("data/" + feature + "_train_y.csv", skip_blank_lines=False)
        # to bring the values from 0-4.
        raw_feature_train_y["stress_level"] += -1
        raw_feature_train_y_indices = pd.read_csv("data/" + feature + "_train_y_indices.csv", skip_blank_lines=False)

        # Finding Last Valid index.
        last_valid_index = raw_feature_train_y_indices.iloc[-1, :].values
        last_valid_index = int(last_valid_index)

        # Truncating the raw_x features for which y values do not exist.
        raw_feature_train_x = raw_feature_train_x.iloc[:last_valid_index + 1, :]

        # Hardcore indexing, to convert single index to multi so that max, min and avg can be taken easily.
        # Train set.
        list_a = []
        list_b = raw_feature_train_x.index.values
        feature_indices_list = raw_feature_train_y_indices['indices'].values

        for idx in feature_indices_list:
            if len(list_a) == 0:
                list_a += [idx for k in range(0, idx + 1)]
            else:
                list_a += [idx for k in range(len(list_a) - 1, idx)]

        index_keys = [
            np.array(list_a),
            np.array(list_b)
        ]

        raw_feature_train_x.set_index(keys=index_keys, inplace=True)

        # Colapsing Multindex to fin min, max and mean of the seq.
        raw_feature_train_x_min = raw_feature_train_x.min(level=0)
        raw_feature_train_x_max = raw_feature_train_x.max(level=0)
        raw_feature_train_x_mean = raw_feature_train_x.mean(level=0)

        raw_feature_train_x = pd.concat([raw_feature_train_x_min,
                                         raw_feature_train_x_max.iloc[:, 1:],
                                         raw_feature_train_x_mean.iloc[:, 1:]],
                                        axis=1,
                                        ignore_index=True)

        # splitting data into test and train splits. Keeping 30% of labels for Val.
        total_y_labels = len(raw_feature_train_y_indices)
        val_samples = total_y_labels * val_set_size
        val_samples = math.floor(val_samples)

        # Selecting new subset of data.
        feature_train_x = raw_feature_train_x.iloc[:total_y_labels - val_samples + 1]
        feature_train_y = raw_feature_train_y.dropna().iloc[:total_y_labels - val_samples + 1]

        feature_val_x = raw_feature_train_x.iloc[total_y_labels - val_samples + 1:]
        feature_val_y = raw_feature_train_y.dropna().iloc[total_y_labels - val_samples + 1:]

        train_x_list.append(feature_train_x)
        val_x_list.append(feature_val_x)

    # removing extra student_id columns.
    # train
    first_feature = train_x_list[0]
    student_id_col = first_feature.iloc[:, 0]
    final_list = [feature.iloc[:, 1:] for feature in train_x_list]
    final_list.insert(0, student_id_col)

    # resetting indices for each feature.
    for i in range(len(final_list)):
        final_list[i].reset_index(drop=True, inplace=True)

    train_x = pd.concat(final_list, axis=1, ignore_index=True)

    # val
    first_feature = val_x_list[0]
    student_id_col = first_feature.iloc[:, 0]
    final_list = [feature.iloc[:, 1:] for feature in val_x_list]
    final_list.insert(0, student_id_col)

    # resetting indices for each feature.
    for i in range(len(final_list)):
        final_list[i].reset_index(drop=True, inplace=True)

    val_x = pd.concat(final_list, axis=1, ignore_index=True)

    train_target = feature_train_y
    val_target = feature_val_y

    np_train_x = train_x.as_matrix()
    np_val_x = val_x.as_matrix()

    # normalizing the data.
    np_train_x = normalize(np_train_x)
    np_val_x = normalize(np_val_x)

    np_train_target = train_target.as_matrix()
    np_val_target = val_target.as_matrix()

    tensor_train_x = torch.from_numpy(np_train_x)
    tensor_val_x = torch.from_numpy(np_val_x)
    tensor_train_target = torch.from_numpy(np_train_target)
    tensor_val_target = torch.from_numpy(np_val_target)

    if is_cuda_available:
        tensor_train_x = tensor_train_x.cuda()
        tensor_val_x = tensor_val_x.cuda()
        tensor_train_target = tensor_train_target.cuda()
        tensor_val_target = tensor_val_target.cuda()

    train_input_seq = Variable(tensor_train_x, requires_grad=False).float()
    train_target = Variable(tensor_train_target, requires_grad=False).long()
    val_input_seq = Variable(tensor_val_x, requires_grad=False).float()
    val_target = Variable(tensor_val_target, requires_grad=False).long()

    return train_input_seq, train_target, val_input_seq, val_target
