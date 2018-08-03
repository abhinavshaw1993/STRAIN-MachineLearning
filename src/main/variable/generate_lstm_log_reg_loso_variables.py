import pandas as pd
import torch
from torch.autograd import Variable
from main.definition import ROOT_DIR
from sklearn.preprocessing import normalize


def adjust_stress_values(stress_level):
    mapping = {
        1: 2,
        2: 3,
        3: 4,
        4: 1,
        5: 0
    }

    try:
        return mapping[stress_level]
    except KeyError:
        return None


def generate_lstm_log_reg_loso_variables(student_list,
                                         feature_list,
                                         restrict_seqlen=-1,
                                         is_cuda_available=False,
                                         standardize=True):
    train_feature_list = []

    for student in student_list:

        train_feature_dict = {}

        for feature in feature_list:
            # Read CSV and skip the index column and time columns.
            raw_feature_train_x = pd.read_csv(
                "{}/StudentLife Data/{}/{}_train_x.csv".format(ROOT_DIR, student, feature),
                skip_blank_lines=False).iloc[:, 2:]
            raw_feature_train_y = raw_feature_train_x.iloc[:, -1]
            raw_feature_train_y = raw_feature_train_y.apply(adjust_stress_values)
            raw_feature_train_x = raw_feature_train_x.iloc[:, :-1]

            # Indices
            raw_feature_train_y_indices = pd.read_csv(
                "{}/StudentLife Data/{}/{}_train_y_indices.csv".format(ROOT_DIR, student, feature),
                skip_blank_lines=False).iloc[:, 1]

            # If want to truncate training data
            if restrict_seqlen != -1:
                feature_indices = raw_feature_train_y_indices[:restrict_seqlen]
                last_idx = feature_indices.iloc[-1]
                feature_train_x = raw_feature_train_x.iloc[:last_idx + 1]
                feature_train_y = raw_feature_train_y.iloc[:last_idx + 1]

            else:
                feature_indices = raw_feature_train_y_indices
                feature_train_x = raw_feature_train_x
                feature_train_y = raw_feature_train_y

            np_feature_train_x = feature_train_x.as_matrix()
            np_feature_train_y = feature_train_y.as_matrix()
            np_feature_indices = feature_indices.as_matrix()

            # normalize
            if standardize:
                np_feature_train_x = normalize(np_feature_train_x)

            # Extracting shape for reshaping.
            x, y = np_feature_train_x.shape
            shape = (x, 1, y)
            np_feature_train_x = np_feature_train_x.reshape(shape)

            # Selecting only those which have a y label associated with it. rest are not required.
            np_feature_train_y = np_feature_train_y[np_feature_indices]
            # print(np_feature_train_y)
            np_feature_train_y = np_feature_train_y.reshape((np_feature_train_y.shape[0], 1, 1))

            # Initializing Input Seq , Target Seq and Indices for Train Set.
            input_seq_tensor = torch.from_numpy(np_feature_train_x)
            target_tensor = torch.from_numpy(np_feature_train_y.reshape(-1))
            indices = torch.from_numpy(np_feature_indices.reshape(-1))

            if is_cuda_available:
                input_seq_tensor = input_seq_tensor.cuda()
                target_tensor = target_tensor.cuda()
                indices = indices.cuda()

            # Generating Variables.
            input_seq = Variable(input_seq_tensor, requires_grad=False).float()
            target = Variable(target_tensor, requires_grad=False).long()
            indices = Variable(indices, requires_grad=False)

            # creating everything into a feature dict.
            train_feature_dict[feature] = (input_seq, target, indices)

        train_feature_list.append(train_feature_dict.copy())

    return train_feature_list
