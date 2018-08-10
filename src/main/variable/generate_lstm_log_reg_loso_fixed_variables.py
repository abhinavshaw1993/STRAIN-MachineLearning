import pandas as pd
import numpy as np
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
            input_seq, inpute_mask, target = np.load(
                "{}/StudentLife Data/{}/{}_train_x.npz".format(ROOT_DIR, student, feature),
                )

        # normalize
        if standardize:
            input_seq = normalize(input_seq)


        print(input_seq.shape)















