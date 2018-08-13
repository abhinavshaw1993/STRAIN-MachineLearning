import numpy as np
import torch
from torch.autograd import Variable
from main.definition import ROOT_DIR

def generate_lstm_log_reg_loso_fixed_variables(  feature_list,
                                                 student_list=-1,
                                                 restrict_seqlen=-1,
                                                 is_cuda_available=False,
                                                 standardize=True):
    train_feature_list = []

    if student_list == -1:
        import os
        student_list = os.listdir("{}/{}".format(ROOT_DIR, "StudentLife Data"))
        student_list = [_ for _ in student_list if "student" in _]

    # import os
    # student_list = os.listdir("{}/{}".format(ROOT_DIR, "StudentLife Data"))
    # student_list = [_ for _ in student_list if "student" in _]

    print("Student List:", student_list)

    for student in student_list:

        train_feature_dict = {}

        # Skip Student 0 for bad data
        if student == "student 0":
            continue

        for feature in feature_list:

            # print("{}/StudentLife Data/{}/{}_train_x.npz".format(ROOT_DIR, student, feature))
            # Read CSV and skip the index column and time columns.
            numpy_array = np.load(
                "{}/StudentLife Data/{}/{}_train_x.npz".format(ROOT_DIR, student, feature),
                )

            input_seq = numpy_array['input_seq']
            input_mask = numpy_array['mask']
            target = numpy_array['target']

            if restrict_seqlen != -1:
                input_seq = input_seq[:restrict_seqlen]
                input_mask = input_mask[:restrict_seqlen]
                target = target[:np.sum(input_mask)]

            # Initializing Input Seq , Target Seq and Indices for Train Set.
            input_seq_tensor = torch.from_numpy(input_seq)
            target_tensor = torch.from_numpy(target.reshape(-1))
            input_mask_tensor = torch.from_numpy(input_mask)
            input_mask_tensor = input_mask_tensor.byte()
            # print("Mask Tensor Shape:", input_mask_tensor.shape)

            if is_cuda_available:
                input_seq_tensor = input_seq_tensor.cuda()
                target_tensor = target_tensor.cuda()
                input_mask_tensor = input_mask_tensor.cuda()

            # Generating Variables.
            input_seq_final = Variable(input_seq_tensor, requires_grad=False).float()
            target_final = Variable(target_tensor, requires_grad=False).long()
            input_mask_final = Variable(input_mask_tensor, requires_grad=False)

            # creating everything into a feature dict.
            train_feature_dict[feature] = (input_seq_final, target_final, input_mask_final)

        train_feature_list.append(train_feature_dict.copy())

    return train_feature_list
