from main.variable.generate_lstm_log_reg_loso_variables import generate_lstm_log_reg_loso_variables
import torch


def get_inputs(student_list, feature_list, restrict_seqlen=5, standardize=False):
    # Getting the input and generating respective sequences.
    train_feature_list = generate_lstm_log_reg_loso_variables(student_list=student_list,
                                                              feature_list=feature_list,
                                                              restrict_seqlen=restrict_seqlen,
                                                              is_cuda_available=torch.cuda.is_available(),
                                                              standardize=standardize)

    data = []

    for train_feature_dict in train_feature_list:
        # return
        input_size_list = []
        index_list = []
        input_list = []
        target = None

        for key in train_feature_dict.keys():
            input_seq, target, indices = train_feature_dict[key]
            input_size_list.append(input_seq.shape[2])
            input_list.append(input_seq)
            index_list.append(indices)

        data.append((input_list, input_size_list, index_list, target))

    return data
