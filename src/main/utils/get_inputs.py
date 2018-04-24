from main.utils.generate_variables import generate_variables
import torch

def get_inputs(restrict_seqlen=5):
    # Set Cuda.

    feature_list = [
        "activity_details",
        "sms_details",
        "audio_details",
        "conversation_details",
        "dark_details",
        "phonecharge_details",
        "phonelock_details",
        "gps_details"]

    # Getting the input and generating respective sequences.
    train_feature_dict, val_feature_dict = generate_variables(feature_list=feature_list,
                                                              restrict_seqlen=restrict_seqlen,
                                                              is_cuda_available=torch.cuda.is_available(),
                                                              val_set_size=0.4)

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

    ####################### Val Set ########################

    val_index_list = []
    val_input_list = []
    val_target = None

    for key in val_feature_dict.keys():
        val_input_seq, val_target, val_indices = train_feature_dict[key]
        val_input_list.append(val_input_seq)
        val_index_list.append(val_indices)

    return input_list, input_size_list, index_list, target, val_input_list, val_index_list, val_target
