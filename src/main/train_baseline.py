from main.model.lstm_log_reg import Strain, weights_init
from main.utils.generate_baseline_variables import generate_baseline_variables
import main.utils.checkpointing as chck
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch
import numpy as np
import os


def train(start_epoch=0,
          epochs=10,
          resume_frm_chck_pt=True,
          force_save_model=False,
          reset_optimizer_state=False,
          restrict_seqlen=4):

    feature_list = [
    "activity_details",
    # "dinning_details" ,
    "sms_details",
    "audio_details",
    "conversation_details",
    "dark_details",
    "phonecharge_details",
    "phonelock_details",
    "gps_details"]

    # getting current working directory initialize checkpoint file name.
    cwd = os.getcwd()
    file_name = cwd + '/output/baseline.tar'

    # Getting inputs.
    input_list, input_size_list, index_list, target, val_input_list, val_index_list, y_true \
        = get_inputs(restrict_seqlen=restrict_seqlen)

    for idx, input_seq in enumerate(input_list):
        if idx == 0:
            print(type(input_seq))

    # Initializing Best_Accuracy as 0
    best_accuracy = Variable(torch.from_numpy(np.array([0])).float())

    print("Force-Saving is set to {}".format(force_save_model))


if __name__ == "__main__":
    train(start_epoch=0,
          epochs=2,
          resume_frm_chck_pt=True,
          force_save_model=True,
          reset_optimizer_state=False,
          restrict_seqlen=2)
