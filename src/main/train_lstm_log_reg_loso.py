from main.model.lstm_log_reg import Strain_log_regression, weights_init
from main.utils.get_lstm_log_reg_loso_inputs import get_inputs
from torch.autograd import Variable
from main.utils import kfold
import main.utils.checkpointing as chck
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from main.definition import ROOT_DIR


def train(start_epoch=0,
          epochs=10,
          resume_frm_chck_pt=True,
          force_save_model=False,
          reset_optimizer_state=False,
          restrict_seqlen=4):
    # getting current working directory initialize checkpoint file name.

    file_name = ROOT_DIR + '/output/log_reg_loso.tar'

    # Getting inputs and extracting the input size list.
    data = get_inputs(restrict_seqlen=restrict_seqlen)
    input_size_list = data[0][1]

    # Initializing Best_Accuracy as 0
    best_accuracy = Variable(torch.from_numpy(np.array([0])).float())

    print("Force-Saving is set to {}".format(force_save_model))

    if not train:
        return

    # declaring Network.
    net = Strain_log_regression(input_size_list=input_size_list)
    # Using default learning rate for Adam.
    optimizer = optim.Adam(net.parameters(), weight_decay=0.01)
    net.apply(weights_init)
    val_soft = torch.nn.Softmax(dim=1)

    criterion = nn.CrossEntropyLoss(size_average=True)

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        net = nn.DataParallel(net)

    if torch.cuda.is_available():
        print("Training on GPU")
        net = net.cuda()
        best_accuracy = best_accuracy.cuda()

    if resume_frm_chck_pt:
        model_state, optimizer_state, start_epoch, best_score = chck.load_checkpoint(file_name)
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    if reset_optimizer_state:
        optimizer = optim.Adam(net.parameters(), weight_decay=0.01)

    print("######################################################")
    print("Start Epoch :", start_epoch)

    # Train the network
    for epoch in range(epochs):
        val_score = []

        print("###################### Epoch {} #######################".format(start_epoch + epoch + 1))

        # Trains on all students in the data list.

        for train_set_list, val_set in kfold.get_splits(data):
            for input_list, _, index_list, target in train_set_list:
                net.train(True)
                optimizer.zero_grad()
                y_hat = net.forward(input_list, index_list)
                loss = criterion(y_hat, target)
                loss.backward()
                optimizer.step()

            ######################## Validating ########################
            val_input_list, val_index_list, y_true = val_set
            net.eval()
            y_pred = net.forward(val_input_list, val_index_list)
            y_pred = val_soft(y_pred)

            _, y_pred = y_pred.max(1)
            accuracy = y_true.eq(y_pred).sum()
            accuracy = accuracy.float() / len(y_true)
            val_score.append(accuracy)

        avg_score = val_score.sum() / len(val_score)

        if torch.gt(avg_score, best_score).all():
            best_score = avg_score
            is_best = True
        else:
            is_best = False

        # force save model without it being the best accuracy.
        if force_save_model:
            is_best = True

        # generating states. Saving checkpoint after every epoch.
        state = chck.create_state(net, optimizer, epoch, start_epoch, best_score)

        chck.save_checkpoint(state, is_best, full_file_name=file_name)

        print("=> loss of '{}' at epoch {} \n=> accuracy of {}".format(loss.data[0], start_epoch + epoch + 1, accuracy))

    print("######################################################")
    print("Loso CrossVal Score :", best_score)
    print("######################################################")


if __name__ == "__main__":
    train(start_epoch=0,
          epochs=1,
          resume_frm_chck_pt=False,
          force_save_model=False,
          reset_optimizer_state=False,
          restrict_seqlen=-1)
