from main.model.lstm_linear_reg import Strain_linear_reg, weights_init
from main.data_getter.get_lstm_log_reg_inputs import get_inputs
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
    # getting current working directory initialize checkpoint file name.
    cwd = os.getcwd()
    file_name = cwd + '/output/linear_reg.tar'

    # Getting inputs.
    input_list, input_size_list, index_list, target, val_input_list, val_index_list, y_true \
        = get_inputs(restrict_seqlen=restrict_seqlen, standardize=True)

    target = target.float()
    y_true = y_true.float()

    # Initializing Best_mse as 0
    best_mse = Variable(torch.from_numpy(np.array([100000000000])).float())

    print("Force-Saving is set to {}".format(force_save_model))

    if not train:
        return

    # declaring Network.
    net = Strain_linear_reg(input_size_list=input_size_list)
    # Using default learning rate for Adam.
    optimizer = optim.Adam(net.parameters(), weight_decay=0.01)
    net.apply(weights_init)
    criterion = nn.MSELoss()

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        net = nn.DataParallel(net)

    if torch.cuda.is_available():
        print("Training on GPU")
        net = net.cuda()
        best_mse = best_mse.cuda()

    if resume_frm_chck_pt:
        model_state, optimizer_state, start_epoch, best_mse = chck.load_checkpoint(file_name)
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    if reset_optimizer_state:
        optimizer = optim.Adam(net.parameters(), weight_decay=0.01)

    print("######################################################")
    print("Start Epoch :", start_epoch)

    # Train the network
    for epoch in range(epochs):

        print("###################### Epoch {} #######################".format(start_epoch + epoch + 1))

        net.train(True)
        optimizer.zero_grad()
        y_hat = net.forward(input_list, index_list)
        loss = criterion(y_hat, target)
        print("working till here")
        loss.backward()
        optimizer.step()

        ######################## Validating ########################

        net.eval()

        y_pred = net.forward(val_input_list, val_index_list)
        mse = criterion(y_pred, y_true)

        if torch.lt(mse, best_mse).all():
            best_mse = mse
            is_best = True
        else:
            is_best = False

        print("Y_pred {}".format(y_pred))
        print("Y_true {}".format(y_true))

        # force save model without it being the best MSE.
        if force_save_model:
            is_best = True

        # generating states. Saving checkpoint after every epoch.
        state = chck.create_state(net, optimizer, epoch, start_epoch, mse)

        chck.save_checkpoint(state, is_best, full_file_name=file_name)

        print("=> loss of '{}' at epoch {} \n=> mse of {}".format(loss.data[0], start_epoch + epoch + 1, mse))

    print("######################################################")
    print("Best MSE :", best_mse)
    print("######################################################")


if __name__ == "__main__":
    train(start_epoch=0,
          epochs=500,
          resume_frm_chck_pt=False,
          force_save_model=False,
          reset_optimizer_state=False,
          restrict_seqlen=-1)
