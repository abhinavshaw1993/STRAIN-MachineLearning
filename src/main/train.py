from main.LSTM_log_reg_model import Strain, weights_init
from main.GenerateVariables import generate_variables
import main.checkpointing as chck
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch
import numpy as np


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


def train(start_epoch=0, epochs=10, resume_frm_chck_pt=True, force_save_model=False, restrict_seqlen=4):
    # Getting inputs.
    input_list, input_size_list, index_list, target, val_input_list, val_index_list, y_true \
        = get_inputs(restrict_seqlen=restrict_seqlen)

    # Initializing Best_Accuracy as 0
    best_accuracy = Variable(torch.from_numpy(np.array([0])).float())

    print("Force-Saving is set to {}".format(force_save_model))

    if not train:
        return

    # declaring Network.
    net = Strain(input_size_list=input_size_list)
    # optimizer = optim.SGD(net.parameters(), 0.001)
    # Using default learning rate for adadelta.
    optimizer = optim.Adadelta(net.parameters())
    net.apply(weights_init)
    val_soft = torch.nn.Softmax(dim=1)

    criterion = nn.CrossEntropyLoss(size_average=True)

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        net = nn.DataParallel(net)

    if torch.cuda.is_available():
        net = net.cuda()
        best_accuracy = best_accuracy.cuda()

    if resume_frm_chck_pt:
        model_state, optimizer_state, start_epoch, best_accuracy = chck.load_checkpoint()
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    print("#########################################################")
    print("Start Epoch :", start_epoch)

    # Train the network
    for epoch in range(epochs):

        print("###################### Epoch {} #######################".format(start_epoch + epoch + 1))

        net.train(True)
        optimizer.zero_grad()
        y_hat = net.forward(input_list, index_list)
        loss = criterion(y_hat, target)
        loss.backward()
        optimizer.step()

        ######################## Validating ########################

        net.eval()
        y_pred = net.forward(val_input_list, val_index_list)
        y_pred = val_soft(y_pred)

        # y_pred = y_pred.data.numpy().argmax(axis=1)
        # accuracy = accuracy_score(y_true, y_pred)

        _, y_pred = y_pred.max(1)
        accuracy = y_true.eq(y_pred).sum()
        accuracy = accuracy.float() / len(y_true)

        # print(type(accuracy), type(best_accuracy))
        # print("best accuracy {} accuracy {}".format(best_accuracy, accuracy))

        if torch.gt(accuracy, best_accuracy).all():
            best_accuracy = accuracy
            is_best = True
        else:
            is_best = False

        print("Y_pred {}".format(y_pred))
        print("Y_true {}".format(y_true))

        # force save model without it being the best accuracy.
        if force_save_model:
            is_best = True

        # generating states. Saving checkpoint after every epoch.
        state = chck.create_state(net, optimizer, epoch, start_epoch, accuracy)
        chck.save_checkpoint(state, is_best)

        print("=> loss of '{}' at epoch {} \n=> accuracy of {}".format(loss.data[0], start_epoch + epoch + 1, accuracy))

    print("#########################################################")
    print("Best Accuracy :", best_accuracy)
    print("#########################################################")


if __name__ == "__main__":
    train(start_epoch=0, epochs=100, resume_frm_chck_pt=False, force_save_model=False, restrict_seqlen=-1)
