from main.model.baseline_model import StrainBaseline, weights_init
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
          restrict_seqlen=4,
          train=True):
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
    train_input_seq, train_target, val_input_seq, val_target \
        = generate_baseline_variables(feature_list=feature_list)

    # squeezing dimensions in target sets.
    train_target = train_target.squeeze(1)
    val_target = val_target.squeeze(1)

    # Extracting shape of input to initilize network.
    x, y = train_input_seq.shape

    print("Initializing net with {} - y features and {} - x".format(y, x))

    # Initializing Best_Accuracy as 0
    best_accuracy = Variable(torch.from_numpy(np.array([0])).float())

    print("Force-Saving is set to {}".format(force_save_model))

    # declaring Network with number of features.
    net = StrainBaseline(y)
    # Using default learning rate for Adam.
    optimizer = optim.Adam(net.parameters(), weight_decay=0.01)
    net.apply(weights_init)
    val_soft = torch.nn.Softmax(dim=1)

    criterion = nn.CrossEntropyLoss(size_average=True)

    if not train:
        return

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        net = nn.DataParallel(net)

    if torch.cuda.is_available():
        print("Training on GPU")
        net = net.cuda()
        best_accuracy = best_accuracy.cuda()

    if resume_frm_chck_pt:
        model_state, optimizer_state, start_epoch, best_accuracy = chck.load_checkpoint(file_name)
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
        y_hat = net.forward(train_input_seq)
        loss = criterion(y_hat, train_target)

        loss.backward()
        optimizer.step()

        ######################## Validating ########################

        net.eval()
        y_pred = net.forward(val_input_seq)
        y_pred = val_soft(y_pred)

        _, y_pred = y_pred.max(1)
        accuracy = val_target.eq(y_pred).sum()
        accuracy = accuracy.float() / len(val_target)

        if torch.gt(accuracy, best_accuracy).all():
            best_accuracy = accuracy
            is_best = True
        else:
            is_best = False

        # Force save model if flag is true.
        if force_save_model:
            is_best = True

        # generating states. Saving checkpoint after every epoch.
        state = chck.create_state(net, optimizer, epoch, start_epoch, accuracy)

        chck.save_checkpoint(state, is_best, full_file_name=file_name)

        print("=> loss of '{}' at epoch {} \n=> accuracy of {}".format(loss.data[0], start_epoch + epoch + 1, accuracy))

    print("######################################################")
    print("Best Accuracy :", best_accuracy)
    print("######################################################")

    # Initializing Best_Accuracy as 0
    best_accuracy = Variable(torch.from_numpy(np.array([0])).float())

    print("Force-Saving is set to {}".format(force_save_model))


if __name__ == "__main__":
    train(start_epoch=0,
          epochs=1000,
          resume_frm_chck_pt=False,
          force_save_model=True,
          reset_optimizer_state=False,
          train=True)
