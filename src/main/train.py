from main.LSTM_log_reg_model import Strain, weights_init
from main.GenerateVariables import generate_variables
import main.checkpointing as chck
import torch.nn as nn
import torch.optim as optim
import torch


def get_inputs():
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
    train_feature_dict, val_feature_dict = generate_variables(feature_list=feature_list, restrict_seqlen=2,\
                                      is_cuda_available=torch.cuda.is_available())

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


def train(start_epoch=1, epochs=10, resume_frm_chck_pt=True, train=False):
    # Getting inputs.
    input_list, input_size_list, index_list, target, val_input_list, val_index_list, pred_true = get_inputs()

    print(val_index_list[0])
    print(pred_true)

    if not train:
        return

    # declaring Network.
    net = Strain(input_size_list=input_size_list)
    optimizer = optim.SGD(net.parameters(), 0.001)
    net.apply(weights_init)

    criterion = nn.CrossEntropyLoss(size_average=True)

    if resume_frm_chck_pt:
        model_state, optimizer_state, start_epoch, best_accuracy = chck.load_checkpoint()
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Train the network
    for epoch in range(epochs):
        net.train(True)
        optimizer.zero_grad()
        y_hat = net.forward(input_list, index_list)
        loss = criterion(y_hat, target)
        loss.backward()
        optimizer.step()

        net.eval()
        y_pred = net.forward(val_input_list, val_index_list)
        print(y_pred)

        # generating states. Saving checkpoint after every epoch.
        state = chck.create_state(net, optimizer, epoch, start_epoch, None)
        chck.save_checkpoint(state, True)

        print("=> loss of '{}' at epoch {}".format(loss.data[0], start_epoch+epoch+1))


if __name__ == "__main__":
    train(start_epoch=0, epochs=2, train=True)