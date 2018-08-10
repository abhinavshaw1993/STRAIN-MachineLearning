import main.utils.checkpointing as chck
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from main.model.lstm_log_reg import Strain_log_regression, weights_init
from main.data_getter.get_lstm_log_reg_loso_inputs import get_inputs
from torch.autograd import Variable
from main.utils import kfold
from main.definition import ROOT_DIR
from main.utils.read_config import read_config
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class StudentLifeDataset(Dataset):

    def __init__(self, data_list):
        self.len = len(data_list)
        self.data = data_list

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def train(student_list,
          feature_list,
          start_epoch=0,
          epochs=10,
          resume_frm_chck_pt=True,
          force_save_model=False,
          reset_optimizer_state=False,
          restrict_seqlen=4,
          train_model=True,
          train_split=-1):
    # Getting inputs and extracting the input size list.
    data = get_inputs(student_list=student_list, feature_list=feature_list,restrict_seqlen=restrict_seqlen)
    input_size_list = data[0][1]

    if not train_model:
        print("Not in Train Mode")
        return

    device = torch.device("cuda:0")

    print("Force-Saving is set to {}".format(force_save_model))

    print("######################################################")

    # Trains on all students in the data list.

    print("################# No. of Students: {} #################".format(len(data)))

    for counter, split in enumerate(kfold.get_splits(data)):

        # Train Only specific Model for Loso if not -1.
        if train_split != -1:
            if counter != train_split:
                continue

        print("################# Training Split : {} #################".format(counter))
        train_set_list, val_set = split

        val_score = []

        # Declaring Network.
        net = Strain_log_regression(input_size_list=input_size_list)
        optimizer = optim.Adam(net.parameters(), weight_decay=0.01)
        net.apply(weights_init)
        val_soft = torch.nn.Softmax(dim=1)
        criterion = nn.CrossEntropyLoss(size_average=True)
        best_score = Variable(torch.from_numpy(np.array([0])).float())
        file_name = ROOT_DIR + '/output/loso/loso_model_{}.tar'.format(counter)

        if torch.cuda.device_count() > 1:
            print("Using {} GPUs".format(torch.cuda.device_count()))
            net = nn.DataParallel(net)
            net.to(device)

        if torch.cuda.is_available():
            print("Training on GPU")
            net = net.cuda()
            best_score = best_score.cuda()

        if resume_frm_chck_pt:
            model_state, optimizer_state, start_epoch, best_score = chck.load_checkpoint(file_name)
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

        if reset_optimizer_state:
            optimizer = optim.Adam(net.parameters(), weight_decay=0.01)

        for epoch in range(epochs):

            print("###################### Epoch {} #######################".format(start_epoch + epoch + 1))

            student_life_data_loader = DataLoader(dataset=StudentLifeDataset(train_set_list),
                                     batch_size=1, shuffle=False)

            for input_list, _, index_list, target in student_life_data_loader:

                input_list
                net.train(True)
                optimizer.zero_grad()
                y_hat = net.forward(input_list, index_list)
                loss = criterion(y_hat, target)
                loss.backward()
                optimizer.step()

            ######################## Validating ########################
            val_input_list, _, val_index_list, y_true = val_set
            net.eval()
            y_pred = net.forward(val_input_list, val_index_list)
            y_pred = val_soft(y_pred)

            _, y_pred = y_pred.max(1)
            accuracy = y_true.eq(y_pred).sum()
            accuracy = accuracy.float() / len(y_true)
            # val_score.append(accuracy)

            if torch.gt(accuracy, best_score).all():
                best_score = accuracy
                is_best = True
            else:
                is_best = False

            # force save model without it being the best accuracy.
            if force_save_model:
                is_best = True

            # generating states. Saving checkpoint after every epoch.
            state = chck.create_state(net, optimizer, epoch, start_epoch, best_score)
            chck.save_checkpoint(state, is_best, full_file_name=file_name)

            print("=> loss of '{}' at epoch {} \n=> accuracy of \n {}".format(loss.data[0], start_epoch + epoch + 1,
                                                                              accuracy))

        val_score.append(best_score)

    avg_score = sum(val_score) / len(val_score)

    print("######################################################")
    print("Loso CrossVal Score :", avg_score)
    print("######################################################")


if __name__ == "__main__":
    config = read_config("loso_config.yml")

    train(student_list=config["student_list"],
          feature_list=config["feature_list"],
          start_epoch=config["start_epoch"],
          epochs=config["epochs"],
          resume_frm_chck_pt=config["resume_frm_chck_pt"],
          force_save_model=config["force_save_model"],
          reset_optimizer_state=config["reset_optimizer_state"],
          restrict_seqlen=config["restrict_seqlen"],
          train_model=config["train_model"],
          train_split=config["train_split"])
