"""
This Script Iplements al fucntions to checkpoint PyTorch Models.
"""

import torch
import os
import tarfile


def save_checkpoint(state, is_best, full_file_name):
    # Save the check point if the model it best.

    if not os.path.exists(full_file_name):
      tarfile.open(full_file_name)

    if is_best:
        print("=> Saving a new best")
        torch.save(state, full_file_name)
    else:
        print("=> Validation Accuracy did not improve")


def load_checkpoint(full_file_name):
    print("=> Loading Model from Checkpoint")

    if not os.path.exists(full_file_name):
        print("File {} does not exist".format(full_file_name))
        return

    # If GPU available, load on GPU.
    if torch.cuda.is_available:
        state = torch.load(full_file_name)
    else:
        state = torch.load(full_file_name,
                           map_location=lambda storage, loc: 'cpu')

    keys = state.keys()
    model_state, optimizer_state, start_epoch, best_accuracy = None, None, None, None

    if "model_state" in keys:
        model_state = state['model_state']
    if "optimizer_state" in keys:
        optimizer_state = state['optimizer_state']
    if "epoch" in keys:
        start_epoch = state['epoch']
    if "best_accuracy" in keys:
        best_accuracy = state['best_accuracy']

    return model_state, optimizer_state, start_epoch, best_accuracy


def create_state(model, optimizer, epoch, start_epoch, best_accuracy):
    # Static State list created for, epochs, model_weights, optimizer_state and best_accuracy.
    state = {
        'epoch': epoch + start_epoch + 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_accuracy': best_accuracy
    }

    return state
