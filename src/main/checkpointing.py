"""
This Script Iplements al fucntions to checkpoint PyTorch Models.
"""

import torch
import os


def save_checkpoint(state, is_best, file_name_rel='/output/checkpoint.pth.tar'):
    # Save the check point if the model it best.
    cwd = os.getcwd()
    file_name_rel = cwd+file_name_rel

    if is_best:
        print("=> Saving a new best")
        torch.save(state, file_name_rel)
    else:
        print("=> Validation Accuracy did not improve")

def create_state(model, optimizer, epoch, start_epoch, best_accuracy):
    # Static State list created for, epochs, model_weights, optimizer_state and best_accuracy.
    state = {
    'epoch':epoch + start_epoch + 1,
    'model_state':model.state_dict(),
    'optimizer_state':optimizer.state_dict(),
    'best_accuracy':best_accuracy
    }

    return state


def load_checkpoint(file_name_rel='/output/checkpoint.pth.tar'):
    print("=> Loading Model from Checkpoint")
    cwd = os.getcwd()
    file_name_rel = cwd+file_name_rel

    # If GPU available, load on GPU.
    if torch.cuda.is_available:
        state = torch.load(file_name_rel)
    else:
        state = torch.load(file_name_rel,
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



