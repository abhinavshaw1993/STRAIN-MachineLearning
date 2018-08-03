# Accepts a list and returns the Cross Val list for multiple folds.

def get_splits(data_list):

    print(len(data_list[0]))

    for idx, val_data in enumerate(data_list):
        yield data_list[:idx] + data_list[idx+1:], val_data

