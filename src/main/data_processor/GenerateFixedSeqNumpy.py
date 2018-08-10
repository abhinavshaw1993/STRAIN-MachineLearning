import os
import pandas as pd
from datetime import timedelta
from datetime import date as convert_to_date
from sklearn.preprocessing import normalize
from main.definition import ROOT_DIR
import numpy as np


data_dir = ROOT_DIR + "/StudentLife Data"
student_list = os.listdir(data_dir)

student_list = [ _ for _ in student_list if "student" in _]

def adjust_stress_values(stress_level):
    mapping = {
        1: 2,
        2: 3,
        3: 4,
        4: 1,
        5: 0
    }

    try:
        return mapping[stress_level]
    except KeyError:
        return None


for student in student_list:
    print(student)

    files = os.listdir(data_dir + "/{}".format(student))
    files = [_ for _ in files if "train_x.csv" in _]

    for file in files:

        csv_file_name = "{}/{}/{}".format(data_dir, student, file)

        feature_train_x = pd.read_csv("{}/{}/{}".format(data_dir, student, file),
                                      skip_blank_lines=False,
                                      parse_dates=True,
                                      index_col=1
                                      )
        feature_train_x = feature_train_x.iloc[:, 1:]
        resampled_feature_train_x = feature_train_x.resample('2T').max()
        resampled_feature_train_x.iloc[:, :-1] = resampled_feature_train_x.iloc[:, :-1].fillna(method="ffill")

        # Parse Min and Max Date, Convert them to string.
        start_date = resampled_feature_train_x.index.min()
        end_date = resampled_feature_train_x.index.max()
        start_date = start_date.to_datetime()
        end_date = end_date.to_datetime()

        start_date = convert_to_date(start_date.year, start_date.month, start_date.day)
        end_date = convert_to_date(end_date.year, end_date.month, end_date.day) + timedelta(days=1)

        ix = pd.date_range(start=start_date, end=end_date, freq='2T')
        resampled_feature_train_x = resampled_feature_train_x.reindex(ix).iloc[:-1, 1:]

        # Filling NA Values.
        resampled_feature_train_x.iloc[:, :-1] = resampled_feature_train_x.iloc[:, :-1].fillna(method='ffill')
        resampled_feature_train_x.iloc[:, :-1] = resampled_feature_train_x.iloc[:, :-1].fillna(method='bfill')

        unique_dates = list(resampled_feature_train_x.index.map(lambda t: t.date()).unique())

        x = []
        mask = []
        y = []

        for idx, date in enumerate(unique_dates):
            days_train_x = resampled_feature_train_x.loc[str(date): str(date)].iloc[:, :-1]
            days_train_y = resampled_feature_train_x.loc[str(date): str(date)].iloc[:, -1]
            days_train_y = days_train_y.apply(adjust_stress_values)

            days_train_y.reset_index(drop=True, inplace=True)
            days_train_y_index_mask = days_train_y.notnull()
            days_train_y = days_train_y[days_train_y_index_mask]

            days_train_x = days_train_x.as_matrix()
            days_train_y_index_mask = days_train_y_index_mask.as_matrix()
            #             print("Mask Shape:", days_train_y_index_mask.shape)

            # Normalize Days Training Data
            days_train_x = normalize(days_train_x)

            x.append(days_train_x)
            mask.append(days_train_y_index_mask)

            y = y + list(days_train_y)

        # Stacking All the days worth data.
        train_x = np.stack(x, axis=0)
        train_mask = np.stack(mask, axis=0)
        train_mask = train_mask.astype(int)
        train_y = np.array(y)

        #         print("TrainX shape {}, Mask Shape {}".format(train_x.shape, train_mask.shape))

        np.savez("{}/{}/{}".format(data_dir, student, file[:-4]), input_seq=train_x, mask=train_mask, target=train_y)
