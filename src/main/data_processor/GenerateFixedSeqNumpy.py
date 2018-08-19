from datetime import timedelta
from datetime import date as convert_to_date
from sklearn.preprocessing import StandardScaler
from main.definition import ROOT_DIR
import os
import main.data_processor.agg_utils as  agg_func
import copy
import pandas as pd
import numpy as np

resample_freq_min = 2
print("Data points per sequence: ", int(24*60/resample_freq_min))

model_aggregates = True
aggre_list = [agg_func.linear_fit, agg_func.poly_fit, agg_func.mcr]
data_dir = ROOT_DIR + "/StudentLife Data"
student_list = os.listdir(data_dir)
student_list = [_ for _ in student_list if "student" in _]

# Making Sure Student 1 is first in the list.
student_list.remove("student 1")
student_list.insert(0, "student 1")
transformer = {}

for student in student_list:
    print(student)

    files = os.listdir(data_dir + "/{}".format(student))
    files = [_ for _ in files if "train_x.csv" in _]
    files.sort()

    for file in files:

        csv_file_name = "{}/{}/{}".format(data_dir, student, file)
        # print("{}/{}/{}".format(data_dir, student, file))
        feature_train_x = pd.read_csv("{}/{}/{}".format(data_dir, student, file),
                                      skip_blank_lines=False,
                                      parse_dates=True,
                                      index_col=1
                                      )
        feature_train_x = feature_train_x.iloc[:, 1:]
        resampled_feature_train_x = feature_train_x.resample('{}T'.format(resample_freq_min)).max()
        resampled_feature_train_x.iloc[:, :-1] = resampled_feature_train_x.iloc[:, :-1].fillna(method="ffill")

        # Parse Min and Max Date, Convert them to string.
        start_date = resampled_feature_train_x.index.min()
        end_date = resampled_feature_train_x.index.max()
        start_date = start_date.to_pydatetime()
        end_date = end_date.to_pydatetime()

        start_date = convert_to_date(start_date.year, start_date.month, start_date.day)
        end_date = convert_to_date(end_date.year, end_date.month, end_date.day) + timedelta(days=1)

        ix = pd.date_range(start=start_date, end=end_date, freq='{}T'.format(resample_freq_min))
        resampled_feature_train_x = resampled_feature_train_x.reindex(ix).iloc[:-1, 1:]

        # Filling NA Values.
        # resampled_feature_train_x.iloc[:, :-1] = resampled_feature_train_x.iloc[:, :-1].fillna(method='ffill')
        resampled_feature_train_x.iloc[:, :-1] = resampled_feature_train_x.iloc[:, :-1].fillna(value=-10)

        # print("Type: ", type(resampled_feature_train_x.index.map(lambda t: t.date())))

        unique_dates = list(resampled_feature_train_x.index.map(lambda t: t.date()).unique())

        x = []
        mask = []
        y = []

        if student == "student 1":

            # Set Stdev and mean for scalar.
            transformer[file[:-4]] = copy.deepcopy(StandardScaler())
            temp_data = resampled_feature_train_x.iloc[:, :-1]
            df_x, df_y = temp_data.shape
            dow = pd.DataFrame(temp_data.index.dayofweek)

            # Model Aggregates if on.
            if model_aggregates:
                aggregates = []

                for i in range(df_y):
                    return_values = temp_data.iloc[:, i].apply(aggre_list, axis=0)
                    aggregates = aggregates + np.hstack(return_values.values).tolist()

                aggregates = np.array([aggregates,]*len(temp_data))
                temp_data = np.concatenate([temp_data.as_matrix(), aggregates], axis=1)

            date_converter = convert_to_date.today().weekday()
            temp_data = np.concatenate([temp_data, dow], axis=1)
            transformer[file[:-4]].fit(temp_data)

        for idx, date in enumerate(unique_dates):
            days_train_x = resampled_feature_train_x.loc[str(date): str(date)].iloc[:, :-1]
            df_x, df_y = days_train_x.shape

            if model_aggregates:
                aggregates = []

                for i in range(df_y):
                    return_values = days_train_x.iloc[:, i].apply(aggre_list, axis=0)
                    aggregates = aggregates + np.hstack(return_values.values).tolist()

                aggregates = np.array([aggregates, ]*int(24*60/resample_freq_min))

            days_train_y = resampled_feature_train_x.loc[str(date): str(date)].iloc[:, -1]
            days_train_y = days_train_y.apply(agg_func.adjust_stress_values)
            days_train_y.reset_index(drop=True, inplace=True)
            days_train_y_index_mask = days_train_y.notnull()
            days_train_y = days_train_y[days_train_y_index_mask]

            days_train_x = days_train_x.as_matrix()

            if model_aggregates:
                days_train_x = np.concatenate([days_train_x, aggregates], axis=1)

            dow = np.full((int(24*60/resample_freq_min), 1), date.today().weekday())
            days_train_x = np.concatenate([days_train_x, dow], axis=1)

            # applying custom aggregate functions
            days_train_y_index_mask = days_train_y_index_mask.as_matrix()

            # Normalize Days Training Data
            days_train_x = transformer[file[:-4]].transform(days_train_x)

            x.append(days_train_x)
            mask.append(days_train_y_index_mask)

            y = y + list(days_train_y)

        # Stacking All the days worth data.
        train_x = np.stack(x, axis=0)
        train_mask = np.stack(mask, axis=0)
        train_mask = train_mask.astype(int)
        train_y = np.array(y)

        np.savez("{}/{}/{}".format(data_dir, student, file[:-4]), input_seq=train_x, mask=train_mask, target=train_y)
