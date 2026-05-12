
import numpy as np

import analysis.utils.funcs_data_manipulation as fdm

def rsquared_from_rmse(data, rmse):
    rmse = np.array(rmse, dtype=float)
    ss_res = np.power(rmse , 2) * len(data)
    ss_tot = np.sum((data - np.mean(data)) ** 2)
    return 1 - (ss_res / ss_tot)

def mcfadden_r2_from_accuracy(num_predictions, num_of_classes, accuracy):
    assert num_of_classes >= 2, print('For Cohen effect size, num_of_classes should be greater than 1')
    accuracy = np.array(accuracy, dtype=float)
    log_l_model = num_predictions * np.log(accuracy)
    log_l_chance = num_predictions * np.log(1/num_of_classes)

    return 1 - log_l_model/log_l_chance

def r_squared_real_or_pseudo_from_score(data_dir, dataset, scores):
    y_data = fdm.load_y_test_data_from_dataset(data_dir, dataset)
    num_of_classes = fdm.get_number_of_classes(data_dir, dataset)
    num_predictions = len(y_data)

    if num_of_classes >= 2:
        return mcfadden_r2_from_accuracy(num_predictions, num_of_classes, scores)
    else:
        return rsquared_from_rmse(y_data, scores)

def df_scores_to_df_rsquared(df, data_dir):
    df_rsquared = df.copy()
    for dataset in df.index:
        df_rsquared.loc[dataset] = r_squared_real_or_pseudo_from_score(data_dir, dataset, df.loc[dataset])

    return df_rsquared

