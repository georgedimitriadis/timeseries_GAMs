
import os
import numpy as np
from collections import OrderedDict

import pandas as pd

import analysis.rankings.new_models_results as nmr
from analysis.utils.funcs_maths import df_scores_to_df_rsquared

data_dir = './talent_benchmark/data'
datasets = os.listdir(data_dir)
logs_dir = './talent_benchmark/logs/2026_01_22_base_data_tuned_models'
the_log_dir = 'ecmac'
logs_dir = os.path.join(logs_dir, the_log_dir)
model_name = 'ecmac'

df_reg, df_bin, df_multi = nmr.fill_dfs_with_model_results(logs_dir=logs_dir, datasets=datasets, model=model_name)
num_models = len(df_reg.columns)

def get_df(data_type):
    if data_type == 'reg':
        df = df_reg
    elif data_type == 'bin':
        df = df_bin
    elif data_type == 'multi':
        df = df_multi
    return df

def get_model_order_in_dataset(model, dataset, data_type, use_rsquared, df):
    a = np.argsort(df.loc[dataset, :].to_numpy())
    ordered_models = df.columns[a].to_numpy()
    if data_type != 'reg' or use_rsquared:
        ordered_models = np.flip(ordered_models)
    pos = np.argwhere(ordered_models == model)[0][0]
    return pos

def get_model_positions_in_all_datasets(model, use_rsquared=False):
    model_positions_over_datasets = {}
    for data_type in ['reg', 'bin', 'multi']:
        keys = []
        values = []
        temp = OrderedDict()
        df = get_df(data_type=data_type)
        if use_rsquared:
            df = df_scores_to_df_rsquared(df, data_dir)
        for dataset in df.index:
            pos = get_model_order_in_dataset(model=model, dataset=dataset, data_type=data_type,
                                             use_rsquared=use_rsquared, df=df)
            keys.append(dataset)
            values.append(pos)
            ordered_values_ind = np.argsort(values)
        for i in ordered_values_ind:
            temp[keys[i]] = values[i]
        model_positions_over_datasets[data_type] = temp
    return model_positions_over_datasets


def get_datasets_where_model_scores_better_than(score, positions_over_datasets, data_type):
    positions = positions_over_datasets[data_type]
    df = get_df(data_type=data_type)
    try:
        datasets_better_than, at_positions = zip(*[(n, positions[n]) for n in df.index if positions[n] <= score])
    except:
        datasets_better_than = []
        at_positions = []
    return datasets_better_than, at_positions

def get_datasets_where_model_scores_worse_than(score, positions_over_datasets, data_type):
    positions = positions_over_datasets[data_type]
    df = get_df(data_type=data_type)
    try:
        datasets_worse_than, at_positions = zip(*[(n, positions[n]) for n in df.index if positions[n] >= num_models - score])
    except:
        datasets_worse_than = []
        at_positions = []
    return datasets_worse_than, at_positions

def get_good_and_bad_datasets_for_model(model, use_rsquared=False, scores_good=(4,2,2), scores_bad=(4,2,2)):
    good_at = {}
    good_positions = {}
    bad_at = {}
    bad_positions = {}
    positions_over_datasets = get_model_positions_in_all_datasets(model, use_rsquared)
    for i, data_type in enumerate(['reg', 'bin', 'multi']):
        good_at[data_type], good_positions[data_type] = get_datasets_where_model_scores_better_than(score=scores_good[i],
                                                                         positions_over_datasets=positions_over_datasets,
                                                                         data_type=data_type)
        bad_at[data_type], bad_positions[data_type] = get_datasets_where_model_scores_worse_than(score=scores_bad[i],
                                                                                                 positions_over_datasets=positions_over_datasets,
                                                                                                 data_type=data_type)

    return good_at, good_positions, bad_at, bad_positions


#df_reg, df_bin, df_multi = nmr.fill_dfs_with_model_results(logs_dir=logs_dir, datasets=datasets, model='ecmac')
df_reg_r2 = df_scores_to_df_rsquared(df_reg, data_dir)
df_bin_r2 = df_scores_to_df_rsquared(df_bin, data_dir)
df_multi_r2 = df_scores_to_df_rsquared(df_multi, data_dir)

positions_over_datasets = get_model_positions_in_all_datasets(model=model_name, use_rsquared=True)

#ecmac_good_at_datasets, ecmac_good_at_positions, ecmac_bad_at_datasets, ecmac_bad_at_positions = \
#    get_good_and_bad_datasets_for_model(model='ecmac', use_rsquared=True,
#                                        scores_good=(4, 1, 1), scores_bad=(4, 1, 1))

def get_df_with_dataset_criteria(df_reg, df_bin, df_multi,
                                 df_reg_r2, df_bin_r2, df_multi_r2,
                                 use_rsquared_for_position_calc = True):

    df = pd.DataFrame(columns=['lin_pos', 'model_pos', 'lgbm_pos',
                               'lin_score', 'model_score', 'lgbm_score',
                               'lin_r2_score', 'model_r2_score', 'lgbm_r2_score',
                               'cat_inputs', 'model_type'],
                      index=np.concat([df_reg.index, df_bin.index, df_multi.index]))
    df.iloc[:len(df_reg.index), np.argwhere(df.columns=='model_type')[0][0]] = 'REG'
    df.iloc[len(df_reg.index):len(df_reg.index)+len(df_bin.index), np.argwhere(df.columns=='model_type')[0][0]] = 'BIN'
    df.iloc[-len(df_multi.index):, np.argwhere(df.columns=='model_type')[0][0]] = 'MULTI'

    df.loc[:, 'model_score'] = pd.concat([df_reg.loc[:, model_name], df_bin.loc[:, model_name], df_multi.loc[:, model_name]])
    df.loc[:, 'lin_score'] = pd.concat([df_reg.loc[:, 'LinearRegression'], df_bin.loc[:, 'LogReg'], df_multi.loc[:, 'LogReg']])
    df.loc[:, 'lgbm_score'] = pd.concat([df_reg.loc[:, 'lightgbm'], df_bin.loc[:, 'lightgbm'], df_multi.loc[:, 'lightgbm']])

    df.loc[:, 'model_r2_score'] = pd.concat([df_reg_r2.loc[:, model_name], df_bin_r2.loc[:, model_name], df_multi_r2.loc[:, model_name]])
    df.loc[:, 'lin_r2_score'] = pd.concat([df_reg_r2.loc[:, 'LinearRegression'], df_bin_r2.loc[:, 'LogReg'], df_multi_r2.loc[:, 'LogReg']])
    df.loc[:, 'lgbm_r2_score'] = pd.concat([df_reg_r2.loc[:, 'lightgbm'], df_bin_r2.loc[:, 'lightgbm'], df_multi_r2.loc[:, 'lightgbm']])

    df_pos_column_names = ['model_pos', 'lin_pos', 'lgbm_pos']
    reg_model_names = [model_name, 'LinearRegression', 'lightgbm']
    non_reg_model_names = [model_name, 'LogReg', 'lightgbm']
    df_reg_used, df_bin_used, df_multi_used = (df_reg, df_bin, df_multi) if not use_rsquared_for_position_calc\
        else (df_reg_r2, df_bin_r2, df_multi_r2)
    for m, df_pos in zip(reg_model_names, df_pos_column_names):
        positions = []  
        for dataset in df_reg.index:
            positions.append(get_model_order_in_dataset(model=m, dataset=dataset, data_type='reg',
                                             use_rsquared=use_rsquared_for_position_calc, df=df_reg_used))
        df.iloc[:len(df_reg.index), np.argwhere(df.columns==df_pos)[0][0]] = positions

    for m, df_pos in zip(non_reg_model_names, df_pos_column_names):
        positions = []
        for dataset in df_bin.index:
            positions.append(get_model_order_in_dataset(model=m, dataset=dataset, data_type='reg',
                                             use_rsquared=use_rsquared_for_position_calc, df=df_bin_used))
        df.iloc[len(df_reg.index):len(df_reg.index)+len(df_bin.index), np.argwhere(df.columns == df_pos)[0][0]] = positions

    for m, df_pos in zip(non_reg_model_names, df_pos_column_names):
        positions = []
        for dataset in df_multi.index:
            positions.append(get_model_order_in_dataset(model=m, dataset=dataset, data_type='reg',
                                             use_rsquared=use_rsquared_for_position_calc, df=df_multi_used))

        df.iloc[-len(df_multi.index):, np.argwhere(df.columns == df_pos)[0][0]] = positions

    for dataset in df.index:
        df.at[dataset, 'cat_inputs'] = True if os.path.exists(os.path.join(data_dir, dataset, 'C_train.npy')) else False

    return df

df_dataset_criteria = get_df_with_dataset_criteria(df_reg, df_bin, df_multi,
                                                   df_reg_r2, df_bin_r2, df_multi_r2,
                                                   True)

df_best_datasets = df_dataset_criteria.loc[(df_dataset_criteria['lgbm_r2_score'] > 0.7) & (df_dataset_criteria['lin_r2_score'] < 0.6) & (df_dataset_criteria['cat_inputs'] == False)]


df_dataset_criteria.to_csv('df_dataset_criteria.csv')

df_dataset_criteria['model_pos'].hist(bins=len(df_reg.columns))