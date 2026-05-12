
import os
import numpy as np
from src.analysis.utils import funcs_knn_ood_data_generation as ogkf, funcs_data_manipulation as dmf
from src.analysis.visualisation import data_visualisation_funcs as dvf


data_dir_ood = os.path.join('talent_benchmark', 'data_ood')
data_dir = os.path.join('talent_benchmark', 'data')
datasets = ogkf.get_datasets_with_only_numerical_data(data_dir, remove_data_with_nans=True)
#datasets = os.listdir(data_dir_ood)

dataset = datasets[1]
features_combination = 0
center_point_index = 2

data_dir = os.path.join(data_dir_ood, dataset)
dataset_name = f'{dataset}__f_{features_combination}__c_{center_point_index}'
info = dmf.get_dataset_info(data_dir_ood, dataset_name)
features_names = [name for name in info['num_feature_intro']]
x_train = np.load(os.path.join(data_dir_ood, dataset_name,'N_train.npy'))
x_test = np.load(os.path.join(data_dir_ood, dataset_name,'N_test.npy'))
x_val = np.load(os.path.join(data_dir_ood, dataset_name,'N_val.npy'))
dvf.show_2d_combinations([x_train, x_val, x_test], features_names)


num_of_ood_subsets = []
for dataset in datasets:
    print(f'{dataset}__f')
    num_of_ood_subsets.append(len([i for i in os.listdir(data_dir_ood) if f'{dataset}__f' in i]))



