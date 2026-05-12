
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from TALENT.model.classical_methods.base import classical_methods
from TALENT.model.lib.data import Dataset, get_dataset


def normalise_data(data):
    num_of_features = data.shape[1]
    norm_data = np.copy(data)

    for i in range(num_of_features):
        norm_data[:, i] = ((data[:, i] - data[:, i].min()) / (data[:, i].max() - data[:, i].min()))

    return norm_data

def get_lengths_of_train_val_test_dataset(data_dir, dataset_name):
    data_path = str(os.path.join(data_dir, dataset_name))

    train_len = len(np.load(os.path.join(data_path, 'N_train.npy')))
    val_len = len(np.load(os.path.join(data_path, 'N_val.npy')))
    test_len = len(np.load(os.path.join(data_path, 'N_test.npy')))

    return train_len, val_len, test_len

def dataset_has_cat_data(dataset_path):
    return os.path.isfile(os.path.join(dataset_path, 'C_train.npy'))

def get_dataset_info(data_dir, dataset_name):
    data_path = str(os.path.join(data_dir, dataset_name))
    info_path = os.path.join(data_path, "info.json")
    with open(info_path, 'rb') as f:
        info = json.load(f)
    return info


def get_number_of_classes(datadir, dataset):
    if 'reg' in get_type_of_dataset(datadir, dataset):
        return 1
    elif 'bin' in get_type_of_dataset(datadir, dataset):
        return 2
    else:
        d = os.path.join(datadir, dataset, 'info.json')
        with open(d, 'r') as f:
            info = json.load(f)
        try:
            num_of_classes = info['num_classes']
        except:
            try:
                num_of_classes = info['n_classes']
            except:
                y_data = load_y_train_data_from_dataset(datadir, dataset)
                num_of_classes = y_data.max() + 1

        return num_of_classes

def get_features_names(data_dir, dataset_name, num_of_features):
    try:
        features_names = list(get_dataset_info(data_dir, dataset_name)['num_feature_intro'].values())
    except:
        features_names =[f'X_{i}' for i in range(num_of_features)]

    return features_names

def get_total_sample_size(data_dir, dataset_name):
    files = os.listdir(os.path.join(data_dir, dataset_name))
    total_samples = 0
    for f in files:
        file_to_load = os.path.join(data_dir, dataset_name, f)
        if 'json' not in f:
            data_array = np.load(file_to_load, allow_pickle=True)
            if 'y_train' in f or 'y_val' in f or 'y_test' in f:
                total_samples += len(data_array)
    return total_samples

def get_feature_numbers(data_dir, dataset_name):
    files = os.listdir(os.path.join(data_dir, dataset_name))
    n_features = 0
    c_features = 0
    for f in files:
        file_to_load = os.path.join(data_dir, dataset_name, f)
        if 'json' not in f:
            data_array = np.load(file_to_load, allow_pickle=True)
            if 'N_train' in f:
                n_features = data_array.shape[1]
            if 'C_train' in f:
                c_features = data_array.shape[1]
    return {'total': n_features + c_features, 'N': n_features, 'C': c_features}

def get_type_of_dataset(data_dir, dataset_name):
    info = get_dataset_info(data_dir, dataset_name)
    if info['task_type'] == 'regression':
        return 'reg'
    elif info['task_type'] == 'multiclass':
        return 'multi'
    elif info['task_type'] == 'binclass':
        return 'bin'

def generate_new_data_split_deprecated(data, random_split_ratio=None, test_indices=None):
    raw_x_data, raw_y_data = data[0], data[1]
    if random_split_ratio is not None:
        X_train, X_test, y_train, y_test = train_test_split(raw_x_data, raw_y_data,
                                                            random_state=104,
                                                            test_size=random_split_ratio,
                                                            shuffle=True)
    else:
        X_test = raw_x_data[test_indices, :]
        y_test = raw_y_data[test_indices, :]
        train_indices = list(set(range(raw_x_data.shape[0])).difference(set(test_indices)))

        X_train = raw_x_data[train_indices, :]
        y_train = raw_y_data[train_indices, :]

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      random_state=104,
                                                      test_size=len(X_test),
                                                      shuffle=True)

    return X_train, X_test, X_val, y_train, y_test, y_val

def generate_new_data_split_from_test_indices(dataset_name, data_dir, test_indices, k_ratio):
    raw_n_data, raw_y_data, raw_c_data = load_nyc_data(data_dir, dataset_name)
    non_test_indices = list(set(np.arange(len(raw_n_data))) - set(test_indices))
    n_test = raw_n_data[test_indices]
    y_test = raw_y_data[test_indices]
    rest_x = raw_n_data[non_test_indices]
    rest_y = raw_y_data[non_test_indices]
    if raw_c_data is not None:
        c_test = raw_c_data[test_indices]
        rest_c = raw_c_data[non_test_indices]
        n_train, n_val, c_train, c_val, y_train, y_val = train_test_split(rest_x, rest_c, rest_y,
                                                                          train_size=k_ratio,
                                                                          random_state=0)
    else:
        c_test = None
        c_train = None
        c_val = None
        n_train, n_val, y_train, y_val = train_test_split(rest_x, rest_y,
                                                          train_size=k_ratio,
                                                          random_state=0)

    return n_train, n_val, n_test, c_train, c_val, c_test, y_train, y_val, y_test

def generate_new_data_split_from_train_val_test_indices(dataset_name, data_dir, train_indices, val_indices, test_indices):
    raw_n_data, raw_y_data, raw_c_data = load_nyc_data(data_dir, dataset_name)
    n_test = raw_n_data[test_indices]
    y_test = raw_y_data[test_indices]
    n_train = raw_n_data[train_indices]
    y_train = raw_y_data[train_indices]
    n_val = raw_n_data[val_indices]
    y_val = raw_y_data[val_indices]
    if raw_c_data is not None:
        c_test = raw_c_data[test_indices]
        c_train = raw_c_data[train_indices]
        c_val = raw_c_data[val_indices]
    else:
        c_test = None
        c_train = None
        c_val = None

    return n_train, n_val, n_test, c_train, c_val, c_test, y_train, y_val, y_test

def generate_trainval_test_structure_from_individual_datasets(train_n_data, val_n_data, test_n_data,
                                                                       train_c_data, val_c_data, test_c_data,
                                                                       train_y_data, val_y_data, test_y_data):

    N = None if train_n_data is None else {'train': train_n_data, 'val':val_n_data, 'test': test_n_data}
    C = None if train_c_data is None else {'train': train_c_data, 'val': val_c_data, 'test': test_c_data}
    y = None if train_y_data is None else {'train': train_y_data, 'val': val_y_data, 'test': test_y_data}

    N_trainval = None if N is None else {key: N[key] for key in
                                         ["train", "val"]} if "train" in N and "val" in N else None
    N_test = None if N is None else {key: N[key] for key in ["test"]} if "test" in N else None

    C_trainval = None if C is None else {key: C[key] for key in
                                         ["train", "val"]} if "train" in C and "val" in C else None
    C_test = None if C is None else {key: C[key] for key in ["test"]} if "test" in C else None

    y_trainval = {key: y[key] for key in ["train", "val"]}
    y_test = {key: y[key] for key in ["test"]}

    # tune hyper-parameters
    train_val_data = (N_trainval, C_trainval, y_trainval)
    test_data = (N_test, C_test, y_test)

    return train_val_data, test_data


def save_new_dataset_deprecated(data, info, new_data_dir, new_dataset_name):
    X_train, X_test, X_val, y_train, y_test, y_val = data

    if not os.path.isdir(os.path.join(new_data_dir, new_dataset_name)):
        os.makedirs(os.path.join(new_data_dir, new_dataset_name))

    np.save(os.path.join(new_data_dir, new_dataset_name, 'N_train.npy'), X_train)
    np.save(os.path.join(new_data_dir, new_dataset_name, 'N_test.npy'), X_test)
    np.save(os.path.join(new_data_dir, new_dataset_name, 'N_val.npy'), X_val)
    np.save(os.path.join(new_data_dir, new_dataset_name, 'y_train.npy'), y_train)
    np.save(os.path.join(new_data_dir, new_dataset_name, 'y_test.npy'), y_test)
    np.save(os.path.join(new_data_dir, new_dataset_name, 'y_val.npy'), y_val)
    with open(os.path.join(new_data_dir, new_dataset_name, 'info.json'), 'w') as f:
        json.dump(info, f, indent=4)

def transfer_info_json(dataset_source_path, dataset_target_path, iteration=None):
    with open(os.path.join(dataset_source_path, 'info.json'), 'r') as f:
        info = json.load(f)
    if iteration is not None:
        info["iteration"] = iteration
    with open(os.path.join(dataset_target_path, 'info.json'), 'w') as f:
        json.dump(info, f)

def categorise_datasets(data_dir):
    datasets = os.listdir(data_dir)

    datasets_separated = {'reg': [], 'bin': [], 'multi': []}
    for d in datasets:
        with open(os.path.join(data_dir, d, 'info.json')) as f:
            info = json.load(f)
            if info['task_type'] == 'regression':
                datasets_separated['reg'].append(d)
            elif info['task_type'] == 'multiclass':
                datasets_separated['multi'].append(d)
            elif info['task_type'] == 'binclass':
                datasets_separated['bin'].append(d)

    return datasets_separated

def drop_missing_data(n_data, y_data, c_data):
    df_x = pd.DataFrame(n_data)
    df_x = df_x.dropna()
    df_y = pd.DataFrame(y_data)
    df_y = df_y.iloc[df_x.index]
    if c_data is not None:
        df_c = pd.DataFrame(c_data)
        df_c = df_c.iloc[df_x.index]
        c_data = df_c.to_numpy()
    return df_x.to_numpy(), df_y.to_numpy(), c_data

class Args:
    def __init__(self, base_config):
        for c in base_config:
            self.__setattr__(c, base_config[c])
        self.config = base_config
        self.config['fit']={'n_bins': base_config['n_bins']}
        self.__setattr__('seed', 0)
        self.__setattr__('num_new_value', None)
        self.__setattr__('cat_new_value', None)
        self.__setattr__('imputer', None)

def create_basic_method(is_regression, y_train_data_mean, y_train_data_std):
    with open(os.path.join('talent_benchmark', 'TALENT', 'configs', 'classical_configs.json'), 'r') as f:
        base_config = json.load(f)

    args = Args(base_config)

    # Create the ECMACMethod object (with the saved model)
    method = classical_methods(args=args, is_regression=is_regression)
    method.num_new_value = None
    method.cat_new_value = None
    method.imputer = method.args.imputer
    method.y_info = {'policy': 'mean_std', 'mean': y_train_data_mean,
                        'std': y_train_data_std}
    method.label_encoder = None
    method.n_bins = method.args.config['n_bins']
    method.num_encoder = None
    method.ord_encoder = None
    method.mode_values = None
    method.cat_encoder = None
    method.normalizer = method.args.normalization

    return method

def load_y_train_data_from_dataset(data_dir, dataset):
    y_file = os.path.join(data_dir, dataset, 'y_train.npy')
    y_train_data = np.load(y_file, allow_pickle=True)
    return y_train_data

def load_y_val_data_from_dataset(data_dir, dataset):
    y_file = os.path.join(data_dir, dataset, 'y_val.npy')
    y_val_data = np.load(y_file, allow_pickle=True)
    return y_val_data

def load_y_test_data_from_dataset(data_dir, dataset):
    y_file = os.path.join(data_dir, dataset, 'y_test.npy')
    y_test_data = np.load(y_file, allow_pickle=True)
    return y_test_data

def load_ncy_train_val_test_data(data_dir, dataset_name):
    data_path = str(os.path.join(data_dir, dataset_name))
    c_data_in_dataset = dataset_has_cat_data(data_path)

    n_train = np.load(os.path.join(data_path, 'N_train.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(data_path, 'y_train.npy'), allow_pickle=True)
    c_train = None if not c_data_in_dataset else np.load(os.path.join(data_path, 'C_train.npy'), allow_pickle=True)
    n_val = np.load(os.path.join(data_path, 'N_val.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(data_path, 'y_val.npy'), allow_pickle=True)
    c_val = None if not c_data_in_dataset else np.load(os.path.join(data_path, 'C_val.npy'), allow_pickle=True)
    n_test = np.load(os.path.join(data_path, 'N_test.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(data_path, 'y_test.npy'), allow_pickle=True)
    c_test = None if not c_data_in_dataset else np.load(os.path.join(data_path, 'C_test.npy'), allow_pickle=True)

    return n_train, n_val, n_test, c_train, c_val, c_test, y_train, y_val, y_test

def load_nyc_data(data_dir, dataset_name):
    """
    Loads from disk and joins together the train, val and test data (in that order) for the N, y and C data sets.
    :param data_dir: The folder where the dataset folder is in
    :param dataset_name: The dataset folder name
    :return: The N, y and C data
    """
    data_path = str(os.path.join(data_dir, dataset_name))
    c_data_in_dataset = dataset_has_cat_data(data_path)

    raw_n_data = np.load(os.path.join(data_path, 'N_train.npy'), allow_pickle=True)
    raw_y_data = np.load(os.path.join(data_path, 'y_train.npy'), allow_pickle=True)
    raw_c_data = None if not c_data_in_dataset else np.load(os.path.join(data_path, 'C_train.npy'), allow_pickle=True)
    for datatype in ['val', 'test']:
        raw_n_data = np.concatenate((raw_n_data, np.load(os.path.join(data_path, f'N_{datatype}.npy'), allow_pickle=True)), axis=0)
        raw_y_data = np.concatenate((raw_y_data, np.load(os.path.join(data_path, f'y_{datatype}.npy'), allow_pickle=True)), axis=0)
        if c_data_in_dataset:
            raw_c_data = np.concatenate((raw_c_data, np.load(os.path.join(data_path, f'C_{datatype}.npy'), allow_pickle=True)), axis=0)
    return raw_n_data, raw_y_data.reshape(-1, 1), raw_c_data

def load_preprocessed_xy_data(dataset, data_dir):
    """
    It concatenates a dataset's train, val and test data (in that order) into X_data and y_data sets.
    It also uses the TALENT model's basic functionality to turn any categorical variables into numerical ones and to deal with
    the missing values in the same way that the basic model deals with these two issues (following the default config).
    :param dataset: The name of the dataset
    :param data_dir: The directory where the dataset is
    :return: x_data, y_data arrays
    """

    train_val_data, test_data, info = get_dataset(dataset, data_dir)

    basic_method = create_basic_method(is_regression=info['task_type'] == 'regression',
                                       y_train_data_mean=train_val_data[2]['train'].mean(),
                                       y_train_data_std=train_val_data[2]['train'].std())

    # Put the test data together with the train data so they get preprocessed
    N, C, y = train_val_data
    train_length = len(N['train'])
    train_slice = slice(0, train_length, 1)
    N['train'] = np.concatenate((N['train'], test_data[0]['test']))
    y['train'] = np.concatenate((y['train'], test_data[2]['test']))
    if C is not None:
        C['train'] = np.concatenate((C['train'], test_data[1]['test']))
    test_slice = slice(train_length, len(N['train']), 1)

    basic_method.D = Dataset(N, C, y, info)
    basic_method.N, basic_method.C, basic_method.y = basic_method.D.N, basic_method.D.C, basic_method.D.y
    basic_method.is_binclass, basic_method.is_multiclass, basic_method.is_regression = basic_method.D.is_binclass, basic_method.D.is_multiclass, basic_method.D.is_regression
    basic_method.data_format(is_train = True, y=y)

    # Concatenate the val data together with the train+test data (now everything is properly preprocessed)
    x_data = np.concatenate((basic_method.N['train'][train_slice], basic_method.N['val'], basic_method.N['train'][test_slice]))
    y_data = np.concatenate((basic_method.y['train'][train_slice], basic_method.y['val'], basic_method.y['train'][test_slice]))

    return x_data, y_data

def save_data(data_arrays, data_names, data_path):
    for d, n in list(zip(data_arrays, data_names)):
        if d is not None:
            np.save(os.path.join(data_path, f'{n}.npy'), d)

