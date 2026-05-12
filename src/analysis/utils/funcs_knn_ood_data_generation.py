
import os
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from analysis.utils import funcs_data_manipulation as dmf


def get_datasets_with_only_numerical_data(data_dir, remove_data_with_nans=False):
    datasets = os.listdir(data_dir)
    numerical_datasets = []
    for dataset in datasets:
        if not os.path.isfile(os.path.join(data_dir, dataset, 'C_train.npy')):
            raw_x_data, raw_y_data, _ = dmf.load_nyc_data(data_dir, dataset)
            has_nan = np.any((np.isnan(raw_x_data)))
            norm_x_data = dmf.normalise_data(raw_x_data)
            has_norm_nans = np.any((np.isnan(norm_x_data)))
            if not(remove_data_with_nans and (has_nan or has_norm_nans)):
                numerical_datasets.append(dataset)
    return numerical_datasets

def get_non_edge_point_indices(data, max_allowed_num_of_features_at_edge: int = 2):
    set_of_all_data_points = {tuple(dp) for dp in data}
    set_of_all_data_points = list(set_of_all_data_points)

    features_ranges = list(zip(np.min(data, axis=0), np.max(data, axis=0)))

    list_of_non_edge_datapoints_indices = []
    for p, dp in enumerate(set_of_all_data_points):
        num_of_edge_features = 0
        for i, f in enumerate(dp):
            if f == features_ranges[i][0] or f == features_ranges[i][1]:
                num_of_edge_features += 1
        if num_of_edge_features < max_allowed_num_of_features_at_edge + 1:
            list_of_non_edge_datapoints_indices.append(p)

    return list_of_non_edge_datapoints_indices


def get_knn_based_point_groups(data, portion_of_data_to_ood=0.1, for_points_indices=None, num_of_combinations=10):
    min_num_of_features_to_use = 3

    norm_data = dmf.normalise_data(data)
    num_of_features = data.shape[1]

    n_neighbours = int(portion_of_data_to_ood * data.shape[0])

    knn = NearestNeighbors(n_neighbors=n_neighbours)

    if num_of_features < min_num_of_features_to_use + 3:
        min_num_of_features_to_use = num_of_features - 2

    if num_of_combinations - 1 > num_of_features - min_num_of_features_to_use - 1:
        num_of_combinations = num_of_features - min_num_of_features_to_use

    features_to_use = range(num_of_features)
    num_of_features_to_delete =  np.random.choice(np.arange(1, num_of_features - min_num_of_features_to_use), size=num_of_combinations - 1, replace=False).astype(int).tolist()
    num_of_features_to_delete.append(0)

    features_groups_lists = {}
    for f in num_of_features_to_delete:
        features_groups_lists[f] = []
        feature_comb = np.random.choice(features_to_use, size=len(features_to_use) - f, replace=False)
        features_groups_lists[f] = feature_comb

    possible_kneighbors_groups = {k: dict() for k in num_of_features_to_delete}
    features_used = {}
    for f in tqdm(num_of_features_to_delete, desc='Num features to delete'): # num_of_features - f is the number of features to use to create feature combinations
        features_used[f] = {}
        feature_comb = tuple([int(i) for i in features_groups_lists[f]])
        knn.fit(norm_data[:, feature_comb]) # do a knn fit to the non-edge data with the selected features
        features_used[f][feature_comb] = []
        possible_kneighbors_groups[f][feature_comb] = []
        if for_points_indices is None:
            for p_idx, p in enumerate(norm_data): # for each data point index
                p = p[feature_comb] # get the datapoint with the correct features
                possible_kneighbors_groups[f][feature_comb].append(knn.kneighbors(p.reshape(1, -1), return_distance=False)[0]) # get the knns and add them to the set
                features_used[f][feature_comb].append((p_idx, p))
        else:
            for p_idx in for_points_indices:
                p = norm_data[p_idx, feature_comb]  # get the datapoint with the correct features
                possible_kneighbors_groups[f][feature_comb].append(knn.kneighbors(p.reshape(1, -1), return_distance=False)[0])  # get the knns and add them to the set
                features_used[f][feature_comb].append((p_idx, p))

    return possible_kneighbors_groups, features_used

def find_point_closer_to_mean(data):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    # Find centroid in scaled space
    centroid_scaled = np.mean(scaled, axis=0).reshape(1, -1)

    # Find closest point
    distances = cdist(scaled, centroid_scaled, metric='euclidean').flatten()
    most_central_idx = distances.argmin()

    return most_central_idx