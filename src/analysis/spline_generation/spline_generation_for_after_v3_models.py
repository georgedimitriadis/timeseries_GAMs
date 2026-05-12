
import argparse
import json
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from ec.elco import ECRegressor
from ec import spline_responses as spl_res
from analysis.visualisation import spline_visualisation as spl_vis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_path", default='ecmac_2025_11_27_ood_v1',
                        help="The folder inside the results_model folder that has the models.")
    parser.add_argument("-p", "--data_path", default='data_ood',
                        help="The base folder of the data folders in the talent_benchmark folder.")
    parser.add_argument("-d", "--dataset", default='f_8__concrete_compressive_strength__c_0',
                        help="The name of the dataset to be used. This needs to be a folder in the data_path folder.")
    parser.add_argument("-n", "--num_response_points", default=151,
                        help="The number of x points to generate the response of all splines from -1 to 1. Make it an odd number.")
    parser.add_argument("-m", "--model_name", default='ecmac',
                        help="The name of the model that was used to generate the data.")
    args = parser.parse_args()

    results_model_path = os.path.join('talent_benchmark', 'results_model')
    experiment_path = args.experiment_path
    data_dir = os.path.join('talent_benchmark', args.data_path)

    dataset = args.dataset
    num_of_points_in_feature = int(args.num_response_points)
    if num_of_points_in_feature % 2 == 0:
        num_of_points_in_feature += 1

    model_path = os.path.join(results_model_path, experiment_path, f'{dataset}-{args.model_name}',
                              'Norm-none-Nan-mean-new-Cat-ordinal', 'best-val-0.pkl')

    N_train_path = os.path.join(data_dir, dataset, 'N_train.npy')
    info_path = os.path.join(data_dir, dataset, 'info.json')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(info_path, 'rb') as f:
        info = json.load(f)

    x_train_data = np.load(N_train_path)

    num_of_features = x_train_data.shape[1]
    try:
        feature_names = list(info['num_feature_intro'].values())
    except:
        feature_names = [f'X_{i}' for i in np.arange(num_of_features)]

    assert type(model.steps[1][1]) == ECRegressor, print(
        f"Model {dataset} doesn't have an ECRegressor. Cannot create splines for it")
    assert not os.path.isfile(os.path.join(data_dir, dataset, 'C_train.npy')), print(
        f"The dataset {dataset} has categorical data. The visualisation doesn't work for that.")

    print('Creating response over the whole -1 to 1 range for 1D splines')
    unscaled_feature_inputs = spl_res.unscale_features(model, num_of_features, num_of_points_in_feature)
    just_inputs_splines, single_feature_spline_components, single_feature_composite_spline = spl_res.get_single_feature_splines(
        model, num_of_features, num_of_points_in_feature)
    # spl_vis.plot_single_feature_splines(model, feature_names, x_train_data, unscaled_feature_inputs,
    #                           just_inputs_splines, single_feature_spline_components, single_feature_composite_spline)

    print('Creating response over the whole -1 to 1 range for 2D splines')
    double_feature_spline_components, double_feature_composite_spline = spl_res.get_double_feature_splines(model,
                                                                                                      num_of_features,
                                                                                                      num_of_points_in_feature)
    print('Plotting 2D splines')
    plt.rcParams['font.size'] = 4
    spl_vis.plot_all_features_and_combos(feature_names, x_train_data, unscaled_feature_inputs,
                                         single_feature_composite_spline, double_feature_composite_spline, vis_dim='2d')

    print('Creating response only at the collected data points for 1D splines')
    just_inputs_splines, single_feature_spline_components, single_feature_composite_spline = spl_res.get_single_feature_response_to_existing_data(
        model, x_train_data)
    print('Plotting 1D splines')
    plt.rcParams['font.size'] = 10
    spl_vis.plot_single_feature_splines(model, feature_names, x_train_data, x_train_data, just_inputs_splines,
                                        single_feature_spline_components, single_feature_composite_spline,
                                        from_data=True)
    # print(4)
    # 2D splines
    # double_feature_spline_components, double_feature_composite_spline = spl_res.get_double_feature_response_to_existing_data(model, feature_names, x_train_data)
    # spl_vis.plot_all_features_and_combos(feature_names, x_train_data, x_train_data, single_feature_composite_spline,
    # double_feature_composite_spline, vis_dim='2d', from_data=True)


#TODO: Create arguments that do specific jobs (like only 1D splines generation, or only plotting)
if __name__ == "__main__":
    main()