
import argparse
import json
import pickle
import os
os.environ['KERAS_BACKEND'] = 'jax'

import numpy as np
from matplotlib import pyplot as plt

from ec.elco import ECRegressor
from ec import spline_responses as sr
from analysis.visualisation import spline_visualisation as sg

def get_model(data_dir, results_model_path, dataset, num_of_points_in_feature = 151):
    #results_model_path = os.path.join('talent_benchmark', 'results_model')
    #data_dir = os.path.join('talent_benchmark', 'data')

    #'3D_Estimation_using_RSSI_of_WLAN_dataset'
    if num_of_points_in_feature % 2 == 0:
        num_of_points_in_feature += 1

    model_path = os.path.join(results_model_path, f'{dataset}-ecmac',
                              'Norm-none-Nan-mean-new-Cat-ordinal', 'best-val-0.pkl')

    N_train_path = os.path.join(data_dir, dataset, 'N_train.npy')
    info_path = os.path.join(data_dir, dataset, 'info.json')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(info_path, 'rb') as f:
        info = json.load(f)

    assert type(model.steps[1][1]) == ECRegressor, print(
        f"Model {dataset} doesn't have an ECRegressor. Cannot create splines for it")
    assert not os.path.isfile(os.path.join(data_dir, dataset, 'C_train.npy')), print(
        f"The dataset {dataset} has categorical data. The visualisation doesn't work for that.")

    x_train_data = np.load(N_train_path)

    num_of_features = x_train_data.shape[1]
    try:
        feature_names = list(info['num_feature_intro'].values())
    except:
        feature_names = [f'X_{i}' for i in np.arange(num_of_features)]

    return model, feature_names, x_train_data

#1D splines for full response
def create_1d_full_response_splines(model, feature_names, num_of_points_in_feature, x_train_data, font_size=30,
                                    linewidth=0.5, save=False):
    num_of_features = len(feature_names)
    unscaled_feature_inputs = sr.unscale_features(model, num_of_features, num_of_points_in_feature)
    plt.rcParams['font.size'] = font_size

    just_inputs_splines, single_feature_spline_components, single_feature_composite_spline =\
        sr.get_single_feature_splines(model, num_of_features, num_of_points_in_feature)

    sg.plot_single_feature_splines(model, feature_names, x_train_data, unscaled_feature_inputs, just_inputs_splines,
                                   single_feature_spline_components, single_feature_composite_spline,
                                   linewidth=linewidth, save=save)

#2D splines for full response with data anchors
def create_2d_full_response_splines(model, feature_names, num_of_points_in_feature, x_train_data, font_size=30,
                                    linewidth=2, save=False, data_size=15):
    num_of_features = len(feature_names)
    unscaled_feature_inputs = sr.unscale_features(model, num_of_features, num_of_points_in_feature)
    plt.rcParams['font.size'] = font_size

    just_inputs_splines, single_feature_spline_components, single_feature_composite_spline = \
        sr.get_single_feature_splines(model, num_of_features, num_of_points_in_feature)
    double_feature_spline_components, double_feature_composite_spline = \
        sr.get_double_feature_splines(model, num_of_features, num_of_points_in_feature)
    sg.plot_all_features_and_combos(feature_names, x_train_data, unscaled_feature_inputs,
                                    single_feature_composite_spline, double_feature_composite_spline, vis_dim='2d',
                                    save=save, linewidth=linewidth, data_size=data_size)

#1D splines for data only
def create_1d_data_based_splines(model, feature_names, x_train_data, font_size=30, linewidth=0.5, save=False):
    plt.rcParams['font.size'] = font_size
    just_inputs_splines, single_feature_spline_components, single_feature_composite_spline = sr.get_single_feature_response_to_existing_data(
        model, x_train_data)
    sg.plot_single_feature_splines(model, feature_names, x_train_data, x_train_data, just_inputs_splines,
                                   single_feature_spline_components, single_feature_composite_spline, from_data=True,
                                   linewidth=linewidth, save=save)
#2D splines for data only
def create_2d_data_based_splines(model, feature_names, x_train_data, font_size=30, linewidth=2, save=False):
    plt.rcParams['font.size'] = font_size
    just_inputs_splines, single_feature_spline_components, single_feature_composite_spline =\
        sr.get_single_feature_response_to_existing_data(model, x_train_data)
    input_x_input_splines, double_feature_spline_components, double_feature_composite_spline = \
        sr.get_double_feature_response_to_existing_data(model, x_train_data)
    sg.plot_all_features_and_combos(feature_names, x_train_data, x_train_data, single_feature_composite_spline,
                                    double_feature_composite_spline, vis_dim='2d', from_data=True, save=save,
                                    linewidth=linewidth)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default='./talent_benchmark/data',
                        help="The folder of the datasets.")
    parser.add_argument("-l", "--results_model_dir", default='./talent_benchmark/results_model',
                        help="The folder of the saved models.")
    parser.add_argument("-s", "--dataset", default='airfoil_self_noise',
                        help="The dataset to visualise.")

    args = parser.parse_args()
    data_dir = args.data_dir
    results_model_path = args.results_model_dir

    sg.SUBPLOT_ADJUST = {'left': 0.03, 'bottom': 0.03, 'right': 0.97, 'top': 0.9, 'wspace': 0.29, 'hspace': 0.26}
    sg.FIG_SIZE = (12, 6)
    sg.FIG_DPI = 150

    num_of_points_in_feature = 151

    dataset = args.dataset # 'airfoil_self_noise' #'concrete_compressive_strength' 'fifa' 'stock'
    model, feature_names, x_train_data = get_model(data_dir, results_model_path, dataset,
                                                   num_of_points_in_feature = num_of_points_in_feature)

    create_1d_full_response_splines(model, feature_names, num_of_points_in_feature, x_train_data, font_size=8, linewidth=1, save=False)
    create_2d_full_response_splines(model, feature_names, num_of_points_in_feature, x_train_data, font_size=15, linewidth=1,
                                    data_size=3, save=False)

if __name__ == "__main__":
    main()

